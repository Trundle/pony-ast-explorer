from __future__ import annotations

import enum
import fcntl
import os
import struct
import sys
import termios
import tty
from contextlib import contextmanager
from curses import setupterm, tigetstr, tparm
from itertools import chain, cycle, islice
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

import gdb

# Wild hack :(
sys.path.append(os.path.dirname(__file__))
from box_printer import (
    Box,
    BoxPrinter,
    Column,
    Decorated,
    Empty,
    HSpace,
    Row,
    Text,
    group,
    frame,
)


T = TypeVar("T")


## Gdb helpers


class AstNode:
    """
    Wrapper around `gdb.Value` to make working with AST nodes a bit more
    convenient.
    """

    def __init__(self, value: gdb.Value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, AstNode):
            return int(self.value) == int(other.value)
        return NotImplemented

    def __hash__(self):
        return int(self.value)

    def __repr__(self):
        return f"<AstNode@{int(self.value):x}(id={self.token_id})>"

    @property
    def token_id(self) -> str:
        return str(self.value["t"]["id"])

    @property
    def child(self) -> Optional[AstNode]:
        if child := self.value["child"]:
            return AstNode(child)

    @property
    def data(self) -> Optional[AstNode]:
        if data := self.value["data"]:
            return AstNode(data.cast(gdb.lookup_type("ast_t").pointer()))

    @property
    def parent(self) -> Optional[AstNode]:
        if parent := self.value["parent"]:
            return AstNode(parent)

    @property
    def sibling(self) -> Optional[AstNode]:
        """
        The node's direct sibling (if there is one).
        """
        if sibling := self.value["sibling"]:
            return AstNode(sibling)

    @property
    def siblings(self) -> Iterable[AstNode]:
        """
        Returns the node itself and all its siblings.
        """
        current_value = self
        while current_value:
            yield current_value
            current_value = current_value.sibling

    def symtab(self) -> Optional[Dict[str, Tuple[int, int]]]:
        if symtab := self.value["symtab"]:
            i_ptr = gdb.parse_and_eval("(void *)malloc(sizeof(size_t))")
            gdb.selected_inferior().write_memory(
                int(i_ptr), b"\xff" * gdb.lookup_type("size_t").sizeof
            )
            result = {}
            while True:
                sym = gdb.parse_and_eval(
                    f"symtab_next(((ast_ptr_t){int(symtab)}), "
                    f"(size_t *){int(i_ptr)})"
                )
                if not sym:
                    break
                result[sym["name"].string()] = (int(sym["def"]), int(sym["status"]))
            return result

    def token_int(self):
        return int(self.value["t"]["integer"]["low"])

    def token_string(self) -> Optional[str]:
        if self.token_id in {"TK_ID", "TK_STRING"} and (
            string := self.value["t"]["string"]
        ):
            return string.string()


def _is_ast(type_: gdb.Type) -> bool:
    if type_.code == gdb.TYPE_CODE_PTR:
        target = type_.target()
        return target.name == "ast_t" or _is_ast(target)
    elif type_.code == gdb.TYPE_CODE_TYPEDEF:
        return type_.name == "ast_ptr_t"
    return False


def _to_ast_ptr(value: gdb.Value) -> gdb.Value:
    if value.type.code == gdb.TYPE_CODE_PTR:
        if value.type.target().name == "ast_t":
            return value
        else:
            return _to_ast_ptr(value.dereference())
    elif value.type.code == gdb.TYPE_CODE_TYPEDEF and value.type.name == "ast_ptr_t":
        return value
    raise ValueError


def _all_asts_in_selected_frame() -> Iterable[Tuple[gdb.Symbol, AstNode]]:
    frame = gdb.selected_frame()
    try:
        block = frame.block()
    except RuntimeError:
        return
    while block and not (block.is_global or block.is_static):
        for symbol in block:
            if symbol.is_variable or symbol.is_argument:
                if _is_ast(symbol.type):
                    ast_ptr = _to_ast_ptr(symbol.value(frame))
                    if not ast_ptr.is_optimized_out:
                        yield (symbol, AstNode(ast_ptr))
        block = block.superblock


@contextmanager
def _real_stdout():
    # Otherwise using print() in Python triggers gdb's pager
    stdout = sys.stdout
    sys.stdout = sys.__stdout__
    try:
        yield
    finally:
        sys.stdout = stdout


## Gdb commands


class AstCommand(enum.Enum):
    exit = "exit AST view"
    focus_next = "focus next node"
    focus_prev = "focus previous node"
    print_ast = "dump selected node with ast_print"
    print_verbose = "dump selected node with ast_printverbose"
    less_details = "skip some child nodes"
    more_details = "show more child nodes"
    jump_to_data = "jump to data"
    back = "return from last jump"
    zoom_out = "zoom out"
    zoom_in = "zoom into selected node"


class PonyAstCommand(gdb.Command):
    KEYMAP = {
        b"q": AstCommand.exit,
        b"\x1b": AstCommand.exit,
        b"n": AstCommand.focus_next,
        b"r": AstCommand.focus_prev,
        b"l": AstCommand.less_details,
        b"m": AstCommand.more_details,
        b"j": AstCommand.jump_to_data,
        b"b": AstCommand.back,
        b"-": AstCommand.zoom_out,
        b"+": AstCommand.zoom_in,
        b"t": AstCommand.print_ast,
        b"v": AstCommand.print_verbose,
    }

    def __init__(self):
        super().__init__("pony-ast", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        if not from_tty:
            print("[ERROR] no tty :( :(", file=sys.stderr)
            return
        self.dont_repeat()
        stdout = sys.__stdout__
        with _restore_term(stdout), _alternate_screen(stdout), _real_stdout():
            _disable_echo(stdout)
            _cbreak(stdout)
            (height, width) = _term_size(stdout)

            if ast := self._select_ast_var():
                self._view_loop(ast, height, width)

    def _view_loop(self, ast: AstNode, height: int, width: int):
        def new_root(root):
            nonlocal ast, nodes, highlight
            ast = root
            nodes = _BackwardForwardCycle(
                chain([ast], ast.child.siblings if ast.child else [])
            )
            highlight = nodes.next()

        nodes = highlight = None
        new_root(ast)
        skip_redraw = False
        max_level = None
        stdout = sys.__stdout__
        origins = []
        while True:
            if not skip_redraw:
                _clear(stdout)
                lines = (
                    AstPrettyPrinter(max_level, width)
                    .render(ast, highlight=highlight)
                    .splitlines()
                )
                details = self._render_details(highlight, width)
                print("\n".join(lines[: height - details.count("\n") - 3]))
                print("\n")
                print(details, end="")
            skip_redraw = False
            cmd = self.KEYMAP.get(_next_key())
            if cmd is AstCommand.exit:
                break
            elif cmd is AstCommand.focus_prev:
                highlight = nodes.prev()
            elif cmd is AstCommand.focus_next:
                highlight = nodes.next()
            elif cmd is AstCommand.print_ast:
                print()
                gdb.parse_and_eval(
                    f"ast_print((ast_ptr_t)0x{int(highlight.value):x}, {width})"
                )
                skip_redraw = True
            elif cmd is AstCommand.print_verbose:
                print()
                gdb.parse_and_eval(
                    f"ast_printverbose((ast_ptr_t)0x{int(highlight.value):x})"
                )
                skip_redraw = True
            elif cmd is AstCommand.less_details:
                max_level = max(1, max_level - 1) if max_level is not None else 1
            elif cmd is AstCommand.more_details:
                max_level = max_level + 1 if max_level else 1
            elif cmd is AstCommand.jump_to_data:
                if data := highlight.data:
                    origins.append(ast)
                    new_root(data)
                else:
                    print("\nNo data")
                    skip_redraw = True
            elif cmd is AstCommand.back:
                if origins:
                    new_root(origins.pop())
                else:
                    print("\nNowhere to go back to :(")
                    skip_redraw = True
            elif cmd is AstCommand.zoom_in:
                new_root(highlight)
            elif cmd is AstCommand.zoom_out:
                if parent := ast.parent:
                    new_root(parent)
            else:
                skip_redraw = True

    def _render_details(self, node: AstNode, max_width: int) -> List[str]:
        details = Row(
            [self._render_node_details(node), Text("   "), self._render_keymap()]
        )
        return BoxPrinter(details, max_width).render()

    def _render_node_details(self, node: AstNode) -> Box:
        parent_addr = int(node.parent.value) if node.parent else 0
        child_addr = int(node.child.value) if node.child else 0
        if symtab := node.symtab():
            names = [Text(name) for name in symtab]
            if len(names) > 5:
                names[5:] = [Text("…")]
            values = Column(
                [
                    Text(f"0x{addr:x},{status:x}")
                    for (addr, status) in islice(symtab.values(), 5)
                ]
            )
            symbols = Row([Column(names), HSpace(" "), values])
        else:
            symbols = Text("none")
        string = node.token_string()
        details = Row(
            [
                Column(
                    [
                        Text("ID:"),
                        Text("Address:"),
                        Text("Parent:"),
                        Text("Child:"),
                        Text("Data:"),
                        Text("Token String"),
                        Text("Symbols:"),
                    ]
                ),
                Column([Text(" ")]),
                Column(
                    [
                        Text(node.token_id),
                        Text(f"0x{int(node.value):x}"),
                        Text(f"0x{int(parent_addr):x}"),
                        Text(f"0x{int(child_addr):x}"),
                        Text(f"0x{int(node.value['data']):x}"),
                        Text(string if string is not None else ""),
                        symbols,
                    ]
                ),
            ]
        )
        return frame("Node details", details)

    def _render_keymap(self) -> Box:
        keys = []
        descriptions = []
        for (key, cmd) in self.KEYMAP.items():
            key = key.decode("utf-8").replace("\x1b", "<esc>")
            keys.append(Text(f"{key:>5}"))
            descriptions.append(Text(cmd.value))
        return frame(
            "Keymap", Row([Column(keys), Column([Text(" ")]), Column(descriptions)])
        )

    def _select_ast_var(self):
        ast_vars = list(_all_asts_in_selected_frame())
        if not ast_vars:
            return
        print("Select AST:")
        for (i, (name, _)) in enumerate(ast_vars, 1):
            print(f"{i}  {name}")
        while True:
            key = _next_key()
            if b"1" <= key <= b"9":
                i = int(key)
                if i <= len(ast_vars):
                    return ast_vars[i - 1][1]
            elif key == b"\x1b":
                return


## Misc utility


class _BackwardForwardCycle:
    def __init__(self, iterable: Iterable[T]):
        # Invariant: index points to the last value.
        # Only negative on initialization.
        self._i = -1
        self._values = list(iterable)

    def prev(self) -> T:
        self._i = (self._i - 1) % len(self._values)
        return self._values[self._i]

    def next(self) -> T:
        self._i = (self._i + 1) % len(self._values)
        return self._values[self._i]


## Pony AST pretty printer


class AstPrettyPrinter:
    _renderers = {}

    def renders(token_id, *, _renderers=_renderers):
        def decorator(f):
            _renderers[token_id] = f
            return f

        return decorator

    def __init__(self, max_level: Optional[int], max_width: int):
        self.max_level = max_level or sys.maxsize
        self.max_width = max_width
        self._colors = iter(cycle(reversed(range(16))))

    def render(self, node: AstNode, *, highlight=None) -> str:
        node_box = self._render_node(node, highlight, set(), 0)
        box = frame("AST", padding(node_box, 2))
        return BoxPrinter(box, self.max_width).render()

    def _render_node(
        self,
        node: AstNode,
        highlight: AstNode,
        seen,
        level: int,
        render_siblings: bool = False,
    ) -> Box:
        if level > self.max_level:
            return Empty()
        elif node in seen:
            return Row([Text("[Loop detected]")])
        else:
            seen.add(node)

        if renderer := self._renderers.get(node.token_id):
            return renderer(self, node, highlight, seen, level)

        doc = []
        if child := node.child:
            doc.append(HSpace(" "))
            doc.append(
                self._render_node(
                    child, highlight, seen, level + 1, render_siblings=True
                )
            )
        if render_siblings:
            for sibling in islice(node.siblings, 1, None):
                doc.append(HSpace(" "))
                doc.append(self._render_node(sibling, highlight, seen, level + 1))
        return self._highlight(frame(node.token_id, group(doc)), node == highlight)

    @renders("TK_CAP_READ")
    @renders("TK_CAP_SEND")
    @renders("TK_CAP_SHARE")
    @renders("TK_CAP_ALIAS")
    @renders("TK_CAP_ANY")
    def _render_cap(self, node: AstNode, highlight: AstNode, seen, level: int) -> Box:
        return self._highlight(
            Text("#" + node.token_id[len("TK_CAP_") :].lower()), node == highlight
        )

    @renders("TK_ISO")
    @renders("TK_TRN")
    @renders("TK_REF")
    @renders("TK_VAL")
    @renders("TK_BOX")
    @renders("TK_TAG")
    def _render_ref_cap(
        self, node: AstNode, highlight: AstNode, seen, level: int
    ) -> Box:
        return self._highlight(
            Text(node.token_id[len("TK_") :].lower()), node == highlight
        )

    @renders("TK_AT")
    @renders("TK_FALSE")
    @renders("TK_TRUE")
    @renders("TK_NONE")
    @renders("TK_THIS")
    def _render_literal(
        self, node: AstNode, highlight: AstNode, seen, level: int
    ) -> Box:
        text_repr = {
            "TK_AT": "@",
            "TK_FALSE": "false",
            "TK_TRUE": "true",
            "TK_NONE": "None",
            "TK_THIS": "this",
        }
        return self._highlight(Text(text_repr[node.token_id]), node == highlight)

    @renders("TK_INT")
    def _render_int(self, node: AstNode, highlight: AstNode, seen, level: int) -> Box:
        box = Text(str(node.token_int()))
        return self._highlight(box, node == highlight)

    @renders("TK_STRING")
    def _render_string(
        self, node: AstNode, highlight: AstNode, seen, level: int
    ) -> Box:
        box = Text(node.token_string())
        return self._highlight(box, node == highlight)

    @renders("TK_ID")
    def _render_id(self, node: AstNode, highlight: AstNode, seen, level: int) -> Box:
        return self._highlight(Text(node.token_string()), highlight == node)

    @renders("TK_ACTOR")
    @renders("TK_ARROW")
    @renders("TK_ASSIGN")
    @renders("TK_CALL")
    @renders("TK_CLASS")
    @renders("TK_CONSUME")
    @renders("TK_FVAR")
    @renders("TK_FVARREF")
    @renders("TK_FUN")
    @renders("TK_FUNREF")
    @renders("TK_INTERFACE")
    @renders("TK_MEMBERS")
    @renders("TK_MODULE")
    @renders("TK_NEW")
    @renders("TK_NEWREF")
    @renders("TK_NOMINAL")
    @renders("TK_PRIMITIVE")
    @renders("TK_PARAM")
    @renders("TK_PARAMREF")
    @renders("TK_PARAMS")
    @renders("TK_POSITIONALARGS")
    @renders("TK_SEQ")
    @renders("TK_TYPEPARAM")
    @renders("TK_TYPEPARAMS")
    @renders("TK_TYPEPARAMREF")
    @renders("TK_TYPEREF")
    @renders("TK_USE")
    def _render_node_with_children(
        self, node: AstNode, highlight: AstNode, seen, level: int
    ) -> Box:
        children = list(node.child.siblings)
        boxes = [HSpace(" ")] * 2 * len(children)
        boxes[0::2] = [
            self._render_node(child, highlight, seen, level + 1) for child in children
        ]
        return self._highlight(
            frame(node.token_id[3:].lower(), group(boxes)), node == highlight
        )

    def _highlight(self, box: Box, highlight: bool) -> Box:
        # XXX find something smarter (Box annotations?)
        if highlight:
            # 1 == red
            return self._with_color(box, 1)
        else:
            return box

    def _with_color(self, box: Box, color: int) -> Box:
        def colorize(printer: BoxPrinter):
            lines = yield
            set_color = tparm(tigetstr("setaf"), color).decode("utf-8")
            reset_colors = _reset_colors().decode("utf-8")
            for (i, (length, line)) in enumerate(lines):
                lines[i] = (length, set_color + line + reset_colors)
            return lines

        return Decorated(colorize, box)

    del renders


def padding(box: Box, n: int) -> Box:
    def add_padding(printer: BoxPrinter):
        with printer._override("max_width", printer._state.max_width - (n * 2)):
            lines = yield
            columns = lines[0][0] if lines else 0
            new_size = columns + n * 2
            padding = " " * n
            lines = [(new_size, padding + line + padding) for (_, line) in lines]
            return lines

    return Decorated(add_padding, box)


## Terminal helpers


@contextmanager
def _restore_term(fd):
    attrs = termios.tcgetattr(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, attrs)


@contextmanager
def _alternate_screen(fd):
    os.write(fd.fileno(), tigetstr("smcup"))
    try:
        yield
    finally:
        os.write(fd.fileno(), tigetstr("rmcup"))


def _clear(fd):
    os.write(fd.fileno(), tigetstr("clear"))


def _cbreak(fd):
    tty.setcbreak(fd, termios.TCSADRAIN)


def _disable_echo(fd):
    attrs = termios.tcgetattr(fd)
    attrs[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, attrs)


def _next_key():
    return os.read(sys.__stdin__.fileno(), 1)


def _reset_colors():
    return tigetstr("op")


def _term_size(fd):
    return struct.unpack("hhhh", fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8))[0:2]


def _init():
    PonyAstCommand()
    setupterm(None, sys.__stdout__.fileno())


_init()