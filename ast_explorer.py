from __future__ import annotations

import abc
import enum
import os
import sys
import termios
import tty
from contextlib import contextmanager
from curses import setupterm, tigetstr, tparm
from itertools import chain, cycle, islice
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    MutableSequence,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

try:
    import gdb

    is_gdb = True
except ImportError:
    import lldb

    is_gdb = False

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


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


## Debugger abstractions


class Debugger(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def is_null(self, value: T) -> bool:
        ...

    @abc.abstractmethod
    def member(self, value: T, name: str) -> T:
        ...

    @abc.abstractmethod
    def pointer_address(self, value: T) -> int:
        ...

    @abc.abstractmethod
    def enum_description(self, value: T) -> str:
        ...

    @abc.abstractmethod
    def evaluate_expression(self, expr: str) -> T:
        ...

    @abc.abstractmethod
    def int_value(self, value: T) -> int:
        ...

    @abc.abstractmethod
    def string_value(self, value: T) -> str:
        ...

    @abc.abstractmethod
    def pointer_to_ast(self, address: int) -> T:
        ...

    @abc.abstractmethod
    def ast_nodes_in_frame(self) -> Iterable[Tuple[str, AstNode]]:
        """
        Find all AST nodes in the currently selected frame.
        """


class AstNode:
    """
    Wrapper around a debugger value to make working with AST nodes a bit more
    convenient and debugger-independant.
    """

    def __init__(self, debugger: Debugger, value: gdb.Value):
        self._debugger = debugger
        self.value = value

    def __eq__(self, other):
        if isinstance(other, AstNode):
            addr = self._debugger.pointer_address(self.value)
            other_addr = self._debugger.pointer_address(other.value)
            return addr == other_addr
        return NotImplemented

    def __hash__(self):
        return self._debugger.pointer_address(self.value)

    def __repr__(self):
        addr = self._debugger.pointer_address(self.value)
        return f"<AstNode@0x{addr:x}(id={self.token_id})>"

    @property
    def address(self) -> int:
        return self._debugger.pointer_address(self.value)

    @property
    def data_address(self) -> int:
        return self._debugger.pointer_address(self._debugger.member(self.value, "data"))

    @property
    def token_id(self) -> str:
        m = self._debugger.member
        return self._debugger.enum_description(m(m(self.value, "t"), "id"))

    @property
    def child(self) -> Optional[AstNode]:
        child = self._debugger.member(self.value, "child")
        if not self._debugger.is_null(child):
            return AstNode(self._debugger, child)
        return None

    @property
    def data(self) -> Optional[AstNode]:
        data = self._debugger.member(self.value, "data")
        if not self._debugger.is_null(data):
            return AstNode(self._debugger, self._debugger.pointer_to_ast(data))
        return None

    @property
    def parent(self) -> Optional[AstNode]:
        if parent := self._debugger.member(self.value, "parent"):
            return AstNode(self._debugger, parent)
        return None

    @property
    def sibling(self) -> Optional[AstNode]:
        """
        The node's direct sibling (if there is one).
        """
        sibling = self._debugger.member(self.value, "sibling")
        if not self._debugger.is_null(sibling):
            return AstNode(self._debugger, sibling)
        return None

    @property
    def siblings(self) -> Iterable[AstNode]:
        """
        Returns the node itself and all its siblings.
        """
        current_value: Optional[AstNode] = self
        while current_value:
            yield current_value
            current_value = current_value.sibling

    def symtab(self) -> Optional[Dict[str, Tuple[int, str]]]:
        symtab = self._debugger.member(self.value, "symtab")
        if not self._debugger.is_null(symtab):
            return self._read_symtab(symtab)
        return None

    def token_int(self):
        m = self._debugger.member
        return self._debugger.int_value(m(m(m(self.value, "t"), "integer"), "low"))

    def token_string(self) -> Optional[str]:
        m = self._debugger.member
        if self.token_id in {"TK_ID", "TK_STRING"} and (
            string := m(m(self.value, "t"), "string")
        ):
            return self._debugger.string_value(string)
        return None

    def dump(self, width: int, verbose: bool = False):
        ast_node = f"(ast_t *){self._debugger.pointer_address(self.value)}UL"
        if verbose:
            expr = f"ast_printverbose({ast_node})"
        else:
            expr = f"ast_print({ast_node}, {width})"
        self._debugger.evaluate_expression(expr)

    def _read_symtab(self, value) -> Dict[str, Tuple[int, str]]:
        debugger = self._debugger
        i = debugger.evaluate_expression("(void *)malloc(sizeof(size_t))")
        i_addr = debugger.pointer_address(i)
        symtab_addr = debugger.pointer_address(value)
        debugger.evaluate_expression(f"*((size_t *){i_addr}UL) = (size_t)-1")
        result = {}
        while True:
            sym = debugger.evaluate_expression(
                f"symtab_next(((symtab_t *){symtab_addr}UL), (size_t *){i_addr}UL)"
            )
            if debugger.is_null(sym):
                break
            name = debugger.string_value(debugger.member(sym, "name"))
            def_ = debugger.pointer_address(debugger.member(sym, "def"))
            status = debugger.enum_description(debugger.member(sym, "status"))
            result[name] = (def_, status)
        return result


## GDB debugger implementation


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


class Gdb(Debugger):
    def is_null(self, value: gdb.Value) -> bool:
        return int(value) == 0

    def pointer_address(self, pointer: gdb.Value) -> int:
        return int(pointer)

    def pointer_to_ast(self, pointer: gdb.Value) -> gdb.Value:
        return pointer.cast(gdb.lookup_type("ast_t").pointer())

    def ast_nodes_in_frame(self) -> Iterable[Tuple[str, AstNode]]:
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
                            yield (str(symbol), AstNode(self, ast_ptr))
                            block = block.superblock

    def enum_description(self, value: gdb.Value) -> str:
        return str(value)

    def evaluate_expression(self, expr: str) -> gdb.Value:
        return gdb.parse_and_eval(expr)

    def member(self, value: gdb.Value, name: str) -> gdb.Value:
        return value[name]

    def int_value(self, value: gdb.Value) -> int:
        return int(value)

    def string_value(self, value: gdb.Value) -> str:
        return value.string()


@contextmanager
def _real_stdout():
    # Otherwise using print() in Python triggers gdb's pager
    stdout = sys.stdout
    sys.stdout = sys.__stdout__
    try:
        yield
    finally:
        sys.stdout = stdout


## LLDB implementation


class Lldb(Debugger):
    def __init__(self, execution_context: lldb.SBExecutionContext):
        self._context = execution_context

    def ast_nodes_in_frame(self) -> Iterable[Tuple[str, AstNode]]:
        for variable in self._context.frame.get_all_variables():
            if self._is_ast(variable) and variable.value is not None:
                yield (variable.name, self._to_ast(variable))

    def is_null(self, value: lldb.SBValue) -> bool:
        return self.pointer_address(value) == 0  # XXX 0xffffffffffffffff

    def pointer_address(self, value: lldb.SBValue) -> int:
        assert value.type.IsPointerType()
        return value.unsigned

    def pointer_to_ast(self, pointer: lldb.SBValue) -> lldb.SBValue:
        assert pointer.type.IsPointerType()
        addr = self.pointer_address(pointer)
        return self._context.frame.EvaluateExpression(f"(ast_t *){addr}UL")

    def member(self, value: lldb.SBValue, name: str) -> lldb.SBValue:
        return value.GetChildMemberWithName(name)

    def enum_description(self, value: lldb.SBValue) -> str:
        return value.value

    def evaluate_expression(self, expr: str) -> lldb.SValue:
        return self._context.frame.EvaluateExpression(expr)

    def int_value(self, value: lldb.SBValue) -> int:
        return value.value

    def string_value(self, value: lldb.SBValue) -> str:
        # :/ this seems to be the best - ReadCStringFromMemory didn't always work
        return value.summary[1:-1]

    def _is_ast(self, value: lldb.SBValue) -> bool:
        return value.type.name.startswith("ast_t *")

    def _to_ast(self, value: lldb.SBValue) -> AstNode:
        while value.type.name != "ast_t *":
            value = value.deref
        return AstNode(self, value)


## pony-ast command


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


class PonyAstCommand:
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

    def __init__(self, debugger: Debugger):
        self._gdb = debugger

    def __call__(self):
        stdout = sys.__stdout__
        with _restore_term(stdout), _alternate_screen(stdout), _real_stdout():
            _disable_echo(stdout)
            _cbreak(stdout)
            (width, height) = os.get_terminal_size(stdout.fileno())

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

        nodes: _BackwardForwardCycle[AstNode]
        highlight = ast
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
                highlight.dump(width, verbose=False)
                skip_redraw = True
            elif cmd is AstCommand.print_verbose:
                print()
                highlight.dump(width, verbose=True)
                skip_redraw = True
            elif cmd is AstCommand.less_details:
                if max_level is None:
                    max_level = 1
                else:
                    max_level = max(1, max_level - 1)
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

    def _render_details(self, node: AstNode, max_width: int) -> str:
        details = Row(
            [self._render_node_details(node), Text("   "), self._render_keymap()]
        )
        return BoxPrinter(details, max_width).render()

    def _render_node_details(self, node: AstNode) -> Box:
        parent_addr = node.parent.address if node.parent else 0
        child_addr = node.child.address if node.child else 0
        symbols: Box
        if symtab := node.symtab():
            names = [Text(name) for name in symtab]
            if len(names) > 5:
                names[5:] = [Text("…")]
            values = Column(
                [
                    Text(f"0x{addr:x},{status}")
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
                        Text(f"0x{node.address:x}"),
                        Text(f"0x{parent_addr:x}"),
                        Text(f"0x{child_addr:x}"),
                        Text(f"0x{node.data_address:x}"),
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
            key_repr = key.decode("utf-8").replace("\x1b", "<esc>")
            keys.append(Text(f"{key_repr:>5}"))
            descriptions.append(Text(cmd.value))
        return frame(
            "Keymap", Row([Column(keys), Column([Text(" ")]), Column(descriptions)])
        )

    def _select_ast_var(self):
        ast_vars = list(self._gdb.ast_nodes_in_frame())
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


## Gdb implementation of PonyAstCommand


if is_gdb:

    class GdbPonyAstCommand(PonyAstCommand, gdb.Command):
        def __init__(self):
            gdb.Command.__init__(self, "pony-ast", gdb.COMMAND_USER)
            PonyAstCommand.__init__(self, Gdb())

        def invoke(self, arg, from_tty):
            if not from_tty:
                print("[ERROR] no tty :( :(", file=sys.stderr)
                return
            self.dont_repeat()
            self()


## LLDB implementaiton of PonyAstCommand

if not is_gdb:

    class LldbPonyAstCommand(PonyAstCommand):
        def __init__(self, debugger: Lldb):
            super().__init__(debugger)

    def _lldb_cmd_pony_ast(debugger, command, exe_ctx, result, internal_dict):
        lldb = Lldb(exe_ctx)
        cmd = LldbPonyAstCommand(lldb)
        cmd()


## Misc utility


class _BackwardForwardCycle(Generic[T]):
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


class _Registry(Generic[K, V]):
    def __init__(self):
        self._known = {}

    def __call__(self, key: K):
        def decorator(f: V) -> V:
            self._known[key] = f
            return f

        return decorator

    def __getitem__(self, key: K) -> Optional[V]:
        return self._known.get(key)


class AstPrettyPrinter:
    # _renderers: Dict[] = {}

    _renderers: _Registry[
        str, Callable[[AstPrettyPrinter, AstNode, AstNode, Set[AstNode], int], Box]
    ] = _Registry()
    renders = _renderers

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

        if renderer := self._renderers[node.token_id]:
            return renderer(self, node, highlight, seen, level)

        doc: MutableSequence[Box] = []
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
        string = node.token_string()
        assert string is not None
        box = Text(string)
        return self._highlight(box, node == highlight)

    @renders("TK_ID")
    def _render_id(self, node: AstNode, highlight: AstNode, seen, level: int) -> Box:
        id_string = node.token_string()
        assert id_string is not None
        return self._highlight(Text(id_string), highlight == node)

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
        child = node.child
        assert child is not None
        children = list(child.siblings)
        boxes: List[Box] = list(HSpace(" ") for _ in range(2 * len(children)))
        boxes[0::2] = (
            self._render_node(child, highlight, seen, level + 1) for child in children
        )
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


def _init():
    if is_gdb:
        GdbPonyAstCommand()
    else:
        lldb.debugger.HandleCommand(
            "command script add -f ast_explorer._lldb_cmd_pony_ast pony-ast"
        )
    setupterm(None, sys.__stdout__.fileno())


_init()
