from __future__ import annotations

import enum
import fcntl
import os
import struct
import sys
import termios
import tty
from collections import deque
from contextlib import contextmanager
from curses import setupterm, tigetstr, tparm
from dataclasses import astuple, dataclass
from itertools import cycle, islice
from typing import Iterable, List, Optional, Tuple, TypeVar, Union

import gdb


T = TypeVar('T')

## Pretty printing core

@dataclass
class Document:
    """
    Base class for a document. A `Document` represents a set of layouts.
    """
    def __add__(self, other):
        if isinstance(other, Document):
            return Concat(self, other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return Concat(Text(other), self)
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, Document):
            return Alt(self, other)
        return NotImplemented


@dataclass
class Empty(Document):
    """
    The empty document.
    """


@dataclass
class Align(Document):
    """
    Set indentation to current column.
    """
    doc: Document


@dataclass
class Alt(Document):
    left: Document
    right: Document


@dataclass
class Concat(Document):
    left: Document
    right: Document


@dataclass
class Nest(Document):
    """
    Increases the current indentation level.
    """
    n: int
    doc: Document


@dataclass
class Text(Document):
    text: str


@dataclass
class ZeroText(Document):
    """
    A zero width text, e.g. ANSI escape sequences.
    """
    text: str


@dataclass
class Line(Document):
    """
    A new line.
    """


@dataclass
class Spacing(Document):
    space: str


@dataclass
class _RenderProcess:
    current_indent: int
    progress: int
    tokens: List[str]
    rest: List[Tuple[int, Document]]


class PrettyPrinterBase:
    # See https://jyp.github.io/posts/towards-the-prettiest-printer.html

    def __init__(self, max_width):
        self.max_width = max_width

    def _render(self, document: Document) -> str:
        return "".join(self._render_loop([_RenderProcess(0, 0, [], [(0, document)])]))

    def _render_loop(self, processes: List[_RenderProcess]) -> List[str]:
        to_do = list(processes)
        while to_do:
            to_do.sort(key=lambda p: (p.current_indent, p.progress))
            # XXX eleminate old progress?
            process = to_do.pop()
            for result in self._render_all(
                process.progress,
                process.tokens,
                process.current_indent,
                process.rest
            ):
                if isinstance(result, _RenderProcess):
                    to_do.append(result)
                else:
                    return result
        raise RuntimeError("Overflow :( :(")

    def _render_all(
        self, progress: int, tokens: List[str], j: int, docs: Tuple[int, Document]
    ) -> List[Union[List[str], _RenderProcess]]:
        if j > self.max_width:
            return []
        elif not docs:
            # Done!
            return [tokens]
        (i, doc) = docs.pop()
        # XXX a lot of list copies are made here
        if isinstance(doc, Empty):
            return self._render_all(progress, tokens, j, docs)
        if isinstance(doc, Align):
            return self._render_all(progress, tokens, j, docs + [(j, doc.doc)])
        elif isinstance(doc, Alt):
            return (
                self._render_all(progress, tokens, j, docs + [(i, doc.right)])
                + self._render_all(progress, tokens, j, docs + [(i, doc.left)])
            )
        elif isinstance(doc, Concat):
            return self._render_all(progress, tokens, j, docs + [(i, doc.right), (i, doc.left)])
        elif isinstance(doc, Nest):
            return self._render_all(progress, tokens, j, docs + [(i + doc.n, doc.doc)])
        elif isinstance(doc, Line):
            return [_RenderProcess(i, progress, tokens + [f"\n{' ' * i}"], docs)]
        elif isinstance(doc, Spacing):
            # Difference to Text: progress stays same!
            return self._render_all(progress, tokens + [doc.space], j + len(doc.space), docs)
        elif isinstance(doc, Text):
            return self._render_all(progress + 1, tokens + [doc.text], j + len(doc.text), docs)
        elif isinstance(doc, ZeroText):
            return self._render_all(progress + 1, tokens + [doc.text], j, docs)
        else:
            raise ValueError(f"Not a `Document`: {doc!r}")


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

    def iter_dfs(self) -> Iterable[AstNode]:
        """
        Iterates over this node and its descendants, depth-first.
        """
        yield self
        current_node = self.child
        while current_node:
            yield from current_node.iter_dfs()
            current_node = current_node.sibling

    def token_string(self):
        if string := self.value["t"]["string"]:
            return string.string()
        raise ValueError(f"Not a string token: {self.token_id}")


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


class PonyAstCommand(gdb.Command):
    KEYMAP = {
        b"q": AstCommand.exit,
        b"\x1b": AstCommand.exit,
        b"n": AstCommand.focus_next,
        b"r": AstCommand.focus_prev,
    }

    def __init__(self):
        super().__init__("pony-ast", gdb.COMMAND_USER)
        self.dont_repeat()

    def invoke(self, arg, from_tty):
        if not from_tty:
            print("[ERROR] no tty :( :(", file=sys.stderr)
            return
        stdout = sys.__stdout__
        with _restore_term(stdout), _alternate_screen(stdout), _real_stdout():
            _disable_echo(stdout)
            _cbreak(stdout)
            (height, width) = _term_size(stdout)
            if ast := self._select_ast_var():
                nodes = _BackwardForwardCycle(ast.iter_dfs())
                highlight = nodes.next()
                while True:
                    _clear(stdout)
                    print(AstPrettyPrinter(width).render(ast, highlight=highlight))
                    print(self._render_keymap())
                    cmd = self.KEYMAP.get(_next_key())
                    if cmd is AstCommand.exit:
                        break
                    elif cmd is AstCommand.focus_prev:
                        highlight = nodes.prev()
                    elif cmd is AstCommand.focus_next:
                        highlight = nodes.next()

    def _render_keymap(self):
        width = 32
        lines = ["┌" + " Keymap ".ljust(width, "─") + "┐"]
        for (key, cmd) in self.KEYMAP.items():
            key = key.decode("utf-8").replace("\x1b", "<esc>")
            lines.append("│" + f" {key:>5}  {cmd.value}".ljust(width) + "│")
        lines.append("└────────────────────────────────┘")
        return "\n".join(lines)

    def _select_ast_var(self):
        ast_vars = list(_all_asts_in_selected_frame())
        if not ast_vars:
            return
        print('Select AST:')
        for (i, (name, _)) in enumerate(ast_vars, 1):
            print(f"{i}  {name}")
        while True:
            key = _next_key()
            if b'1' <= key <= b'9':
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

class AstPrettyPrinter(PrettyPrinterBase):
    _renderers = {}

    def renders(token_id, *, _renderers=_renderers):
        def decorator(f):
            _renderers[token_id] = f
            return f
        return decorator

    def __init__(self, max_width: int):
        super().__init__(max_width)
        self._colors = iter(cycle(reversed(range(16))))

    def render(self, node: AstNode, *, highlight=None) -> str:
        return self._render(self._render_node(node, highlight, set()))

    @staticmethod
    def _node_descr(node: AstNode) -> str:
        token_id = node.token_id
        if token_id == "TK_CAP_READ":
            return "#read"
        elif token_id == "TK_CAP_SEND":
            return "#send"
        elif token_id == "TK_CAP_SHARE":
            return "#share"
        elif token_id == "TK_CAP_ALIAS":
            return "#alias"
        elif token_id == "TK_CAP_ANY":
            return "#any"

        elif token_id == "TK_ISO":
            return "iso"
        elif token_id == "TK_TRN":
            return "trn"
        elif token_id == "TK_REF":
            return "ref"
        elif token_id == "TK_VAL":
            return "val"
        elif token_id == "TK_BOX":
            return "box"
        elif token_id == "TK_TAG":
            return "tag"

        elif token_id == "TK_ID":
            return f'(ID {node.token_string()})'
        else:
            return token_id

    def _render_node(self, node: AstNode, highlight: AstNode, seen) -> Document:
        if node in seen:
            return Empty()
        else:
            seen.add(node)

        if renderer := self._renderers.get(node.token_id):
            return renderer(self, node, highlight, seen)

        (openp, closep) = self._parens()
        if child := node.child:
            doc = openp
        else:
            doc = Empty()
        doc += self._highlight(Text(self._node_descr(node)), highlight == node)
        if child:
            child_doc = self._render_node(child, highlight, seen)
            doc += (Spacing(" ") + child_doc) | Nest(2, Line() + child_doc)
            doc += closep
        # XXX Should this be handled here at all?
        #if sibling := node.sibling:
        #    sibling_doc = self._render_node(sibling, highlight, seen)
        #    doc += (Spacing(" ") + sibling_doc) | (Line() + sibling_doc)
        return doc

    @renders("TK_NOMINAL")
    def _render_nominal(self, node: AstNode, highlight: AstNode, seen) -> Document:
        (package, id_, typeargs, cap, ephemeral) = islice(node.child.siblings, 5)

        id_doc = Text(id_.token_string())
        typeargs_doc = self._render_node(typeargs, highlight, seen)
        cap_doc = self._highlight(Text(self._node_descr(cap)), cap == highlight)
        # XXX ephemeral
        return self._with_parens(
            self._highlight(Text("nominal"), highlight == node) + Spacing(" ") + (
                (id_doc + Spacing(" ") + typeargs_doc + Spacing(" ") + cap_doc)
                | Align(id_doc + Line() + typeargs_doc + Line() + cap_doc)
            ))

    @renders("TK_TYPEPARAMREF")
    def _render_typeparam_ref(self, node, highlight: AstNode, seen) -> Document:
        (id_, cap, ephemeral) = islice(node.child.siblings, 3)

        id_doc = Text(id_.token_string())
        cap_doc = Text(self._node_descr(cap))
        ephemeral_doc = self._render_node(ephemeral, highlight, seen)
        (openp, closep) = self._parens()
        return self._with_parens(
            self._highlight(Text("typeparamref"), node == highlight) + (
                (Spacing(" ") + id_doc + Spacing(" ") + cap_doc + Spacing(" ") + ephemeral_doc)
                | Align(Line() + id_doc + Line() + cap_doc + Line() + ephemeral_doc)
            ))

    def _highlight(self, doc: Document, highlight: bool):
        # XXX find something smarter (annotate Documents?)
        if highlight:
            # 1 == red
            return self._with_color(doc, 1)
        else:
            return doc

    def _parens(self):
        color = next(self._colors)
        return (self._with_color(Text("╭"), color), self._with_color(Text("╮"), color))

    def _with_parens(self, doc: Document):
        (openp, closep) = self._parens()
        return openp + doc + closep

    def _with_color(self, doc: Document, color: int):
        set_color = ZeroText(tparm(tigetstr("setaf"), color).decode("utf-8"))
        reset_colors = ZeroText(_reset_colors().decode("utf-8"))
        return set_color + doc + reset_colors

    del renders


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
