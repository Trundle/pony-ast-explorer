from __future__ import annotations

import dataclasses
import enum
from contextlib import contextmanager
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Generator, Iterable, List, Optional, Protocol, Sequence, Tuple


RenderedLine = Tuple[int, str]
RenderedBox = List[RenderedLine]


@dataclass
class Box:
    pass


@dataclass
class Alt(Box):
    """
    Alternation: either left or right, with bias to left.
    """

    left: Box
    right: Box


class DecoratorCallback(Protocol):
    def __call__(
        self, printer: BoxPrinter
    ) -> Generator[None, RenderedBox, RenderedBox]:
        pass


@dataclass
class Decorated(Box):
    decorator: DecoratorCallback
    box: Box


@dataclass
class Empty(Box):
    """
    A box without any content.
    """


@dataclass
class Column(Box):
    """
    A column of boxes.
    """

    boxes: Sequence[Box]


@dataclass
class Row(Box):
    """
    A row of boxes.
    """

    boxes: Sequence[Box]


@dataclass
class HSpace(Box):
    """
    Horizontal space. Ignored in vertical direction.
    """

    space: str


@dataclass
class VSpace(Box):
    """
    Vertical space. Ignored in horizontal direction.
    """

    space: str


@dataclass
class Text(Box):
    """
    Text fragment, without newlines.
    """

    text: str
    length: Optional[int] = None

    def __post_init__(self):
        if self.length is None:
            self.length = len(self.text)


def _horizontal_concat(boxes: Iterable[RenderedBox]) -> RenderedBox:
    (rows, cols) = max_sum((len(lines), lines[0][0]) for lines in boxes if lines)
    boxes = [_resize_box(box, rows, box[0][0] if box else 0) for box in boxes]
    join_line = lambda parts: (sum(x[0] for x in parts), "".join(x[1] for x in parts))
    return [join_line(parts) for parts in zip(*boxes)]


def _vertical_concat(boxes: Iterable[RenderedBox]) -> RenderedBox:
    lines = []
    for box in boxes:
        lines.extend(box)
    columns = max(line[0] for line in lines) if lines else 0
    return [(columns, line + " " * (columns - length)) for (length, line) in lines]


class Direction(enum.Enum):
    HORIZONTAL = _horizontal_concat
    VERTICAL = _vertical_concat


class BoxPrinter:
    @dataclass
    class _State:
        max_width: int
        direction: Direction
        boxes: List[RenderedBox] = dataclasses.field(default_factory=lambda: [[]])

        @property
        def current_box(self) -> RenderedBox:
            return self.boxes[-1]

        def push_box(self):
            self.boxes.append([])

        def pop_box(self):
            box = self.boxes.pop()
            merged_box = self.direction([self.current_box, box])
            if merged_box and merged_box[0][0] > self.max_width:
                raise OverflowError()
            self.boxes[-1] = merged_box

    def __init__(self, box: Box, max_width: int):
        self._box = box
        self._state = self._State(max_width, Direction.HORIZONTAL)

    def render(self) -> str:
        self._render_box(self._box)
        assert len(self._state.boxes) == 1
        return "\n".join(line[1] for line in self._state.current_box)

    @singledispatchmethod
    def _render_box(self, box: Box):
        raise ValueError(f"Not a box: {box!r}")

    @_render_box.register
    def _render_empty(self, empty: Empty):
        pass

    @_render_box.register
    def _render_text(self, text: Text):
        with self._new_box():
            assert text.length is not None
            self._state.boxes[-1] = [(text.length, text.text)]

    @_render_box.register
    def _render_alt(self, alt: Alt):
        try:
            with self._snapshot():
                self._render_box(alt.left)
        except OverflowError:
            self._render_box(alt.right)

    @_render_box.register
    def _render_decorated(self, decorated: Decorated):
        with self._new_box():
            decorator = decorated.decorator(self)
            next(decorator)
            self._render_box(decorated.box)
            try:
                decorator.send(self._state.current_box)
            except StopIteration as e:
                self._state.boxes[-1] = e.value
            else:
                raise RuntimeError(
                    f"Decorator for {decorated} didn't throw expected "
                    "StopIteration exception with result"
                )

    @_render_box.register
    def _render_column(self, column: Column):
        with self._new_box():
            with self._override("direction", Direction.VERTICAL):
                for box in column.boxes:
                    self._render_box(box)

    @_render_box.register
    def _render_row(self, row: Row):
        with self._new_box():
            with self._override("direction", Direction.HORIZONTAL):
                for box in row.boxes:
                    self._render_box(box)

    @_render_box.register
    def _render_hspace(self, hspace: HSpace):
        if self._state.direction is Direction.HORIZONTAL:
            self._render_box(Text(hspace.space))

    @_render_box.register
    def _render_vspace(self, vspace: VSpace):
        if self._state.direction is Direction.VERTICAL:
            self._render_box(Text(vspace.space))

    @contextmanager
    def _snapshot(self):
        saved_state = self._state
        # XXX not a deep copy
        self._state = dataclasses.replace(saved_state, boxes=list(saved_state.boxes))
        try:
            yield
        except Exception:
            self._state = saved_state
            raise

    @contextmanager
    def _override(self, name, value):
        saved_value = getattr(self._state, name)
        try:
            setattr(self._state, name, value)
            yield
        finally:
            setattr(self._state, name, saved_value)

    @contextmanager
    def _new_box(self):
        self._state.push_box()
        try:
            yield
        finally:
            self._state.pop_box()


def _resize_box(box: RenderedBox, rows: int, cols: int) -> RenderedBox:
    """
    Resize the given box, represented by a list of rendered lines, to the given size.
    """
    result = []
    for (length, row) in box:
        result.append((cols, row + " " * (cols - length)))
    for _ in range(rows - len(result)):
        result.append((cols, " " * cols))
    return result


def max_sum(iterable: Iterable[Tuple[int, int]]) -> Tuple[int, int]:
    """
    max() and sum(), but in a single pass.
    """
    current_max = 0
    current_sum = 0
    for (max_value, sum_value) in iterable:
        current_max = max(current_max, max_value)
        current_sum += sum_value
    return (current_max, current_sum)


def group(boxes: Sequence[Box]) -> Box:
    return Alt(Row(boxes), Column(boxes))


def frame(title: Optional[str], box: Box) -> Box:
    def add_frame(printer: BoxPrinter) -> Generator[None, RenderedBox, RenderedBox]:
        with printer._override("max_width", printer._state.max_width - 4):
            lines = yield
            columns = lines[0][0] if lines else 0
            spaced_title = " " + title + " " if title else "──"
            if len(spaced_title) > columns:
                spaced_title = spaced_title[: max(columns - 1, 0)] + "…"
            result = [(columns + 4, "╭─" + spaced_title.ljust(columns + 1, "─") + "╮")]
            for line in lines:
                result.append((columns + 4, "│ " + line[1] + " │"))
            result.append((columns + 4, "╰" + "─" * (columns + 2) + "╯"))
            return result

    return Decorated(add_frame, box)
