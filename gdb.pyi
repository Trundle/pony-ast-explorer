from typing import Iterator, Optional

TYPE_CODE_PTR: int
TYPE_CODE_TYPEDEF: int

class Block:
    # http://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/python/py-block.c;hb=HEAD

    is_global: bool
    is_static: bool
    superblock: Optional[Block]
    def __iter__(self) -> Iterator[Symbol]: ...

class Frame:
    # http://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/python/py-frame.c;hb=HEAD
    def block(self) -> Block:
        """
        Return the frame's code block.
        """

class Symbol:
    # http://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/python/py-symbol.c;hb=HEAD

    is_argument: bool
    is_variable: bool
    type: Optional[Type]
    def value(self, frame: Frame = ...) -> Value: ...

class Type:
    # http://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/python/py-type.c;hb=HEAD

    code: int
    name: Optional[str]
    def pointer(self) -> Type:
        """
        Return a type of pointer to this type.
        """
    def target(self) -> Type:
        """
        Return the target type of this type.
        """

class Value:
    """
    GDB value object.
    """

    # http://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=gdb/python/py-value.c;hb=HEAD

    is_optimized_out: bool
    type: Type
    def __getitem__(self, name: str) -> Value: ...
    def __int__(self) -> int: ...
    def cast(self, type: Type) -> Value: ...
    def dereference(self) -> Value: ...
    def string(
        self, encoding: str = ..., errors: str = ..., length: int = ...
    ) -> str: ...

class Command:
    pass

def lookup_type(name: str, block: Block = ...) -> Type:
    """
    Return a Type corresponding to the given name.
    """

def parse_and_eval(expr: str) -> Value:
    """
    Parse String as an expression, evaluate it, and return the result as a Value.
    """

def selected_frame() -> Frame: ...
