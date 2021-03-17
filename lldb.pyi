from typing import Iterator

class SBExecutionContext:
    frame: SBFrame

class SBFrame:
    def get_all_variables(self) -> SBValuesList: ...
    def EvaluateExpression(self, expr: str) -> SBValue: ...

class SBType:
    name: str
    def IsPointerType(self) -> bool: ...

class SBValue:
    # https://lldb.llvm.org/python_reference/lldb-pysrc.html#SBValue

    deref: SBValue
    signed: int
    summary: str
    type: SBType
    unsigned: int
    value: str
    def GetChildMemberWithName(self, name: str) -> SBValue: ...

class SBValuesList:
    def __iter__(self):
        Iterator[SBValue]
