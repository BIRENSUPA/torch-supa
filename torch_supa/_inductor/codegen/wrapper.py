# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

from typing import Optional
from torch_supa.utils import torch_version_ge
if torch_version_ge(2, 6, 0):
    from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen, PythonWrapperCodegen
else:
    from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen, WrapperCodeGen as PythonWrapperCodegen

try:
    from torch._inductor.ir import GraphPartitionSignature
except ImportError:
    # dummy
    GraphPartitionSignature = None


class SUPAWrapperCodeGen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str,
        parent_wrapper: PythonWrapperCodegen,
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            if torch_version_ge(2, 7, 0):
                return SubgraphPythonWrapperCodegen(subgraph_name, parent_wrapper, partition_signatures)
            else:
                return SubgraphPythonWrapperCodegen(subgraph_name, parent_wrapper)
        return SUPAWrapperCodeGen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            """
                import torch_supa
            """,
            strip=True,
        )

        self.header.splice(
            """
                empty_strided_cuda = torch_supa._C._dynamo._empty_strided_supa
            """,
            strip=True,
        )
