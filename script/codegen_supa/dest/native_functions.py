from typing import List, Optional, Union

import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.types import NativeSignature

from torchgen.context import native_function_manager, with_native_function_and_index
from torchgen.model import BackendIndex, NativeFunction, NativeFunctionsGroup
from torchgen.utils import mapMaybe
from torchgen.dest.native_functions import gen_unstructured


# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function_and_index
def compute_native_function_declaration(
    g: Union[NativeFunctionsGroup, NativeFunction], backend_index: BackendIndex
) -> List[str]:
    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            return list(mapMaybe(lambda f: gen_supa_structured(f, backend_index), g.functions()))
        else:
            return list(mapMaybe(lambda f: gen_supa_unstructured(f, backend_index), g.functions()))
    else:
        x = gen_supa_unstructured(g, backend_index)
        return [] if x is None else [x]


@with_native_function_and_index
def gen_supa_unstructured(f: NativeFunction, backend_index: BackendIndex) -> str:
    if getattr(f, "supa_kernel", False) is True:
        return gen_unstructured(f, backend_index)


@with_native_function_and_index
def compute_tonative_function_declaration(
    g: Union[NativeFunctionsGroup, NativeFunction], backend_index: BackendIndex
) -> List[str]:
    ret: List[str] = []
    if isinstance(g, NativeFunctionsGroup):
        funcs = g.functions()
    else:
        funcs = [g]
    for func in funcs:
        if getattr(func, "native_kernel", None):
            sig = NativeSignature(func.func, symint=False)
            returns_type = sig.returns_type().cpp_type()
            args_str = ", ".join(a.defn() for a in sig.arguments())
            name_str = f"{returns_type} {func.native_kernel}({args_str});"
            ret.append(name_str)
    return ret


def gen_supa_structured(f: NativeFunction, backend_index: BackendIndex) -> Optional[str]:
    if getattr(f, "supa_kernel", False) is True:
        return gen_unstructured(f, backend_index)
    else:
        return ""


def _supa_structured_parent_class(g: NativeFunctionsGroup) -> str:
    native_metadata = getattr(g.functional, "native_metadata", None)
    assert native_metadata is not None, f"Missing native metadata for structured group {g.functional.func}"
    return f"{native_metadata.cpp_namespace}::structured_{native_metadata.kernel}"


def _gen_supa_meta_declarations(g: NativeFunctionsGroup) -> List[str]:
    args = structured.meta_arguments(g)
    args_str = ", ".join(a.decl() for a in args)

    if g.out.precomputed:
        return [f"meta_return_ty meta({args_str});"]
    else:
        return [f"void meta({args_str});"]


@with_native_function_and_index
def gen_structured(g: NativeFunctionsGroup, backend_index: BackendIndex) -> List[str]:
    """generate declaration for structured group. adding generating impl_supa()"""
    metadata = backend_index.get_kernel(g)
    if metadata is None:
        return []
    if not getattr(g, "has_supa_structs", False) and not getattr(g, "has_supa_meta", False):
        return []

    with native_function_manager(g.out):
        meta_name = meta.name(g)
        prefix = "" if backend_index.external else "TORCH_API "
        parent_class = _supa_structured_parent_class(g)

        declarations = []
        if getattr(g, "has_supa_meta", False):
            declarations.extend(_gen_supa_meta_declarations(g))
        if getattr(g, "has_supa_structs", False):
            out_args = structured.impl_arguments(g)
            args = f"{', '.join(a.decl() for a in out_args)}"
            declarations.append(f"void impl_supa({args});")

        body = "\n".join(declarations)
        return [
            f"""
struct {prefix}structured_{meta_name} : public {parent_class} {{
{body}
}};
"""
        ]
