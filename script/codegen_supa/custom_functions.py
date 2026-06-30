import re
from collections import namedtuple
from typing import Sequence

from torchgen.code_template import CodeTemplate
from torchgen.gen import FileManager, cpp_string
from torchgen.model import NativeFunction
from torchgen.utils import concatMap
from torchgen.context import with_native_function, native_function_manager
from torchgen.api.types import DispatcherSignature
from torchgen.api import cpp


# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple("ParsedYaml", ["native_functions", "backend_indices"])


CUSTOM_FUNCTIONS_DECLARATION = CodeTemplate(
    """\
${return_type} ${func_name}(${args_str});
"""
)

EXPORT_CUSTOM_FUNCTIONS_DECLARATION = CodeTemplate(
    """\
__attribute__((__visibility__("default"))) \
${return_type} ${func_name}(${args_str});
"""
)

CUSTOM_FUNCTIONS_DEFINITION = CodeTemplate(
    """\
${return_type} ${func_name}(${args_str}) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("custom::${base_name}", "${overload}").typed<${schema}>();
    return op.${func_type}(${args_exprs_str});
}
"""
)

SKIP_PYTHON_BINDINGS_SIGNATURES = []


@with_native_function
def should_generate_ops_patch(f: NativeFunction) -> bool:
    func_signature = str(f.func)

    if f.root_name.startswith("_"):
        return False

    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == func_signature:
            return False

    return True


METHOD_DEFINITION = CodeTemplate(
    """\
${return_type} ${name}(${args_str}) {
  ${unpack_out}
  ${type_definition_body}
}

"""
)

TRACE_DISPATCH = CodeTemplate(
    """\
return ${impl_name}(${args_exprs_str});"""
)


@with_native_function
def compute_op_definition(f: NativeFunction):
    if is_backend_agnostic(f):
        return []
    out_num = len(f.func.arguments.out)
    sig = DispatcherSignature.from_schema(f.func, prefix=f"wrapper_{f.func.name.overload_name}_")
    name = sig.name()
    args = sig.arguments()
    args_str = ", ".join(a.defn() for a in args)

    args_exprs_str = ", ".join(a.name for a in args)

    impl_name = f"at::supa::SUPANativeFunctions::{cpp.name(f.func)}"

    check_out = [f'TORCH_CHECK(out.size() == {out_num}, "expected tuple of {out_num} elements but got ", out.size());']
    unpack_out = (
        check_out + [f"at::Tensor {args[-out_num + i].name} = out[{i}];" for i in range(out_num)] if out_num > 1 else ""
    )
    out_return_type = "::std::tuple<{}>".format(", ".join(["at::Tensor"] * out_num))

    return [
        METHOD_DEFINITION.substitute(
            return_type=out_return_type if out_num > 1 else cpp.returns_type(f.func.returns).cpp_type(),
            name=name,
            args_str=",".join(a.defn() for a in args[:-out_num]) + ", at::TensorList out" if out_num > 1 else args_str,
            unpack_out=unpack_out,
            type_definition_body=[TRACE_DISPATCH.substitute(impl_name=impl_name, args_exprs_str=args_exprs_str)],
        )
    ]


def is_backend_agnostic(f: NativeFunction):
    return "backend_agnostic" in f.tags


@with_native_function
def compute_register_symbol(f: NativeFunction):
    out_num = len(f.func.arguments.out)
    if out_num > 1:
        decl = re.compile(r"(?P<name>[^\(]+)\((?P<args>.*)\) -> (?P<returns>.*)").findall(str(f.func))[0]
        func_schema = (
            decl[0]
            + "("
            + ",".join(decl[1].split(",")[:-out_num])
            + ", Tensor[] out) -> ("
            + ", ".join(["Tensor"] * out_num)
            + ")"
        )
    else:
        func_schema = str(f.func)
    if is_backend_agnostic(f):
        """backend_agnositc OP is registered by python_torch_function.cpp
        because schema parser doesn't support device=integer.
        """
        return []
    else:
        return [f"m.def({cpp_string(func_schema)});\n"]


@with_native_function
def compute_register_impl(f: NativeFunction):
    if is_backend_agnostic(f):
        return []
    else:
        name = DispatcherSignature.from_schema(f.func, prefix=f"wrapper_{f.func.name.overload_name}_").name()
        return [f'm.impl("{f.func.name}", TORCH_FN(at::supa::native::{name}));\n']


def gen_custom_trace(fm: FileManager, custom_trace_functions: Sequence[NativeFunction]):
    """generate register code for schema.
    by registering schema, op can be accessed by 'torch.ops.supa.xxxxxx'
    """
    fm.write_with_template(
        "CustomRegisterSchema.cpp",
        "CustomRegisterSchema.cpp",
        lambda: {
            "custom_op_definitions": list(concatMap(lambda f: compute_op_definition(f), custom_trace_functions)),
            "custom_schema_registrations": list(
                concatMap(lambda f: compute_register_symbol(f), custom_trace_functions)
            ),
            "custom_impl_registrations": list(concatMap(lambda f: compute_register_impl(f), custom_trace_functions)),
        },
    )


def gen_custom_ops_patch(fm: FileManager, custom_trace_functions: Sequence[NativeFunction]):

    def fmt_string(f: NativeFunction):
        if is_backend_agnostic(f):
            return "torch_supa.{ops} = torch_supa._C._BackendAgnostic.{ops}"
        return "torch_supa.{ops} = torch.ops.custom.{ops}"

    valid_native_functions = list(filter(should_generate_ops_patch, custom_trace_functions))

    ops = set([(f.func.name.name, fmt_string(f)) for f in valid_native_functions])

    fm.write_with_template(
        "custom_ops.py", "custom_ops.py", lambda: {"custom_ops": [fmt.format(ops=name) for name, fmt in ops]}
    )


def compute_custom_functions_declaration(f: NativeFunction, func_type: str):
    with native_function_manager(f):
        sig = DispatcherSignature.from_schema(f.func)
        name = sig.name()
        args = sig.arguments()
        if func_type == "call":
            args_str = ", ".join(a.decl() for a in args)
        if func_type == "redispatch":
            args_str = "c10::DispatchKeySet dispatchKeySet, " + ", ".join(a.decl() for a in args)

        if (func_type == "call") and (name == "supa_slice_out"):
            return [
                EXPORT_CUSTOM_FUNCTIONS_DECLARATION.substitute(
                    return_type=cpp.returns_type(f.func.returns).cpp_type(),
                    func_name=name,
                    args_str=args_str,
                )
            ]

        return [
            CUSTOM_FUNCTIONS_DECLARATION.substitute(
                return_type=cpp.returns_type(f.func.returns).cpp_type(),
                func_name=name,
                args_str=args_str,
            )
        ]


def compute_custom_functions_definition(f: NativeFunction, func_type: str):
    with native_function_manager(f):
        sig = DispatcherSignature.from_schema(f.func)
        name = sig.name()
        args = sig.arguments()
        if func_type == "call":
            args_str = ", ".join(a.defn() for a in args)
            args_exprs_str = ", ".join(a.name for a in args)
        if func_type == "redispatch":
            args_str = "c10::DispatchKeySet dispatchKeySet, " + ", ".join(a.defn() for a in args)
            args_exprs_str = "dispatchKeySet, " + ", ".join(a.name for a in args)

        return [
            CUSTOM_FUNCTIONS_DEFINITION.substitute(
                return_type=cpp.returns_type(f.func.returns).cpp_type(),
                base_name=f.func.name.name,
                func_name=name,
                overload=f.func.name.overload_name,
                args_str=args_str,
                func_type=func_type,
                schema=sig.type(),
                args_exprs_str=args_exprs_str,
            )
        ]


def gen_custom_functions_dispatch(fm: FileManager, custom_functions: Sequence[NativeFunction]) -> None:
    func_type_list = ["call", "redispatch"]
    file_name_list = ["CustomFunctions", "CustomRedispatch"]

    for func_type, file_name in zip(func_type_list, file_name_list):
        fm.write_with_template(
            f"{file_name}.h",
            f"{file_name}.h",
            lambda: {
                "custom_function_declarations": list(
                    concatMap(lambda f: compute_custom_functions_declaration(f, func_type), custom_functions)
                )
            },
        )

        fm.write_with_template(
            f"{file_name}.cpp",
            f"{file_name}.cpp",
            lambda: {
                "custom_function_definitions": list(
                    concatMap(lambda f: compute_custom_functions_definition(f, func_type), custom_functions)
                )
            },
        )
