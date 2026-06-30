# Generates VariableType.h/cpp
#
# **If any changes are being made to the VariableType codegen please also check
# if updates are needed in torch/csrc/autograd/autograd_not_implemented_fallback.cpp
#
# VariableType is a subclass of at::Type that provides the binding code
# necessary to provide a differentiable version of ATen operators. There are a
# number of different things we could mean:
#
#   - Given a non-differentiable forward implementation, we might
#     directly associate it with a backward implementation to make
#     it differentiable.  This is the common case.
#
#   - Some functions don't need a backwards implementation, because
#     backpropagation will never propagate beyond them.  There are a
#     number of different reasons why this may be the case:
#
#       - The function has no differentiable inputs
#       - The function's output is not differentiable
#       - The function has no data dependency on its input
#
#   - Some function don't need a backwards implementation because they
#     are implemented as a composition of other (differentiable) ATen
#     functions.  These are dispatched directly to the Type superclass,
#     which will in turn dispatch back to VariableType for its
#     differentiable subcomponents.
#
import re
from typing import List

from torchgen.api import cpp
from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo

from torchgen.code_template import CodeTemplate
from torchgen.context import (
    native_function_manager,
)
from torchgen.utils import FileManager

from torchgen.packaged.autograd.gen_inplace_or_view_type import (
    gen_formals,
    METHOD_DEFINITION,
    use_derived,
)
from torchgen.packaged.autograd.gen_trace_type import (
    type_wrapper_name,
)
from torchgen.packaged.autograd.gen_variable_type import emit_body, gen_wrapper_registration


def gen_supa_variable_type(
    out: str,
    fns_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:
    """Generate VariableTypeSUPA.cpp body

    Generate variable type definition for supa method here.
    """
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)

    try_jit_decomposition_pattern = (
        r'if \(\(.*?\)\) \{.*?static c10::OperatorName full_name\("aten::.*?", .*?\);\n.*?'
        r"return impl::run_jit_decomposition_with_args_for_jvp<.*?>"
        r'\(".*?", \*opt_op, ks, .*?\);\n\s*\} '
        r"else \{\n\s*(.*?)\n\s*\}"
    )
    use_count_pattern = (
        r"if \(\S+\.has_storage\(\) && !at::impl::dispatch_mode_enabled\(\) && "
        r"!at::impl::tensor_has_dispatch\(\S+\)\) {\s+TORCH_INTERNAL_ASSERT\("
        r'\S+\.storage\(\)\.use_count\(\) == 1, "function: \S+"\);\s+}'
    )

    supa_method_definitions: List[str] = []
    wrapper_registrations_supa: List[str] = []
    wrapper_registrations_atan: List[str] = []
    for fn in fns_with_diff_infos:
        if use_derived(fn):
            f = fn.func
            with native_function_manager(f):
                formals = gen_formals(f)
                for key in fn.info.keys():
                    type_definition = METHOD_DEFINITION.substitute(
                        return_type=cpp.returns_type(f.func.returns).cpp_type(),
                        type_wrapper_name=type_wrapper_name(f),
                        type_definition_body=emit_body(fn, key),
                        formals=formals,
                    )
                    type_definition = re.sub(try_jit_decomposition_pattern, r"\1", type_definition, flags=re.DOTALL)
                    type_definition = re.sub(use_count_pattern, "", type_definition, flags=re.DOTALL)
                    if "native_fwd" not in f.tags:
                        type_definition = type_definition.replace("at::redispatch", "at::supa::redispatch")
                        wrapper_registrations_supa.append(gen_wrapper_registration(f, key))
                    else:
                        wrapper_registrations_atan.append(gen_wrapper_registration(f, key))
                supa_method_definitions.append(type_definition)

    fm.write_with_template(
        "VariableType.cpp",
        "VariableType.cpp",
        lambda: {
            "type_derived_method_definitions": supa_method_definitions,
            "wrapper_registrations_aten": wrapper_registrations_atan,
            "wrapper_registrations_supa": wrapper_registrations_supa,
        },
    )

    """Generate VariableType.h body
    """
    METHOD_HEADER_DEFINITION = CodeTemplate(
        """\
${return_type} ${type_wrapper_name}(${formals});
"""
    )
    variable_type_header: List[str] = []
    for fn in fns_with_diff_infos:
        f = fn.func
        with native_function_manager(f):
            formals = gen_formals(f)
            wrapper_name = type_wrapper_name(f)

            type_header_definition = METHOD_HEADER_DEFINITION.substitute(
                return_type=cpp.returns_type(f.func.returns).cpp_type(),
                type_wrapper_name=wrapper_name,
                formals=formals,
            )
            variable_type_header.append(type_header_definition)

    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template(
        "VariableType.h",
        "VariableType.h",
        lambda: {
            "generated_comment": "@" f"generated from {template_path}/VariableType.h",
            "supa_variable_type": variable_type_header,
        },
    )
