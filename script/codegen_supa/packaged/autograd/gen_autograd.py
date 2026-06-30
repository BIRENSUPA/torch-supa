"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.autograd.gen_autograd \
       aten/src/ATen/native/native_functions.yaml \
       aten/src/ATen/native/tags.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/csrc/autograd/generated/
"""

# gen_autograd.py generates C++ autograd functions and Python bindings.
#
# It delegates to the following scripts:
#
#  gen_autograd_functions.py: generates subclasses of torch::autograd::Node
#  gen_variable_type.py: generates VariableType.h which contains all tensor methods
#  gen_python_functions.py: generates Python bindings to THPVariable
#

import inspect
import os
import re
from typing import List

from torchgen.api import cpp
from torchgen.api.autograd import (
    match_differentiability_info,
    NativeFunctionWithDifferentiabilityInfo,
)
from torchgen.model import NativeFunction


def patch_sys_code_for_tensor_array():
    from torchgen.packaged.autograd import gen_variable_type, gen_autograd_functions

    func_obj = gen_variable_type.emit_body
    sources = inspect.getsource(func_obj)

    sources = "from typing import Sequence\n" + sources
    sources = re.sub(r"^(\s*)(body.append\(emit_save_outputs\(\))", r"\1is_foreach=True;\n\1\2", sources, 1, re.M)
    sources = sources.replace(
        "sorted(saved_variables,", "sorted(filter(lambda arg: arg.expr is not None, saved_variables),", 1
    )
    sources = re.sub(
        r"^(\s*)(body.append\(emit_call\(f, )",
        r"\1has_hidden_outputs = any(map(lambda o: o.expr is None, info.all_saved_outputs))\1\2",
        sources,
        1,
        re.M,
    )
    sources = re.sub(
        r"^(\s*)(any_has_forward_grad = \()",
        r'\1if has_hidden_outputs: guard = guard + "\\nGradFnHolder<decltype(grad_fn)::element_type> holder(grad_fn);"\n\1\2',
        sources,
        1,
        re.M,
    )

    codes = compile(sources, inspect.getfile(func_obj), "exec")
    exec(codes, gen_variable_type.__dict__)

    func_obj = gen_autograd_functions.process_function
    sources = inspect.getsource(func_obj)
    sources = "from typing import Sequence\n" + sources.replace(
        'info.func.func.name.name.base.startswith("_foreach") and is_output', "is_output", 1
    )
    codes = compile(sources, inspect.getfile(func_obj), "exec")
    exec(codes, gen_autograd_functions.__dict__)


patch_sys_code_for_tensor_array()

from torchgen.packaged.autograd.gen_autograd_functions import gen_autograd_functions_lib
from torchgen.packaged.autograd.gen_inplace_or_view_type import gen_inplace_or_view_type
from torchgen.packaged.autograd.gen_variable_factories import is_factory_function, process_function
from .gen_variable_type import gen_supa_variable_type


def gen_custom_variable_factories(
    out: str, custom_autograd_functions: List[NativeFunction], template_path: str
) -> None:
    """code based on torchgen.packaged.autograd.gen_variable_factories.gen_variable_factories()
    changed its interface so that no need to parse yaml again.
    """
    factory_functions = [fn for fn in custom_autograd_functions if is_factory_function(fn)]
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template(
        "variable_factories.h",
        "variable_factories.h",
        lambda: {
            "generated_comment": "@" + f"generated from {fm.template_dir}/variable_factories.h",
            "ops_headers": [f"#include <ATen/ops/{fn.root_name}.h>" for fn in factory_functions],
            "function_definitions": list(mapMaybe(process_function, factory_functions)),
        },
    )


def gen_autograd_custom(
    autograd_dir: str, custom_funcs: List[NativeFunction], native_funcs: List[NativeFunction]
) -> None:
    """generate code to register backward function into pytorch autograd structure.

    Args:
        autograd_dir (str): src folder of resources, config yaml.
        custom_funcs (List[NativeFunction]): list of custom native functions
    """
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    from torchgen.packaged import autograd

    torch_template_path = os.path.join(os.path.dirname(os.path.abspath(autograd.__file__)), "templates")
    out = os.path.join(autograd_dir, "generated")

    differentiability_infos = load_custom_derivatives(
        os.path.join(autograd_dir, "derivatives.yaml"), custom_funcs, native_funcs
    )

    # collect custom functions by definition of derivatives.yaml
    custom_funcs = [info.func for x in differentiability_infos.values() for info in x.values()]

    funcs_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = None
    funcs_with_diff_infos = match_differentiability_info(custom_funcs, differentiability_infos)

    AUTOGRAD_BLACK_LIST = {"supa_format_cast.Tensor", "supa_format_cast_", "supa_format_cast_.acl_format"}
    funcs_with_diff_infos = [f for f in funcs_with_diff_infos if str(f.func.func.name) not in AUTOGRAD_BLACK_LIST]

    # Generate VariableType.h/cpp
    gen_supa_variable_type(out, funcs_with_diff_infos, template_path)

    # Generate ADInplaceOrView cpp
    gen_inplace_or_view_type(out, None, None, funcs_with_diff_infos, template_path)

    # Generate Functions.h/cpp
    gen_autograd_functions_lib(out, differentiability_infos, template_path)

    # Generate variable_factories.h
    gen_custom_variable_factories(out, custom_funcs, torch_template_path)


from torchgen.api.python import PythonSignatureNativeFunctionPair, signature
from torchgen.context import with_native_function
from torchgen.packaged.autograd import gen_python_functions


def cpp_dispatch_target(f: NativeFunction) -> str:
    symint = f.func.has_symint()
    name = cpp.name(f.func, symint_overload=symint)
    if "backend_agnostic" in f.tags:
        return f"at::supa::SUPANativeFunctions::{name}"
    raise RuntimeError(f"could not dispatch, neither function nor method: {f.func}")


from typing import Optional, Tuple, Sequence, List
from torchgen.model import BaseOperatorName
from torchgen.api.python import PythonSignature
from torchgen.api.types.signatures import DispatcherSignature, CppSignatureGroup
from torchgen.api.translate import translate


def cpp_dispatch_exprs(
    f: NativeFunction,
    *,
    python_signature: Optional[PythonSignature] = None,
) -> Tuple[str, ...]:
    sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)
    for sig in sig_group.signatures():
        # See Note [The ATen Operators API]
        target_sig = DispatcherSignature.from_schema(f.func)
        exprs = translate(sig.arguments(), target_sig.arguments())
        return (e.expr for e in exprs)


def gen_has_torch_function_check(name: BaseOperatorName, module: Optional[str], *, noarg: bool, method: bool) -> str:
    if noarg:
        if method:
            return f"""\
if(check_has_torch_function(self_)) {{
  return handle_torch_function(self_, "{name}");
}}
"""
        else:
            return ""

    self_ = "self_" if method else "nullptr"
    namespace = (
        {
            "torch_supa": "THPTorchSUPAFunctionsModule",
            "torch": "THPVariableFunctionsModule",
            "torch.nn": "THPNNVariableFunctionsModule",
            "torch.fft": "THPFFTVariableFunctionsModule",
            "torch.linalg": "THPLinalgVariableFunctionsModule",
            "torch.nested": "THPNestedVariableFunctionsModule",
            "torch.sparse": "THPSparseVariableFunctionsModule",
            "torch.special": "THPSpecialVariableFunctionsModule",
        }[module]
        if module
        else "THPVariableClass"
    )

    return f"""\
if(_r.has_torch_function()) {{
  // TODO: not ready yet. need to set correct Module obj for torch_supa.
  return handle_torch_function(_r, {self_}, args, kwargs, {namespace}, "{module or "torch.Tensor"}");
}}
"""


def method_def(
    name: BaseOperatorName,
    module: Optional[str],
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
) -> str:
    """
    Generate method def entry.
    """
    pycname = gen_python_functions.get_pycname(name)

    if name.dunder_method:
        # PyMethodDef entry for binary op, throws not implemented error
        pycname = f"TypeError_to_NotImplemented_<{pycname}>"

    if gen_python_functions.is_noarg(overloads):
        flags = "METH_NOARGS" if method else "METH_VARARGS | METH_KEYWORDS"
    else:
        pycname = f"castPyCFunctionWithKeywords({pycname})"
        flags = "METH_VARARGS | METH_KEYWORDS"

    if module == "torch" or module == "torch_supa":
        flags += " | METH_STATIC"

    return f'{{"{name}", {pycname}, {flags}, NULL}},'


def gen_python_interfaces(autograd_dir: str, custom_funcs: List[NativeFunction]):
    """generate python bindings for custom OP with backend_agnostic tag.
       Default JIT schema parser doesn't handle situation of device=interger. needs manaully python binding to take care of it.
       note: 'backend_agnostic' tag means it is backend independent, generally it is used to create tensor.

    Args:
        autograd_dir (str): target directory for holding output file.
        custom_funcs (List[NativeFunction]): list of custom OPs.
    """
    funcs = filter(lambda x: "backend_agnostic" in x.tags, custom_funcs)

    @with_native_function
    def gen_signature_pairs(f: NativeFunction) -> PythonSignatureNativeFunctionPair:
        return PythonSignatureNativeFunctionPair(
            signature=signature(f),
            function=f,
        )

    pairs = list(map(gen_signature_pairs, funcs))
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    out = os.path.join(autograd_dir, "generated")
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)

    gen_python_functions.cpp_dispatch_target = cpp_dispatch_target
    gen_python_functions.cpp_dispatch_exprs = cpp_dispatch_exprs
    gen_python_functions.gen_has_torch_function_check = gen_has_torch_function_check
    gen_python_functions.method_def = method_def
    gen_python_functions.create_python_bindings_sharded(
        fm,
        pairs=pairs,
        pred=lambda x: True,
        module="torch_supa",
        filename="python_torch_functions.cpp",
        num_shards=1,
        method=False,
    )
