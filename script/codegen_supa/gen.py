from torchgen.code_template import CodeTemplate
from .custom_functions import gen_custom_trace, gen_custom_ops_patch, gen_custom_functions_dispatch
from .utils import (
    assert_never,
    concatMap,
    context,
    FileManager,
    make_file_manager,
    mapMaybe,
    NamespaceHelper,
    Target,
    SwitchedTemplateFolder,
)
from torchgen.yaml_utils import YamlDumper, YamlLoader
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.native_function_generation import (
    add_generated_native_functions,
    pre_group_native_functions,
)
import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Literal,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import yaml
from . import supa_utils
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
from . import dest

from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
    Binding,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    NamedCType,
    NativeSignature,
    SpecialArgName,
)
from torchgen.context import (
    method_with_native_function,
    native_function_manager,
    with_native_function,
    with_native_function_and_indices,
)

from torchgen.gen import error_check_native_functions, cpp_string

from torchgen.model import (
    Argument,
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    FRAGMENT_NAMESPACES,
    FunctionSchema,
    is_cuda_dispatch_key,
    Location,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    OperatorName,
    OptionalType,
    SchemaKind,
    SelfArgument,
    STRUCTURED_DISPATCH_KEYS,
    TensorOptionsArguments,
    Type,
    Variant,
    ViewSchemaKind,
)

DEFAULT_KERNEL_NAMESPACE = "at"


T = TypeVar("T")

# Welcome to the ATen code generator v2!  The ATen code generator is
# responsible for parsing native_functions.yaml and then generating
# various generated files (e.g., TypeDefault.cpp) based on the operators
# defined in this file.  This means that the code generator knows how to
# parse function schema, and then translate this into various C++ types
# and boilerplate code.
#
# Some things to know about this file when you modify it:
#
# - This file has STRICT mypy typechecking.  Typecheck it with
#   `mypy --config mypy-strict.ini` in the root source directory
#
# - Most of the heavy lifting lives in external modules:
#   - 'model' has the data model for native_functions.yaml.  The classes
#     in those file represent what you see when you look at
#     a native_functions.yaml
#   - 'api' has conversions for how to translate JIT schema into
#     the various C++ APIs that the codegen interacts with.  There
#     are in fact THREE different C++ APIs: the public C++ API,
#     the dispatcher API, and the legacy dispatcher API.  See each
#     of these respective files for more information

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         HELPER FUNCTIONS
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

local_templates_folder: str = os.path.join(pathlib.Path(__file__).parent.resolve(), "templates")

# A custom loader for YAML to let us also keep track of line numbers
# of each entry in the YAML file


class LineLoader(YamlLoader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        # Add 1 so line numbering starts at 1
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}
_GLOBAL_PARSE_TAGS_YAML_CACHE = {}

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple("ParsedYaml", ["native_functions", "backend_indices"])


def parse_native_yaml_struct(
    es: object,
    valid_tags: Set[str],
    ignore_keys: Optional[Set[DispatchKey]] = None,
    path: str = "<stdin>",
    skip_native_fns_gen: bool = False,
) -> ParsedYaml:
    assert isinstance(es, list)
    rs: List[NativeFunction] = []
    bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    for e in es:
        assert isinstance(e.get("__line__"), int), e
        loc = Location(path, e["__line__"])
        funcs = e.get("func")
        with context(lambda: f"in {loc}:\n  {funcs}"):
            func, m = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
            rs.append(func)
            BackendIndex.grow_index(bs, m)
    error_check_native_functions(rs)
    # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
    indices: Dict[DispatchKey, BackendIndex] = defaultdict(
        lambda: BackendIndex(
            dispatch_key=DispatchKey.Undefined,
            use_out_as_primary=True,
            external=False,
            device_guard=False,
            # I'm actually not sure about this; undefined could be hit on
            # empty TensorList, hypothetically that could have sizes in it
            index={},
        )
    )
    if not skip_native_fns_gen:
        add_generated_native_functions(rs, bs)
    for k, v in bs.items():
        # All structured in-tree operators are implemented in terms of their out operator.
        indices[k] = BackendIndex(
            dispatch_key=k,
            use_out_as_primary=True,
            external=False,
            # Only cuda-like devices in tree require device guards
            device_guard=is_cuda_dispatch_key(k),
            index=v,
        )
    return ParsedYaml(rs, indices)


def parse_tags_yaml_struct(es: object, path: str = "<stdin>") -> Set[str]:
    assert isinstance(es, list)
    rs: Set[str] = set()
    for e in es:
        assert isinstance(e.get("__line__"), int), e
        loc = Location(path, e["__line__"])
        tags = e.get("tag")
        with context(lambda: f"in {loc}:\n  {tags}"):
            e_i = e.copy()
            name = e_i.pop("tag")
            desc = e_i.pop("desc", "")
            # ensure that each tag has a non-empty description
            assert desc != ""
            rs.add(name)
    return rs


@functools.lru_cache(maxsize=None)
def parse_tags_yaml(path: str) -> Set[str]:
    if path not in _GLOBAL_PARSE_TAGS_YAML_CACHE:
        with open(path) as f:
            es = yaml.load(f, Loader=LineLoader)
            # add backend_agnostic tag for custom op.
            es.append({"tag": "backend_agnostic", "desc": "no backend binded", "__line__": 0})
            es.append({"tag": "fused_torch_api", "desc": "fused backend implementation for a Python-level torch API", "__line__": 0})
            _GLOBAL_PARSE_TAGS_YAML_CACHE[path] = parse_tags_yaml_struct(es, path=path)

    return _GLOBAL_PARSE_TAGS_YAML_CACHE[path]


def parse_native_yaml(
    path: str,
    tags_yaml_path: str,
    ignore_keys: Optional[Set[DispatchKey]] = None,
    *,
    skip_native_fns_gen: bool = False,
    loaded_yaml: Optional[object] = None,
) -> ParsedYaml:
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tags_yaml_path)

        # if a loaded yaml is provided, use that instead of reading from path
        if loaded_yaml is None:
            with open(path) as f:
                es = yaml.load(f, Loader=LineLoader)
        else:
            es = loaded_yaml

        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = parse_native_yaml_struct(
            es,
            valid_tags,
            ignore_keys,
            path=path,
            skip_native_fns_gen=skip_native_fns_gen,
        )

    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        C++ CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Most functions in this section are curried: they consist of a function
# that takes some parameters (e.g., what is to be generated) which itself
# returns a function that actually maps NativeFunction to the code
# to be generated.  This pattern makes it convenient to use map, concatMap
# and similar functional combinators.


def static_dispatch_keys(backends: List[BackendIndex]) -> List[DispatchKey]:
    if len(backends) == 0:
        return []
    else:
        return [backend.dispatch_key for backend in backends] + [
            DispatchKey.CompositeImplicitAutograd,
            DispatchKey.CompositeImplicitAutogradNestedTensor,
            DispatchKey.CompositeExplicitAutograd,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
        ]


def get_static_dispatch_backend(f: NativeFunction, backend_index: BackendIndex) -> Optional[DispatchKey]:
    if f.structured_delegate is not None or backend_index.has_kernel(f):
        # TODO: for ops with structured_delegate it should check the dispatch table of
        # the out variant instead. For now, these structured ops all have CPU/CUDA kernels
        # so we always dispatch to the `backend`, but this could be wrong when we
        # migrate math/default_backend ops to use structured delegate.
        return backend_index.dispatch_key
    elif f.has_composite_explicit_autograd_kernel:
        return DispatchKey.CompositeExplicitAutograd
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return DispatchKey.CompositeExplicitAutogradNonFunctional
    elif f.has_composite_implicit_autograd_kernel:
        return DispatchKey.CompositeImplicitAutograd
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return DispatchKey.CompositeImplicitAutogradNestedTensor
    return None


def static_dispatch_ops_header(f: NativeFunction, backend_index: List[BackendIndex]) -> Optional[str]:
    if backend_index is None or f.manual_kernel_registration:
        return None

    output = []
    for index in backend_index:
        dispatch_key = get_static_dispatch_backend(f, index)
        if dispatch_key is not None:
            output.append(f"#include <ATen/ops/{f.root_name}_{dispatch_key.lower()}_dispatch.h>")
    return "\n".join(output)


def static_dispatch_extra_headers(backends: List[BackendIndex]) -> List[str]:
    return [f"#include <ATen/{dispatch_key}Functions.h>" for dispatch_key in static_dispatch_keys(backends)]


# Translates arguments of `sig` to CppSignature bindings.
# Note that we have a special case for `memory_format` argument and this case is not covered by
# tools.codegen.api.translate() yet as its application is limited to static dispatch.
def translate_args(
    sig: Union[CppSignature, DispatcherSignature],
    cpp_sig: CppSignature,
) -> str:
    # Adds SpecialArgName.possibly_redundant_memory_format NamedCType for memory_format bindings
    def add_spl_memory_format_binding(input_bindings: List[Binding]) -> List[Binding]:
        output_bindings: List[Binding] = []
        for binding in input_bindings:
            if binding.name == "memory_format":
                spl_mem_format_binding = Binding(
                    nctype=NamedCType(
                        SpecialArgName.possibly_redundant_memory_format,
                        binding.nctype.type,
                    ),
                    name=binding.name,
                    default=binding.default,
                    argument=binding.argument,
                )
                output_bindings.append(spl_mem_format_binding)
            else:
                output_bindings.append(binding)
        return output_bindings

    src_bindings = list(sig.arguments())
    goal_bindings = list(cpp_sig.arguments())
    # When last argument of CPP signature has SpecialArgName.possibly_redundant_memory_format NCType,
    # get memory_format bindings of dispatcher signature to have the same NCType as well
    for arg in goal_bindings:
        if arg.nctype.name == SpecialArgName.possibly_redundant_memory_format:
            src_bindings = add_spl_memory_format_binding(src_bindings)
            break
    exprs = translate(src_bindings, goal_bindings)
    return ", ".join(a.expr for a in exprs)


def generate_static_dispatch_backend_call(
    sig: Union[CppSignature, DispatcherSignature],
    f: NativeFunction,
    backend_index: BackendIndex,
) -> str:
    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    backend_metadata = backend_index.get_kernel(f)
    kernel_ns = (
        backend_metadata.cpp_namespace
        if backend_metadata and backend_metadata.cpp_namespace
        else DEFAULT_KERNEL_NAMESPACE
    )
    ns = kernel_ns.replace("::native", "")
    return f"return {ns}::{backend_index.dispatch_key.lower()}::{name}({exprs});"


def generate_static_dispatch_fallback_call(
    sig: Union[CppSignature, DispatcherSignature],
    f: NativeFunction,
    backend_indices: List[BackendIndex],
) -> str:
    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    ns = DEFAULT_KERNEL_NAMESPACE.replace("::native", "")
    if f.has_composite_explicit_autograd_kernel:
        return f"return {ns}::{DispatchKey.CompositeExplicitAutograd.lower()}::{name}({exprs});"
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return f"return {ns}::{DispatchKey.CompositeExplicitAutogradNonFunctional.lower()}::{name}({exprs});"
    elif f.has_composite_implicit_autograd_kernel:
        return f"return {ns}::{DispatchKey.CompositeImplicitAutograd.lower()}::{name}({exprs});"
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return f"return {ns}::{DispatchKey.CompositeImplicitAutogradNestedTensor.lower()}::{name}({exprs});"
    else:
        return f"""TORCH_CHECK(false, "Static dispatch does not support {name} for\
{', '.join([str(index.dispatch_key)for index in backend_indices])} ");"""


def static_dispatch(
    sig: Union[CppSignature, DispatcherSignature],
    f: NativeFunction,
    backend_indices: List[BackendIndex],
) -> str:
    """
    For a given `NativeFunction`, find out the corresponding backend and dispatch to it. If more than one
    backends exsit, fallback to static dispatch by determining dispatch key from inputs.
    Arguments:
        sig: A CppSignature or DispatcherSignature for this native function we want to use.
        f: NativeFunction to generate static dispatch.
        backend_indices: All available backends.
    Return:
        C++ code to call backend-specific functions, e.g., "return at::cpu::add(self, other, scale);"
    """
    if len(backend_indices) == 0 or f.manual_kernel_registration:
        return ""

    keys = [
        b
        for b in backend_indices
        if b.has_kernel(f) or (f.structured_delegate is not None and b.dispatch_key in STRUCTURED_DISPATCH_KEYS)
    ]
    if len(keys) == 1:
        return generate_static_dispatch_backend_call(sig, f, keys[0])
    elif len(keys) == 0:
        return generate_static_dispatch_fallback_call(sig, f, backend_indices)

    native_tensor_args = [
        a.name
        for a in sig.arguments()
        if isinstance(a.argument, SelfArgument) or isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
    ]
    tensor_args = ", ".join(native_tensor_args)
    tensor_opts = f.func.arguments.tensor_options

    stmts = []
    subexprs: List[str] = []
    if tensor_opts is not None:
        subexprs.append("DispatchKeySet(c10::computeDispatchKey(dtype, layout, device))")
    if tensor_args != "":
        subexprs.append(f"c10::detail::multi_dispatch_key_set({tensor_args})")
    stmts.append(f"""DispatchKeySet _dk_set = {' | '.join(subexprs)};""")
    stmts.append("DispatchKey _dk = c10::highestPriorityBackendTypeId(_dk_set);")

    dispatch_code = []
    for index in keys:
        dispatch_code.append(f"""case DispatchKey::{index.dispatch_key}:""")
        dispatch_code.append(f"""\t{generate_static_dispatch_backend_call(sig, f, index)};""")

    fallback = generate_static_dispatch_fallback_call(sig, f, backend_indices)
    connector = "\n\t\t"

    return f"""
    {connector.join(stmts)}
    switch (_dk) {{
        {connector.join(dispatch_code)}
        default:
            {fallback}
    }}
    """


# Generates RegisterSchema.cpp.  Depending on the selector, either
# all schemas are registered, or only some are (in the case of
# selective build)
@dataclass(frozen=True)
class RegisterSchema:
    selector: SelectiveBuilder
    known_tags: Dict[str, int] = field(default_factory=dict)

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if not self.selector.is_native_function_selected(f):
            return None
        tags = "{" + ", ".join(f"at::Tag::{tag}" for tag in sorted(f.tags)) + "}"
        if tags == "{}":
            return f"m.def({cpp_string(str(f.func))});\n"
        maybe_tags = ""
        if tags not in self.known_tags:
            idx = len(self.known_tags)
            self.known_tags[tags] = idx
            maybe_tags = f"const std::vector<at::Tag> tags_{idx} = {tags};\n"
        return f"{maybe_tags}m.def({cpp_string(str(f.func))}, tags_{self.known_tags[tags]});\n"


# Generates Operators.h and Operators.cpp.
# These provide macros that, given an operator and overload name, allow users
# to access an "un-overloaded" function version of the operator. This
# is useful for extension writers who want to (1) want to decltype the operator
# and (2) don't want to worry about method-only operators.
@dataclass(frozen=True)
class ComputeOperators:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: List[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig = DispatcherSignature.from_schema(f.func)
        name = f.func.name.unambiguous_name()

        if self.target is Target.DECLARATION:
            # Note [The ATen Operators API]
            # The ATen Operators API lives in the at::_ops namespace, and contains compile-time
            # metadata about each operator + entry points into the Dispatcher.
            # The C++ function, method, and redispatch API's are all implemented as wrappers
            # into various bits of the structs defined here.
            #
            # Important characteristics about the Operators API:
            # (1) It follows the Dispatcher API.
            #     This is kind of necessary to avoid overhead.
            #     For example: if it followed the C++ API, then all of the faithful C++ factory functions
            #     would need to wrap their arguments into TensorOptions only to unwrap them again.
            # (2) Overload names are disambiguated.
            #     This is helpful for pytorch extenders who would like to decltype() an aten operator,
            #     that has overloads, e.g. decltype(at::_ops::mul_Tensor::call)
            # (3) No argument defaulting is allowed.
            #     This is more of an implementation detail to avoid #include cycles,
            #     since TensorBody.h (which defines the Tensor class) needs to include this file.
            # (4) manual_cpp_bindings and faithful names are not included in the API.
            #     This applies to stuff like __dispatch__is_complex(), and add_outf().
            #     These aren't "real aten ops", they're just additional functions provided by the C++ API.
            #     They're implemented as wrappers in Functions.h that call into the actual operators
            #     defined here, i.e. at::_ops::is_complex::call() and at::_ops::add_out::call().
            #     This means that ATEN_OP(is_complex) will not fastpath, and will go through the dispatcher.
            return f"""
struct TORCH_API {name} {{
  using schema = {sig.type()};
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::{f.func.name.name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "{f.func.name.overload_name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, {cpp_string(str(f.func))})
  static {sig.defn(name="call", is_redispatching_fn=False)};
  static {sig.defn(name="redispatch", is_redispatching_fn=True)};
}};"""

        elif self.target is Target.DEFINITION:
            defns = f"""
// aten::{f.func}
static C10_NOINLINE c10::TypedOperatorHandle<{name}::schema> create_{name}_typed_handle() {{
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow({name}::name, {name}::overload_name)
      .typed<{name}::schema>();
}}
"""
            for is_redispatching_fn in [False, True]:
                if is_redispatching_fn:
                    dispatcher_exprs_str = ", ".join(["dispatchKeySet"] + [a.name for a in sig.arguments()])
                    method_base = "redispatch"
                else:
                    dispatcher_exprs_str = ", ".join([a.name for a in sig.arguments()])
                    method_base = "call"

                dispatcher_call = method_base
                method_name = f"{name}::{method_base}"

                fn_body = f"""
    static auto op = create_{name}_typed_handle();
    return op.{dispatcher_call}({dispatcher_exprs_str});"""

                if not is_redispatching_fn and len(self.static_dispatch_backend_indices) > 0:
                    # call() should go through static dispatch
                    fn_body = static_dispatch(sig, f, backend_indices=self.static_dispatch_backend_indices)
                defns += f"""
// aten::{f.func}
{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{
    {fn_body}
}}
"""
            return defns
        else:
            assert_never(self.target)


# Generates Functions.h, which provides the functional public C++ API,
# and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeFunction:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)
        has_symint = f.func.has_symint()

        result = ""
        for sig in sig_group.signatures():
            # See Note [The ATen Operators API]
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ", ".join([e.expr for e in exprs])

            if sig.symint:
                intlike_t = "c10::SymInt"
            else:
                intlike_t = "int64_t"

            if Variant.function in f.variants:
                result += f"""
// aten::{f.func}
inline {sig.decl()} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}"""

            # The template function can be used from template situations
            # where you want to switch between the symint or not version
            # depending on a template argument
            #
            # NB: we ALWAYS generate this even for methods.  But we put it in
            # this header so it can take advantage of per-op headers
            if has_symint:
                result += f"""
namespace symint {{
  template <typename T, typename = std::enable_if_t<std::is_same<T, {intlike_t}>::value>>
  {sig.decl(suppress_symint_suffix=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
  }}
}}
"""
        return result


# Generates TensorBody.h. This file provides the object-oriented (method-based)
# public C++ API, and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeTensorMethod:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: List[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.method not in f.variants:
            return None

        assert not f.func.is_out_fn()
        assert f.func.arguments.self_arg is not None

        sig_group = CppSignatureGroup.from_native_function(f, method=True, fallback_binding=f.manual_cpp_binding)

        if self.target is Target.DECLARATION:
            result = ""
            for sig in sig_group.signatures():
                result += f"{sig.decl()} const;\n"
            return result

        if self.target is not Target.DEFINITION:
            assert_never(self.target)

        result = ""

        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments(), method=True)
            exprs_str = ", ".join([e.expr for e in exprs])

            result += f"""
// aten::{f.func}
inline {sig.defn(prefix="Tensor::")} const {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}
"""

        return result


# Generates RedispatchFunctions.h.
# This is similar to the C++ API defined in Functions.h, but provides access
# to the dispatcher's redispatch API.
@dataclass(frozen=True)
class ComputeRedispatchFunction:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        # We unconditionally generate function variants of the redispatch API.
        # This is mainly because we can namespace functions separately, but not methods,
        sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)

        result = ""
        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ", ".join(["dispatchKeySet"] + [a.expr for a in exprs])

            result += f"""
// aten::{f.func}
inline {sig.decl(is_redispatching_fn=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::redispatch({exprs_str});
}}
"""

        return result


# Generates ATenOpList.cpp, a runtime accessible list of all aten
# operators.
# TODO: This was historically used to help some JIT interop code
# figure out whether or not to treat aten namespace'd operators
# one way or another, we should reevaluate if this is actually needed.
@with_native_function
def compute_aten_op(f: NativeFunction) -> str:
    return f'{{"aten::{f.func.name.name}", "{f.func.name.overload_name}"}},'


# Generates MetaFunctions.h
def compute_meta_function_declaration(g: NativeFunctionsGroup) -> Optional[str]:
    if not g.structured:
        return None
    with native_function_manager(g.out):
        name = meta.name(g)
        args = structured.meta_arguments(g)
        args_str = ", ".join(a.decl() for a in args)
        parent_class = g.out.structured_inherits
        if parent_class is None:
            parent_class = "at::impl::MetaBase"
        meta_return = "void"
        precomputed = g.out.precomputed if g.structured else None

        if precomputed:
            # Generate the template declaration with one bool parameter for each
            # precomputed element. Each parameter is true if the corresponding (in
            # terms of position) precomputed element has been set.
            precomputed_values = [*precomputed.replace.values(), precomputed.add]
            precomputed_elements = [elem for replace_list in precomputed_values for elem in replace_list]
            precomputed_template_parameters = [elem.name.upper() for elem in precomputed_elements]
            precomputed_template_params_str = ", ".join(
                f"bool {param} = false" for param in precomputed_template_parameters
            )
            precompute_template_decl = f"template <{precomputed_template_params_str}>"

            # Generate a string containing declarations of all precomputed elements.
            precomputed_elements_with_cpp_types = [
                structured.argument_type(elem, binds=elem.name) for elem in precomputed_elements
            ]

            precomputed_elements_decl = ";\n".join(
                f"{elem.cpp_type(strip_ref=True)} {elem.name}" for elem in precomputed_elements_with_cpp_types
            )

            # Generate "setter" methods for each precomputed element. Each method will return
            # a new instance of precompute_out with the template parameter that corresponds to
            # the member set by the method to true (to indicate that it has been set).
            setter_methods = []
            for i, elem in enumerate(precomputed_elements):
                # Generate the signature. The return type will be the same
                # as the type of `this` but with the template parameter
                # corresponding to the element set by this method set to true.
                # The assert generated below will ensure that this template
                # parameter is false on the type of `this`.
                return_ty_templates = ", ".join(
                    precomputed_template_parameters[:i] + ["true"] + precomputed_template_parameters[i + 1 :]
                )
                return_ty = f"precompute_out<{return_ty_templates}>"
                elem_cpp_ty = precomputed_elements_with_cpp_types[i].cpp_type(strip_ref=True)
                signature = f"{return_ty} set_{elem.name}({elem_cpp_ty} value)"

                # Generate an assert which checks that the
                # template parameter corresponding to the precomputed
                # element that is set by this method is false on the
                # class corresponding to the object that `this` points to.
                # This ensures that each element can be set only once.
                assert_msg = f'"{precomputed_elements[i].name} already set"'
                assert_stmt = f"static_assert({precomputed_template_parameters[i]} == false, {assert_msg});"

                # Generate the new object construction block. All state
                # except the element that this method sets is copied from the
                # object that `this` points to. The value for the element that
                # the method sets is taken from a method parameter.
                construction_stmts = []
                construction_stmts.append(f"{return_ty} ret;")

                for j, elem in enumerate(precomputed_elements):
                    if i == j:
                        construction_stmts.append(f"ret.{elem.name} = value;")
                    else:
                        construction_stmts.append(f"ret.{elem.name} = this->{elem.name};")

                construction_stmts.append("return ret;")
                construction_block = "\n".join(construction_stmts)

                setter_methods.append(
                    f"""
                    {signature} {{
                        {assert_stmt}
                        {construction_block}
                    }}
                """
                )
            setter_methods_decl = "\n".join(setter_methods)

            # Meta should return an instance of the struct containing the precomputed elements.
            meta_return_template_params = ", ".join(["true"] * len(precomputed_template_parameters))
            # This typedef (actually a using statement) is needed so that TORCH_META_FUNC can reuse the return
            # type (which has a variable number of template parameters).
            meta_return_typedef = f"using meta_return_ty = precompute_out <{meta_return_template_params}>;"
            meta_return = "meta_return_ty"
            precomputed_decl = f"""
                {precompute_template_decl}
                struct TORCH_API precompute_out {{
                    {setter_methods_decl}
                    {precomputed_elements_decl};
            }};"""
        else:
            meta_return_typedef = ""
            precomputed_decl = ""

        return f"""\
//{supa_utils.gen_hint(g.out)}
struct TORCH_API structured_{name} : public {parent_class} {{
    {precomputed_decl}
    {meta_return_typedef}
    void impl({', '.join(a.decl() for a in structured.impl_arguments(g))});
}};
"""


def needs_backend_select(f: NativeFunction, selector: SelectiveBuilder) -> bool:
    name = str(f.func.name.name)
    if name.endswith("_like") or name.startswith("new_"):
        return False
    if f.func.arguments.tensor_options is None:
        return False
    return selector.is_native_function_selected(f)


# Generates RegisterBackendSelect.cpp, a series of kernels which provide
# specialized computation of dispatch key for operator signatures which cannot
# be easily done automatically using templating.
@dataclass(frozen=True)
class ComputeBackendSelect:
    target: Literal[Target.DEFINITION, Target.REGISTRATION]

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if not needs_backend_select(f, self.selector):
            return None

        name = native.name(f.func)
        # BackendSelect can go to Meta, so it must preserve symints
        native_sig = NativeSignature(f.func, symint=True)

        native_tensor_args = [
            a for a in native_sig.arguments() if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
        ]

        dispatcher_sig = DispatcherSignature.from_schema(f.func)

        sig: Union[NativeSignature, DispatcherSignature]
        sig = dispatcher_sig
        dispatcher_exprs = dispatcher_sig.exprs()
        dispatch_key = "c10::computeDispatchKey(dtype, layout, device)"

        if self.target is Target.DEFINITION:
            # I don't think there's actually a good reason to generate
            # these two cases differently
            # The first case could probably be improved though- it calls computeDispatchKeySet(),
            # which looks at TLS dispatch keys- there should not be any by the time we reach backend select.
            if native_tensor_args:
                assert f.func.arguments.has_tensor_arg()
                tensor_args = ", ".join(a.name for a in native_tensor_args)
                compute_dk = f"""\
DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});
DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
DispatchKeySet _dk = c10::impl::computeDispatchKeySet(_dk_set, _dk_mask);"""
            else:
                assert not f.func.arguments.has_tensor_arg()
                compute_dk = f"DispatchKeySet _dk = c10::DispatchKeySet({dispatch_key});"
            return f"""\
// aten::{f.func}
C10_ALWAYS_INLINE
{sig.defn(name)} {{
  {compute_dk}
  return at::_ops::{f.func.name.unambiguous_name()}::redispatch(
      _dk, {', '.join(a.expr for a in dispatcher_exprs)});
}}
"""
        elif self.target is Target.REGISTRATION:
            return f"""m.impl("aten::{f.func.name}", TORCH_FN({name}));"""
        else:
            assert_never(self.target)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       YAML CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def format_yaml(data: object) -> str:
    # Ignore alias in Dumper
    YamlDumper.ignore_aliases = lambda self, data: True  # type: ignore[assignment]

    # Support serializing OrderedDict
    def dict_representer(dumper: Any, data: Any) -> Any:
        return dumper.represent_dict(data.items())

    YamlDumper.add_representer(OrderedDict, dict_representer)  # type: ignore[no-untyped-call]
    # Some yaml parsers (e.g. Haskell's) don't understand line breaks.
    # width=1e9 turns off optional line breaks and improves
    # the portability of the outputted yaml.
    return yaml.dump(data, default_flow_style=False, Dumper=YamlDumper, width=1e9)  # type: ignore[no-any-return, call-overload]


# For some reason, some defaults we write to YAML are written as native
# YAML objects, rather than doing them uniformly as strings.  This
# function detects those cases and converts them into native Python
# objects.
def pythonify_default(s: str) -> object:
    if s == "true":
        return True
    elif s == "false":
        return False

    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


# What is a dynamic type?  Over time, the semantic meaning of
# dynamic type has degraded to meaninglessness (in the old days,
# it captured dtype-ness of types, but that has gone away with
# the removal of TH).  These days, it's mostly the same thing as
# the C++ API argument type, except that Tensor and Tensor?
# arguments simply present as Tensor.
#
# TODO: Get rid of dynamic_type, after getting tools/autograd
# to use the new codegen framework
def dynamic_type(t: Type) -> str:
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem)
    # Note we don't use t.is_tensor_like() here because it would
    # also include Tensor[]
    if str(t) == "Tensor":
        return "at::Tensor"
    # This is a legacy concept, so never report SymInt
    return cpp.argumenttype_type(t, mutable=False, binds="__placeholder__", symint=False).cpp_type()


def compute_method_of_yaml(variants: Set[Variant]) -> List[str]:
    # This is written out explicitly to ensure that Tensor and
    # namespace are put into the list in the right order
    method_of = ["Type"]
    if Variant.method in variants:
        method_of.append("Tensor")
    if Variant.function in variants:
        method_of.append("namespace")
    return method_of


def compute_returns_yaml(
    f: NativeFunction,
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    # Note [name and field_name]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # To understand name_to_field_name, we must first talk about this
    # schema:
    #
    #   lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
    #
    # There is something very odd about this schema: it is an out
    # variant of the function (that is to say, it will convert into
    # at::lstsq_out() in the C++ API), but the names of the output
    # return arguments don't match the keyword argument names of
    # the inputs.  It TURNS OUT that in this situation, the historical
    # Declarations.yaml we want to output is this (abbreviated to
    # only show relevant fields):
    #
    #   arguments:
    #     ...
    #   - field_name: solution
    #     name: X
    #   - field_name: QR
    #     name: qr
    #     ...
    #
    #   returns:
    #   - field_name: solution
    #     name: X
    #   - field_name: QR
    #     name: qr
    #
    # The name of the return fields is stored in 'field_name', and the
    # name of the arguments is stored in 'name'.  So when we process
    # arguments, we need a way to get at the corresponding return.  At
    # the moment, this is most conveniently done by constructing a
    # mapping from name (the argument concept) to field_name (the
    # return concept) while processing return arguments, since we don't
    # directly maintain this correspondence in the modeling of function
    # schema itself.
    #
    # See also https://github.com/pytorch/pytorch/issues/43114
    name_to_field_name: Dict[str, str] = {}

    # Compute the returns field of the YAML entry
    names = cpp.return_names(f)
    returns = []
    for i, (r, name) in enumerate(zip(f.func.returns, names)):
        ret = {
            "dynamic_type": dynamic_type(r.type),
            "name": name,
            # legacy, report ints
            "type": cpp.return_type(r, symint=False).cpp_type(),
        }

        if r.name:
            # See Note [name and field_name]
            ret["field_name"] = r.name
            if f.func.is_out_fn():
                name_to_field_name[f.func.arguments.out[i].name] = r.name

        returns.append(ret)

    return returns, name_to_field_name


# arguments in yaml roughly corresponds to the public C++ API
def compute_cpp_argument_yaml(
    cpp_a: Binding,
    *,
    schema_order: bool,
    kwarg_only_set: Set[str],
    out_arg_set: Set[str],
    name_to_field_name: Dict[str, str],
) -> object:
    if isinstance(cpp_a.argument, TensorOptionsArguments):
        arg: Dict[str, object] = {
            "annotation": None,
            "dynamic_type": "at::TensorOptions",
            "is_nullable": False,
            "name": cpp_a.name,
            "type": cpp_a.type,
            "kwarg_only": True,
        }
        if cpp_a.default is not None:
            arg["default"] = cpp_a.default
        return arg
    elif isinstance(cpp_a.argument, SelfArgument):
        raise AssertionError()
    elif isinstance(cpp_a.argument, Argument):
        return compute_argument_yaml(
            cpp_a.argument,
            schema_order=schema_order,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )


def compute_argument_yaml(
    a: Argument,
    *,
    schema_order: bool,
    kwarg_only_set: Set[str],
    out_arg_set: Set[str],
    name_to_field_name: Dict[str, str],
) -> object:
    arg: Dict[str, object] = {
        "annotation": str(a.annotation) if a.annotation else None,
        "dynamic_type": dynamic_type(a.type),
        "is_nullable": a.type.is_nullable(),
        "name": a.name,
        # legacy, report ints
        "type": cpp.argument_type(a, binds="__placeholder__", symint=False).cpp_type(),
    }
    if a.default is not None:
        arg["default"] = pythonify_default(cpp.default_expr(a.default, a.type, symint=False))
    if a.name in kwarg_only_set:
        arg["kwarg_only"] = True
    if a.name in out_arg_set:
        arg["output"] = True
        arg["allocate"] = True
        # See Note [name and field_name]
        if a.name in name_to_field_name:
            arg["field_name"] = name_to_field_name[a.name]
    # Historically, booleans don't get their size recorded, because it
    # is already built into the cpp type (e.g., std::array<bool, 4>)
    l = a.type.is_list_like()
    if l is not None and l.size is not None and str(l.elem) != "bool":
        arg["size"] = l.size
    return arg


@with_native_function
def compute_declaration_yaml(f: NativeFunction) -> object:
    returns, name_to_field_name = compute_returns_yaml(f)

    # These sets are used to conveniently test if an argument is a
    # kwarg-only or out argument
    kwarg_only_set = {a.name for a in f.func.arguments.flat_kwarg_only}
    out_arg_set = {a.name for a in f.func.arguments.out}

    sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    cpp_args = sig_group.signature.arguments()
    arguments = [
        compute_cpp_argument_yaml(
            cpp_a,
            schema_order=False,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for cpp_a in cpp_args
    ]

    schema_order_jit_arguments = list(f.func.schema_order_arguments())

    schema_order_arguments = [
        compute_argument_yaml(
            a,
            schema_order=True,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for a in schema_order_jit_arguments
    ]

    cpp_schema_order_types = [
        # NB: method here doesn't matter
        r.type
        for a in schema_order_jit_arguments
        for r in cpp.argument(
            a,
            method=False,
            cpp_no_default_args=set(),
            faithful=False,
            symint=False,
            has_tensor_options=False,
        )
    ]

    # legacy, report ints
    cpp_returns = cpp.returns_type(f.func.returns, symint=False).cpp_type()
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"

    is_factory_method = (
        any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args) and Variant.method not in f.variants
    )

    return OrderedDict(
        [
            ("name", cpp.name(f.func)),
            ("operator_name", str(f.func.name.name)),
            ("overload_name", str(f.func.name.overload_name)),
            ("manual_kernel_registration", f.manual_kernel_registration),
            (
                "category_override",
                f.category_override if f.category_override is not None else "",
            ),
            ("schema_string", f"aten::{f.func}"),
            ("arguments", arguments),
            ("schema_order_cpp_signature", schema_order_cpp_signature),
            ("schema_order_arguments", schema_order_arguments),
            ("method_of", compute_method_of_yaml(f.variants)),
            ("mode", "native"),
            ("python_module", "" if f.python_module is None else f.python_module),
            ("returns", returns),
            ("inplace", f.func.name.name.inplace),
            ("is_factory_method", is_factory_method),
            ("abstract", f.is_abstract),
            ("device_guard", f.device_guard),
            ("with_gil", False),
            ("deprecated", False),
            ("has_math_kernel", f.has_composite_implicit_autograd_kernel),
        ]
    )


# See Note [Auto generated composite kernels]
def has_autogenerated_composite_kernel(f: NativeFunction) -> bool:
    return (f.structured or f.structured_delegate is not None) and (
        f.func.kind() == SchemaKind.functional or f.func.kind() == SchemaKind.inplace
    )


@with_native_function_and_indices
def compute_registration_declarations(f: NativeFunction, backend_indices: Dict[DispatchKey, BackendIndex]) -> str:
    name = dispatcher.name(f.func)
    returns_type = dispatcher.returns_type(f.func.returns).cpp_type_registration_declarations()
    args = dispatcher.arguments(f.func)
    args_str = ", ".join(a.no_default().decl_registration_declarations() for a in args)
    comment_data: Dict[str, str] = {
        "schema": f"aten::{f.func}",
        # TODO: What exactly is the semantics of the 'dispatch' field?
        "dispatch": str(
            {k for k, v in backend_indices.items() if v.has_kernel(f)} != {DispatchKey.CompositeImplicitAutograd}
            and {k for k, v in backend_indices.items() if v.has_kernel(f)}
            != {
                DispatchKey.CompositeImplicitAutograd,
                DispatchKey.CompositeImplicitAutogradNestedTensor,
            }
        ),
        "default": str(f.has_composite_kernel or has_autogenerated_composite_kernel(f)),
    }
    return f"""{returns_type} {name}({args_str}); // {json.dumps(comment_data)}
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def get_custom_build_selector(
    provided_op_registration_allowlist: Optional[List[str]],
    op_selection_yaml_path: Optional[str],
) -> SelectiveBuilder:
    assert not (provided_op_registration_allowlist is not None and op_selection_yaml_path is not None), (
        "Both provided_op_registration_allowlist and "
        + "op_selection_yaml_path can NOT be provided at the "
        + "same time."
    )

    op_registration_allowlist: Optional[Set[str]] = None
    if provided_op_registration_allowlist is not None:
        op_registration_allowlist = set(provided_op_registration_allowlist)

    if op_registration_allowlist is not None:
        selector = SelectiveBuilder.from_legacy_op_registration_allow_list(
            op_registration_allowlist,
            True,
            False,
        )
    elif op_selection_yaml_path is not None:
        selector = SelectiveBuilder.from_yaml_path(op_selection_yaml_path)
    else:
        selector = SelectiveBuilder.get_nop_selector()

    return selector


def get_grouped_by_view_native_functions(
    native_functions: Sequence[NativeFunction],
) -> Sequence[Union[NativeFunction, NativeFunctionsViewGroup]]:
    def maybe_create_view_group(
        d: Dict[Union[ViewSchemaKind, SchemaKind], NativeFunction],
    ) -> List[Union[NativeFunction, NativeFunctionsViewGroup]]:
        funcs: List[Union[NativeFunction, NativeFunctionsViewGroup]] = []
        if ViewSchemaKind.aliasing in d:
            view = d.pop(ViewSchemaKind.aliasing)
            view_inplace = d.pop(ViewSchemaKind.aliasing_inplace, None)
            view_copy = d.pop(SchemaKind.functional, None)

            funcs.append(
                NativeFunctionsViewGroup(
                    view=view,
                    view_copy=view_copy,
                    view_inplace=view_inplace,
                )
            )
        # Take the remaining functions that weren't part of the view group
        # and emit them separately
        for func in d.values():
            funcs.append(func)
        return funcs

    grouped_by_views: Dict[FunctionSchema, Dict[Union[SchemaKind, ViewSchemaKind], NativeFunction]] = defaultdict(dict)
    for f in native_functions:
        schema = f.func.view_signature()
        view_kind: ViewSchemaKind = f.view_schema_kind
        # We need to group up ops relevant to the same "view", consisting of:
        # view op (ViewSchemaKind.aliasing)
        # view_inplace op (ViewSchemaKind.aliasing_inplace)
        # view_copy op (SchemaKind.functional)
        if view_kind == ViewSchemaKind.non_aliasing:
            kind = f.func.kind()
            assert kind not in grouped_by_views[schema]
            grouped_by_views[schema][kind] = f
        else:
            assert view_kind not in grouped_by_views[schema]
            grouped_by_views[schema][view_kind] = f

    return list(concatMap(maybe_create_view_group, grouped_by_views.values()))


def get_grouped_native_functions(
    native_functions: Sequence[NativeFunction],
) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
    def flatten_pre_group(d: Dict[SchemaKind, NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            # Invariant: any NativeFunctions that are code-generated
            # should have been grouped into NativeFunctionsGroup objects
            assert not any("generated" in f.tags for f in d.values())
            return list(d.values())
        else:
            return [r]

    # TODO: how come ValuesView isn't a Sequence lol
    pre_grouped_native_functions = pre_group_native_functions(native_functions)
    return list(concatMap(flatten_pre_group, list(pre_grouped_native_functions.values())))


def get_ns_grouped_kernels(
    *,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_indices: Dict[DispatchKey, BackendIndex],
    native_function_decl_gen: Callable[
        [Union[NativeFunctionsGroup, NativeFunction], BackendIndex], List[str]
    ] = dest.compute_native_function_declaration,
) -> Dict[str, List[str]]:
    ns_grouped_kernels: Dict[str, List[str]] = defaultdict(list)
    for f in grouped_native_functions:
        native_function_namespaces = set()
        dispatch_keys = set()
        for dispatch_key, backend_idx in backend_indices.items():
            backend_metadata = backend_idx.get_kernel(f)
            if backend_metadata:
                namespace = backend_metadata.cpp_namespace
                dispatch_keys.add(dispatch_key)
                native_function_namespaces.add(namespace)
            else:
                namespace = DEFAULT_KERNEL_NAMESPACE
            assert (
                len(native_function_namespaces) <= 1
            ), f"Codegen only supports one namespace per operator, got {native_function_namespaces} from {dispatch_keys}"
            ns_grouped_kernels[namespace].extend(native_function_decl_gen(f, backend_idx))

        ns_grouped_kernels["at::native"].extend(dest.compute_tonative_function_declaration(f, backend_idx))
    return ns_grouped_kernels


# Return native function declarations grouped by their namespaces.
def get_native_function_declarations(
    *,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_indices: Dict[DispatchKey, BackendIndex],
    native_function_decl_gen: Callable[
        [Union[NativeFunctionsGroup, NativeFunction], BackendIndex], List[str]
    ] = dest.compute_native_function_declaration,
) -> Tuple[List[str], List[str]]:
    """
    Generate kernel declarations, in `NativeFunction(s).h`.
    :param grouped_native_functions: a sequence of `NativeFunction` or `NativeFunctionGroup`.
    :param backend_indices: kernel collections grouped by dispatch key.
    :param native_function_decl_gen: callable to generate kernel declaration for each `NativeFunction`.
    :return: a list of string, from the string with all declarations, grouped by namespaces, split by newline.
    """

    ns_grouped_kernels = get_ns_grouped_kernels(
        grouped_native_functions=grouped_native_functions,
        backend_indices=backend_indices,
        native_function_decl_gen=native_function_decl_gen,
    )
    kernels: List[str] = ns_grouped_kernels[supa_utils.CustomYamlProcessor.cpp_namespace]
    ordered_kernels = list(OrderedDict.fromkeys(kernels))

    native_kernels: List[str] = ns_grouped_kernels.get("at::native", [])
    ordered_at_kernels = list(OrderedDict.fromkeys(native_kernels))
    return ordered_kernels, ordered_at_kernels


def get_kernel_namespace(*, f: Union[NativeFunction, NativeFunctionsGroup], backend_idx: BackendIndex) -> str:
    backend_metadata = backend_idx.get_kernel(f)
    # assert not backend_metadata or "::native" in backend_metadata.cpp_namespace, (
    #     f"The kernel for function {f.func.name if isinstance(f, NativeFunction) else f.functional.func.name} "
    #     f"with dispatch key {backend_idx.dispatch_key}"
    #     f" has a namespace {backend_metadata.cpp_namespace} and it's not ending with '::native'."
    # )
    return backend_metadata.cpp_namespace if backend_metadata else DEFAULT_KERNEL_NAMESPACE


# Return native function definitions grouped by dispatch key and custom namespace.
# Used in RegisterDispatchKey.cpp and etc.
def get_native_function_definitions(
    *,
    fm: FileManager,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    dispatch_key: DispatchKey,
    backend_idx: BackendIndex,
    selector: SelectiveBuilder,
    rocm: bool = False,
    symint: bool,
    skip_dispatcher_op_registration: bool,
    gen_dispatch_helpers: bool,
) -> List[str]:
    definitions: List[str] = []
    ns_definitions: Dict[str, List[str]] = defaultdict(list)
    anonymous_definitions: Dict[str, List[str]] = defaultdict(list)
    registrations: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    newline = "\n"
    class_name = supa_utils.CustomYamlProcessor.class_name
    ns_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.NAMESPACED_DEFINITION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=class_name,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    anonymous_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.ANONYMOUS_DEFINITION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=class_name,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    reg_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.REGISTRATION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=class_name,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    for f in grouped_native_functions:
        kernel_namespace = get_kernel_namespace(f=f, backend_idx=backend_idx)

        ns_definitions[kernel_namespace].extend(
            ns_gen(f),
        )
        anonymous_definitions[kernel_namespace].extend(
            anonymous_gen(f),
        )
        namespace = f.namespace if isinstance(f, NativeFunction) else f.functional.namespace
        if namespace not in registrations[kernel_namespace]:
            registrations[kernel_namespace] = defaultdict(list)
        registrations[kernel_namespace][namespace].extend(
            reg_gen(f),
        )

    for kernel_namespace in ns_definitions:
        if len(ns_definitions[kernel_namespace]) == 0:
            continue
        ns_helper = NamespaceHelper(namespace_str=kernel_namespace, max_level=3)
        registration_body = ""
        for namespace in registrations[kernel_namespace]:
            if not registrations[kernel_namespace][namespace]:
                continue
            registration_body += f"""
TORCH_LIBRARY_IMPL({namespace}, {dispatch_key}, m) {{
    {newline.join(registrations[kernel_namespace][namespace])}
}};"""
        definitions.extend(
            fm.substitute_with_template(
                "RegisterDispatchDefinitions.ini",
                lambda: {
                    "ns_prologue": ns_helper.prologue,
                    "ns_epilogue": ns_helper.epilogue,
                    "dispatch_helpers": dest.gen_registration_helpers(backend_idx) if gen_dispatch_helpers else [],
                    "dispatch_anonymous_definitions": anonymous_definitions[kernel_namespace],
                    "static_init_dispatch_registrations": "" if skip_dispatcher_op_registration else registration_body,
                    "deferred_dispatch_registrations": "",
                    "dispatch_namespace": "cuda",
                    "dispatch_namespaced_definitions": ns_definitions[kernel_namespace],  # generate at::cuda::ops used by torch_supa
                },
            ).split(newline)
        )

    return definitions


# Return native function declarations grouped by dispatch key and custom namespace.
# Used in CPUFunctions_inl.h and etc.
def get_namespaced_declaration(
    *,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    dispatch_key: DispatchKey,
    backend_idx: BackendIndex,
    selector: SelectiveBuilder,
    rocm: bool,
    symint: bool,
) -> List[str]:
    declarations: List[str] = []
    ns_grouped_kernels: Dict[str, List[str]] = defaultdict(list)
    newline = "\n"
    func = dest.RegisterDispatchKey(
        backend_idx,
        Target.NAMESPACED_DECLARATION,
        selector,
        rocm=rocm,
        class_method_name=None,
        skip_dispatcher_op_registration=False,
        symint=symint,
    )
    for f in grouped_native_functions:
        namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace("native", dispatch_key.lower())

        ns_grouped_kernels[namespace].extend(
            func(f),
        )

    for namespace, kernels in ns_grouped_kernels.items():
        if len(kernels) == 0:
            continue
        ns_helper = NamespaceHelper(namespace_str=namespace, entity_name="", max_level=3)
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        declarations.extend(
            f"""
{ns_helper.prologue}
{newline.join(ordered_kernels)}
{ns_helper.epilogue}
        """.split(
                newline
            )
        )
    return declarations


# Return native function schema registration code for aten and other namespaces.
def get_native_function_schema_registrations(
    *,
    native_functions: Sequence[NativeFunction],
    schema_selector: SelectiveBuilder,
) -> Tuple[List[str], str]:
    ns_native_functions: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_native_functions[native_function.namespace].append(native_function)
    schema_registrations = ""
    aten_schema_registrations = []
    custom_namespace = None
    for namespace, funcs in ns_native_functions.items():
        schema_registrations_body = list(mapMaybe(RegisterSchema(schema_selector), funcs))
        # NB: we have to separate aten namespace registration from other namespaces,
        # because in the template we hardcoded an operator for ATen already.
        if namespace == "aten":
            aten_schema_registrations = schema_registrations_body
        else:
            custom_namespace = namespace
            tab = "\t"
            # if the namespace is predefined, we should use define a library fragment
            # instead of a new library
            torch_library_macro = "TORCH_LIBRARY_FRAGMENT" if namespace in FRAGMENT_NAMESPACES else "TORCH_LIBRARY"
            schema_registrations += f"""
{torch_library_macro}({custom_namespace}, m) {{
  {tab.join(schema_registrations_body)}
}};"""
    return (aten_schema_registrations, schema_registrations)


def gen_aggregated_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    static_dispatch_idx: List[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    cpu_fm: FileManager,
    core_fm: FileManager,
    cuda_fm: FileManager,
    functions_keys: Set[DispatchKey],
    dispatch_keys: Sequence[DispatchKey],
    rocm: bool,
) -> None:

    # generate structured impl declarations.
    content = []
    for dispatchkey in backend_indices.values():
        for g in structured_native_functions:
            if g.structured:
                declares = dest.gen_structured(g, dispatchkey)
                content.extend(declares)

    with SwitchedTemplateFolder(core_fm, local_templates_folder) as core_fm:
        core_fm.write_with_template(
            "SUPAStructuredFunctions.h",
            "NativeMetaFunctions.h",
            lambda: {
                "NativeMetaFunctions_includes": [],
                "NativeMetaFunctions_declarations": content,
            },
        )

    declarations, at_declarations = get_native_function_declarations(
        grouped_native_functions=grouped_native_functions,
        backend_indices=backend_indices,
    )
    ns_helper = NamespaceHelper("at::supa")
    core_fm.write_with_template(
        "SUPANativeFunctions.h",
        "DispatchKeyNativeFunctions.h",
        lambda: {
            "generated_comment": "Autogenerated file by gen.py. Do not edit directly!",
            "namespace_prologue": ns_helper.prologue,
            "class_name": supa_utils.CustomYamlProcessor.class_name,
            "namespace_epilogue": ns_helper.epilogue,
            "dispatch_declarations": declarations,
        },
    )
    at_declarations.insert(0, "namespace at::native {")
    at_declarations.append("}")
    core_fm.write(
        "NativeFunctions.h",
        lambda: {
            "NativeFunctions_includes": [],
            "NativeFunctions_declarations": at_declarations,
        },
    )


def gen_headers(
    *,
    native_functions: Sequence[NativeFunction],
    valid_tags: Set[str],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    static_dispatch_idx: List[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    core_fm: FileManager,
    cpu_fm: FileManager,
    cuda_fm: FileManager = None,
    ops_fm: FileManager = None,
    dispatch_keys: Sequence[DispatchKey],
    functions_keys: Set[DispatchKey] = None,
    per_operator_headers: bool = False,
    rocm: bool = False,
) -> None:
    assert not per_operator_headers
    gen_aggregated_headers(
        native_functions=native_functions,
        grouped_native_functions=grouped_native_functions,
        structured_native_functions=structured_native_functions,
        static_dispatch_idx=static_dispatch_idx,
        selector=selector,
        backend_indices=backend_indices,
        cpu_fm=cpu_fm,
        core_fm=core_fm,
        cuda_fm=cuda_fm,
        dispatch_keys=dispatch_keys,
        functions_keys=functions_keys,
        rocm=rocm,
    )


def gen_source_files(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    view_groups: Sequence[NativeFunctionsViewGroup],
    selector: SelectiveBuilder,
    static_dispatch_idx: List[BackendIndex],
    backend_indices: Dict[DispatchKey, BackendIndex],
    core_fm: FileManager,
    cpu_fm: FileManager,
    dispatch_keys: Sequence[DispatchKey],
    skip_dispatcher_op_registration: bool = False,
) -> None:

    for dispatch_key in dispatch_keys:
        fm = cpu_fm

        def operator_headers() -> List[str]:
            headers = [
                '#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"',
                '#include "torch_supa/csrc/aten/VariableFallbackKernel.h"',
                '#include "torch_supa/csrc/aten/core/SUPAStructuredFunctions.h"',
                '#include "torch_supa/csrc/aten/core/NativeFunctions.h"',
                '#include "torch_supa/csrc/profiler/supa_profiler.h"',
                '#include "torch_supa/csrc/core/supa/SUPAGuard.h"',
                '#include "torch_supa/csrc/aten/common/native.h"',
                '#include "torch_supa/csrc/utils/EnvConfig.h"',
            ]
            return headers

        backend_index = backend_indices[dispatch_key]
        ns_grouped_native_functions = defaultdict(list)
        for grouped_native_function in grouped_native_functions:
            namespace = (
                grouped_native_function.namespace
                if isinstance(grouped_native_function, NativeFunction)
                else grouped_native_function.functional.namespace
            )
            ns_grouped_native_functions[namespace].append(grouped_native_function)

        gen_dispatch_helpers: bool = False

        dispatch_definitions = get_native_function_definitions(
            fm=fm,
            grouped_native_functions=grouped_native_functions,
            dispatch_key=dispatch_key,
            backend_idx=backend_index,
            selector=selector,
            symint=True,
            skip_dispatcher_op_registration=skip_dispatcher_op_registration,
            gen_dispatch_helpers=gen_dispatch_helpers,
        )

        dispatch_namespace = supa_utils.NameHelper.source_file_name(dispatch_key)
        fm.write_with_template(
            f"Register{dispatch_namespace}.cpp",
            "RegisterDispatchKey.cpp",
            lambda: {
                "extra_cuda_headers": '#include "torch_supa/csrc/utils/EnvConfig.h"',
                "external_backend_headers": "",
                "dispatch_headers": dest.gen_registration_headers(backend_index, False, False),
                "ops_headers": operator_headers(),
                "dispatch_helpers": "",
                "dispatch_definitions": dispatch_definitions,
            },
        )


def gen_foreach_register(
    *,
    cpu_fm: FileManager,
    native_functions: Sequence[NativeFunction],
    backend_indices: Dict[DispatchKey, BackendIndex],
    cpu_backend_indices: Dict[DispatchKey, BackendIndex],
) -> None:
    header_set = set()
    foreach_dict: Dict[str, str] = {}

    for f in native_functions:
        schema = f.func.name
        if str(schema).startswith("_foreach"):
            header_set.add(f.func.name.name.base)
            if schema in cpu_backend_indices.index and schema not in backend_indices[DispatchKey.PrivateUse1].index:
                foreach_dict[str(schema)] = cpu_backend_indices.index[schema].kernel

    header_template = CodeTemplate("#include <ATen/ops/${function}_native.h>")
    kernel_template = CodeTemplate('m.impl("${schema}", TORCH_FN(at::native::${kernel}));')

    cpu_fm.write(
        "ForeachRegister.cpp",
        lambda: {
            "include_headers": [header_template.substitute(function=h) for h in sorted(header_set)],
            "foreach_kernel": [kernel_template.substitute(schema=k, kernel=v) for k, v in foreach_dict.items()],
        },
    )


def get_torchgen_root() -> pathlib.Path:
    """get path of torchgen in system, for getting tags.yaml and native_functions.yaml"""
    import torchgen

    return pathlib.Path(torchgen.__file__).parent.resolve()


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Generate ATen source files")

    parser.add_argument("--torch-version", type=str, required=True, help="target version for pytorch")

    parser.add_argument(
        "--custom-function-yaml",
        type=str,
        required=True,
        help="path to source yaml file containing operator external definitions",
    )

    parser.add_argument(
        "--impl-path", type=str, required=True, help="path to the source C++ file containing kernel definitions"
    )

    parser.add_argument("--custom-ops-dir", type=str, help="folder to custom_ops.py")

    parser.add_argument("--autograd-dir", type=str, help="folder for 'derivatives.yaml'")

    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        required=True,
        help="output directory. should be 'torch_supa/csrc/aten/'",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    options = parser.parse_args(args)

    selector = get_custom_build_selector(None, None)

    torchgen_root = get_torchgen_root()
    native_yaml_path = os.path.join(torchgen_root, "packaged", "ATen", "native", "native_functions.yaml")
    tags_yaml_path = os.path.join(torchgen_root, "packaged", "ATen", "native", "tags.yaml")
    template_path = os.path.join(torchgen_root, "packaged", "ATen")

    from torchgen.model import dispatch_keys

    # TODO: stop generating CUDA kernels for non-CUDA builds
    ignore_keys = set()
    if True:  # not options.mps:
        ignore_keys.add(DispatchKey.MPS)
        if DispatchKey.MPS in dispatch_keys:
            del dispatch_keys[dispatch_keys.index(DispatchKey.MPS)]

    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path, ignore_keys)
    valid_tags = _GLOBAL_PARSE_TAGS_YAML_CACHE[tags_yaml_path]
    native_functions, backend_indices = (
        parsed_yaml.native_functions.copy(),
        parsed_yaml.backend_indices,
    )
    cpu_backend_indices = backend_indices[DispatchKey.CPU]

    supa_processor: supa_utils.CustomYamlProcessor = supa_utils.CustomYamlProcessor(options.custom_function_yaml)
    # add supa kernel and native metadata to native_functions
    supa_processor.update_backend_index(native_functions, backend_indices)
    supa_processor.add_custom_op(native_functions, backend_indices, valid_tags)

    implements_paths = [os.path.join(options.impl_path, "ops"), os.path.join(options.impl_path, "common")]

    implements = supa_utils.CppImplements(implements_paths)
    implements.search_implements()

    grouped_native_functions = get_grouped_native_functions(native_functions)

    structured_native_functions = [g for g in grouped_native_functions if isinstance(g, NativeFunctionsGroup)]

    implements.check_group(grouped_native_functions, structured_native_functions, backend_indices)

    implements.checking_implements(
        grouped_native_functions,
        structured_native_functions,
        backend_indices,
        supa_processor.supported_functions,
        supa_processor.custom_functions,
    )

    for g in structured_native_functions:
        assert (g.functional is not None) and (g.out is not None)

    # NB: It is mandatory to NOT use os.path.join here, as the install directory
    # will eventually be ingested by cmake, which does not respect Windows style
    # path slashes.  If you switch this to use os.path.join, you'll get an error
    # like:
    #
    #   Syntax error in cmake code when parsing string
    #
    #     C:/Jenkins/workspace/pytorch-builds/pytorch-win-ws2016-cuda9-cudnn7-py3-build/build/aten/src/ATen\core/TensorMethods.h
    #
    #   Invalid character escape '\c'.
    core_install_dir = f"{options.install_dir}/core"
    pathlib.Path(core_install_dir).mkdir(parents=True, exist_ok=True)

    options.source_path = template_path
    core_fm = make_file_manager(options=options, install_dir=core_install_dir)
    cpu_fm = make_file_manager(options=options)

    dispatch_keys = supa_utils.CustomYamlProcessor.supported_dispatch_keys

    static_dispatch_idx: List[BackendIndex] = []

    for f in supa_processor.custom_functions:
        if not f.root_name.startswith("_") and "fused_torch_api" not in f.tags:
            raise AssertionError(
                "custom functions with public names must be explicitly tagged as fused_torch_api"
            )

    gen_source_files(
        native_functions=native_functions,
        grouped_native_functions=grouped_native_functions,
        structured_native_functions=structured_native_functions,
        view_groups=[],
        selector=selector,
        static_dispatch_idx=static_dispatch_idx,
        backend_indices=backend_indices,
        core_fm=core_fm,
        cpu_fm=cpu_fm,
        dispatch_keys=dispatch_keys,
    )
    with SwitchedTemplateFolder(cpu_fm, local_templates_folder) as cpu_fm:
        gen_foreach_register(
            cpu_fm=cpu_fm,
            native_functions=native_functions,
            backend_indices=backend_indices,
            cpu_backend_indices=cpu_backend_indices,
        )

    gen_headers(
        native_functions=native_functions,
        valid_tags=valid_tags,
        grouped_native_functions=grouped_native_functions,
        structured_native_functions=structured_native_functions,
        static_dispatch_idx=static_dispatch_idx,
        selector=selector,
        backend_indices=backend_indices,
        core_fm=core_fm,
        cpu_fm=cpu_fm,
        dispatch_keys=dispatch_keys,
    )

    if len(supa_processor.custom_functions) > 0:
        with SwitchedTemplateFolder(cpu_fm, local_templates_folder) as cpu_fm:
            gen_custom_trace(cpu_fm, supa_processor.custom_functions)
            gen_custom_functions_dispatch(cpu_fm, supa_processor.custom_functions)

    if options.custom_ops_dir:
        py_ops_fm = FileManager(install_dir=options.custom_ops_dir, template_dir=local_templates_folder, dry_run=False)
        gen_custom_ops_patch(py_ops_fm, supa_processor.custom_functions)
    else:
        print("WARNING: missing argument: --custom-ops-dir")


if __name__ == "__main__":
    main()
