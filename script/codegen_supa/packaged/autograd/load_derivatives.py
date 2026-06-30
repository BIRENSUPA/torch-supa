# Parses derivatives.yaml into autograd functions
#
# Each autograd function is represented by `DifferentiabilityInfo` containing
# a list of `Derivative`. See `torchgen.api.autograd` for the data models.
import inspect
import yaml
from collections import defaultdict
from typing import Counter, Dict, List, Set

from torchgen.api.autograd import (
    DifferentiabilityInfo,
    SavedAttribute,
)

from torchgen.api import cpp

from torchgen.api.types import (
    NamedCType,
)

from torchgen.model import (
    FunctionSchema,
    NativeFunction,
    parse_returns,
)

from torchgen.yaml_utils import YamlLoader
from torchgen.packaged.autograd import load_derivatives


def patch_saved_variables():
    func_obj = load_derivatives.saved_variables
    sources = inspect.getsource(func_obj)
    sources = sources.replace(
        'if ".sizes()" in formula or "->sizes()" in formula:',
        r'''from torchgen.api.types import intArrayRefT
    REPLACEMENTS.append((r"{}.sizes\(\)", { "suffix": "_sizes", "nctype": lambda name: NamedCType(name, BaseCType(intArrayRefT))}))
    if False:''',
        1,
    )
    codes = compile(sources, inspect.getfile(func_obj), "exec")
    exec(codes, load_derivatives.__dict__)


def prepare_hidden_output(outputs: str):
    return [
        SavedAttribute(nctype=NamedCType(r.name, cpp.return_type(r, symint=True).remove_const_ref()), expr=None)
        for r in parse_returns(outputs)
    ]


def load_custom_derivatives(
    derivatives_yaml_path: str, custom_functions: List[NativeFunction], native_funcs: List[NativeFunction]
) -> Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]]:

    functions_by_signature: Dict[FunctionSchema, List[NativeFunction]] = defaultdict(list)
    functions_by_schema: Dict[str, NativeFunction] = {}

    # Keep track of how many of which ops we've seen so we can
    # disambiguate them with a numeric suffix.
    op_counter = Counter[str]()

    patch_saved_variables()

    for f in custom_functions:
        functions_by_signature[f.func.signature()].append(f)
        functions_by_schema[str(f.func)] = f

    with open(derivatives_yaml_path) as f:
        definitions = yaml.load(f, Loader=YamlLoader)
    assert isinstance(definitions, list), "bad format of derivatives.yaml"

    # infos is a dict that maps FunctionSchema -> a dict of per dispatch key DifferentiabilityInfos
    # this is useful because in tools/autograd/gen_autograd.py:match_differentiability_info
    # we ultimately need to categorize the DifferentiabilityInfos by FunctionSchema
    infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]] = {}

    used_dispatch_keys: Set[str] = set()
    for defn_dict in definitions:
        # Ensure that the old derivatives.yaml schema with no dispatch key can be loaded.
        if "dispatch" not in defn_dict:
            specification = defn_dict.pop("name")
            output_differentiability = defn_dict.pop("output_differentiability", None)
            hidden_outputs = defn_dict.pop("hidden_outputs", None)
            hidden_outputs = prepare_hidden_output(hidden_outputs) if hidden_outputs else None

            tags = defn_dict.pop("tags", [])
            if "native_fwd" in tags:
                # fwd function is defined in native_functions.yaml instead of custom.
                f: NativeFunction = next(filter(lambda func: str(func.func) == specification, native_funcs))
                functions_by_signature[f.func.signature()].append(f)
                functions_by_schema[str(f.func)] = f
                f.tags.add("native_fwd")
            defn_dict = {"name": specification, "dispatch": {"Default": defn_dict}}
            if output_differentiability:
                defn_dict["output_differentiability"] = output_differentiability

        name, per_dispatch_diffinfos = load_derivatives.create_differentiability_info(
            defn_dict,
            functions_by_signature,
            functions_by_schema,
            op_counter,
            used_dispatch_keys,
        )

        if hidden_outputs:
            outputs = per_dispatch_diffinfos["Default"].all_saved_outputs
            object.__setattr__(per_dispatch_diffinfos["Default"], "all_saved_outputs", outputs + hidden_outputs)

        infos[name] = per_dispatch_diffinfos

    return infos
