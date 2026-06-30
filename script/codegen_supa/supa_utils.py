"""
Copyright © 2024 Shanghai Biren Technology Co., Ltd. All rights reserved.
"""

import re
import os
import yaml
import itertools
from typing import Tuple, List, Dict, Sequence, Union, Set, Optional
from collections import Counter, defaultdict

# from torchgen import model
# model.DEFAULT_KERNEL_NAMESPACE = "at::supa::SUPANativeFunctions"  # noqa: E402, must put it before any other reference

from torchgen.model import (
    NativeFunction,
    NativeFunctionsGroup,
    Location,
    DispatchKey,
    BackendIndex,
    OperatorName,
    BackendMetadata,
    dispatch_keys,
)
from torchgen.gen import LineLoader
from torchgen.utils import FileManager
from torchgen.api import meta, cpp
from torchgen.api.types.signatures import DispatcherSignature
from torchgen.context import native_function_manager


class TorchVersions(object):
    current: int = 0
    versions = {
        f"TORCH_{v1}_{v2}_{v3}": int(v1) * 10000 + int(v2) * 100 + int(v3)
        for v1, v2, v3 in map(
            lambda x: x.split("."),
            [
                "1.10.0",
                "1.10.1",
                "1.10.2",
                "1.11.0",
                "1.12.0",
                "1.12.1",
                "1.13.0",
                "1.13.1",
                "2.0.0",
                "2.1.0",
                "2.2.0",
                "2.3.0",
                "2.4.1",
                "2.5.0",
                "2.5.1",
                "2.6.0",
                "2.7.0",
                "2.8.0",
                "2.9.0",
                "2.10.0",
                "2.11.0",
                "2.12.0"
            ],
        )
    }

    @staticmethod
    def from_str(ver: str) -> int:
        v1, v2, v3 = ver.split("+")[0].split(".")
        return int(v1) * 10000 + int(v2) * 100 + int(v3)

    @classmethod
    def init(cls, ver: str):
        cls.current = cls.from_str(ver)

    @classmethod
    def generate_ver_file(cls, output_dir: str):
        ver_lines = ["#pragma once", "// this file can be removed if it runs only in specified pytorch"]
        ver_lines.extend([f"#define {k} {v}" for k, v in sorted(cls.versions.items())])
        ver_lines.append(f"#define TORCH_VER {cls.current}")
        fm_native_functions: FileManager = FileManager(install_dir=output_dir, template_dir="", dry_run=False)
        fm_native_functions._write_if_changed(output_dir, "\n".join(ver_lines))

    @classmethod
    def get(cls, ver: str) -> int:
        if ver == "TORCH_VER":
            return cls.current
        return cls.versions.get(ver, 0)


BackendIndices = Dict[DispatchKey, BackendIndex]

enable_native: bool = False


class CustomYamlProcessor:
    """process contents in supa_native_functions.yaml.

    1. create a BackendMetadata for OP name defined in supa_native_functions.yaml and link it under DispatchKey.[autograd]PrivateUse1
    2. after introduce native ops, an op may has any of (or both of) Native and SUPA impl.
    3. in torchgen, the existence of BackendMetadata is entry condition of generating code. but need to judge what impl is generate according additional flag of NativeFunction:
       `supa_kernel`: it has SUPA implement. such as "SUPANativeFunctions::xxxx"
       `is_custom_supa`: it is supa custom OPs.
    4. other flags for structured native function group:
       `has_supa_structs`: structured function group implemented by SUPA
       `force_structured`: special flag for structured function group who has no SUPA nor native implementation. original `structured` flag is cleared
       but need to output function name as style of structued in order to keep consistence of legacy code.

    """

    dispatch_key = DispatchKey.PrivateUse1
    autograd_dispatch_key = DispatchKey.AutogradPrivateUse1
    supported_dispatch_keys = (dispatch_key, autograd_dispatch_key)
    cpp_namespace = "at"
    class_name = "SUPANativeFunctions"

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path) as fi:
            self.content = yaml.load(fi, Loader=LineLoader)

        assert "supported" in self.content
        assert "perf_supported" in self.content
        assert "autograd" in self.content
        assert "custom" in self.content
        assert "perf_autograd" in self.content

        self.custom_functions: List[NativeFunction] = []
        """all customer defined native functions.
        """

        self.symint_support_lists = self.content.get("symint", [])

        # add supported dispatch key for NativeFunction.from_yaml()
        dispatch_keys.extend(CustomYamlProcessor.supported_dispatch_keys)

        self.supported_functions: List[NativeFunction] = []

    def update_backend_index(self, functions: List[NativeFunction], backend_indics: BackendIndices) -> None:
        """add custom dispatch keys into backend_indices.

        Args:
            functions (List[NativeFunction]): Existing NativeFunction list
            backend_indics (BackendIndices): BackendIndex list
        """

        def get_native_function(op_name: str) -> NativeFunction:
            for f in functions:
                if str(f.func.name) == op_name:
                    return f
            print(f"{op_name} is not found in current native yaml. skip it.")
            return None

        kernels: Dict[OperatorName, BackendMetadata] = {}
        autograd_kernels: Dict[OperatorName, BackendMetadata] = {}
        for k_dict, content, perf, is_autograd in (
            (kernels, self.content["supported"], False, False),
            (autograd_kernels, self.content["autograd"], False, True),
            (kernels, self.content["perf_supported"], True, False),
            (autograd_kernels, self.content["perf_autograd"], True, True),
        ):
            if content is None:
                continue
            for item in content:
                # Check: forbid .out version in autograd section
                if is_autograd and item.endswith(".out"):
                    raise ValueError(
                        f"ERROR: '.out' version '{item}' is not allowed in autograd section. "
                        f"Please put '.out' version in 'supported' section, and other versions (functional/inplace) in 'autograd' section."
                    )
                f: NativeFunction = get_native_function(item)
                if f:
                    support_symint = item in self.symint_support_lists
                    object.__setattr__(f, "supa_kernel", True)
                    if f in self.supported_functions:
                        continue
                    # generally it should use 'f.func.has_symint()' to determine whether to use symint overload.
                    # but so far, in our implement, only operator in 'symint' section supports symint.
                    # and has to mark with a special flag to replace has_symint() function.
                    object.__setattr__(f, "has_symint_implement", support_symint)
                    object.__setattr__(f, "is_custom_supa", False)
                    object.__setattr__(f, "native_metadata", backend_indics[DispatchKey.CUDA].get_kernel(f))
                    # print(id(f), item)
                    if perf:
                        object.__setattr__(f, "perf", perf)
                        # print(backend_indics[DispatchKey.CUDA].get_kernel(f))
                    # create a new metadata for required dispatch key.
                    k_dict[f.func.name] = BackendMetadata(
                        NameHelper.supa_kernel(f), f.structured, CustomYamlProcessor.cpp_namespace
                    )
                    self.supported_functions.append(f)

        backend_indics.clear()
        backend_indics[CustomYamlProcessor.dispatch_key] = BackendIndex(
            CustomYamlProcessor.dispatch_key, True, False, True, kernels
        )
        backend_indics[CustomYamlProcessor.autograd_dispatch_key] = BackendIndex(
            CustomYamlProcessor.autograd_dispatch_key, True, False, True, autograd_kernels
        )

    def add_custom_op(
        self,
        functions: List[NativeFunction],
        backend_indics: BackendIndices,
        valid_tags: Set[str],
    ):
        """add customerized op into NativeFunction list and backend_index.

        Args:
            functions (List[NativeFunction]): native function list.
            backend_indics (BackendIndices): backend_index list
        """

        bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = {
            CustomYamlProcessor.dispatch_key: {},
            CustomYamlProcessor.autograd_dispatch_key: {},
            DispatchKey.CompositeExplicitAutograd: {},
        }
        for key, src in (
            (CustomYamlProcessor.dispatch_key, self.content["custom"]),
        ):
            if src is None:
                continue
            for e in src:
                loc = Location(self.path, e["__line__"])
                e["dispatch"] = {**e.get("dispatch", {}), **{str(key): "xxxxx"}}

                tags = e.get("tags", [])
                e["tags"] = tags.split(",") if isinstance(tags, str) else tags

                if ("rand" in e["func"]) or ("dropout" in e["func"]):  # for OP related with rand, force add a tag.
                    e["tags"].append("nondeterministic_seeded")

                f, m = NativeFunction.from_yaml(e, loc, valid_tags)
                object.__setattr__(f, "is_custom_supa", True)
                object.__setattr__(f, "supa_kernel", True)
                object.__setattr__(
                    f, "has_symint_implement", (f.func.name.name.__repr__() in self.symint_support_lists)
                )
                self.custom_functions.append(f)
                functions.append(f)
                for v in m[key].values():
                    object.__setattr__(v, "cpp_namespace", CustomYamlProcessor.cpp_namespace)
                    object.__setattr__(v, "kernel", NameHelper.supa_kernel(f))
                BackendIndex.grow_index(bs, m)

        backend_indics[CustomYamlProcessor.dispatch_key].index.update(bs[CustomYamlProcessor.dispatch_key])
        backend_indics[CustomYamlProcessor.autograd_dispatch_key].index.update(
            bs[CustomYamlProcessor.autograd_dispatch_key]
        )

        self.custom_functions.sort(key=lambda x: x.func.name.name.base)


class CtrlState:
    """helper class to handle state change of #if/#elif/#endif"""

    # extract condition expression in preprocessor directives
    condition_re = re.compile(r"#if\s+(\w+)\s*(<=|<|==|>=|>)\s*(\w*)")
    # extract force directives: #if 0/1/true/false
    force_condition_re = re.compile(r"^#if\s+(0|1|true|false)$")

    def __init__(self, name: str, intial: bool = True) -> None:
        """
        process
            #if TORCH_VER >= XXXXX
            #elif TORCH_VER > XXXXX
            #else
        do not support other syntax.
        """
        self.name: str = name
        self.is_version_flag: bool = False
        self.record: bool = False
        self.active: bool = intial
        if intial:
            self.check_condition(name)

    def _evaluate(self, cond: tuple) -> bool:
        var_a = TorchVersions.get(cond[0])
        op = cond[1]
        var_b = TorchVersions.get(cond[2])
        if op == ">":
            return var_a > var_b
        elif op == ">=":
            return var_a >= var_b
        elif op == "==":
            return var_a == var_b
        elif op == "<=":
            return var_a <= var_b
        elif op == "<":
            return var_a < var_b
        return False

    def check_condition(self, name: str):
        m = CtrlState.condition_re.search(name)
        if m:
            if "TORCH_VER" in m.groups():
                self.is_version_flag = True
                self.record = self._evaluate(m.groups())
        else:
            m = CtrlState.force_condition_re.match(name)
            if m:
                self.is_version_flag = True
                self.record = m.group(1) in ["1", "true"]

    def _change_record(self, new_flag):
        if not self.active:
            self.record = False
        else:
            if self.record:
                self.active = False

    def update_else(self):
        if self.record:
            self.active = False
        self.record = (not self.record) and self.active

    def update_elif(self, name: str):
        self.update_else()
        if self.active:
            self.check_condition(name)


class CppImplements:
    def __init__(self, paths: List[str]) -> None:
        self.kernel_files_paths: str = paths
        self.kernel_counter: Counter = Counter()
        """kernel implemented from source file.
        """
        self.supa_structs: Counter = Counter()
        self.struct_counter: Counter = None
        self.pattern_unstruct_impl = re.compile(r"^[^/]*?SUPANativeFunctions::([\w]+)\([^\)]*\)\s*{", re.MULTILINE)
        self.pattern_supa_struct_impl = re.compile(r"^[^/]*?SUPA_IMPL_FUNC\s*\((\w*)\)", re.MULTILINE)

    def __get_impl_names(self, file_path: str) -> Tuple[Counter, Counter]:
        """parse source file and extract valid implement enclosed by macro of TORCH_VER

        Args:
            file_path (str): source file name
            class_name (str): prefix tag for an op function.

        Raises:
            AssertionError: Unable to read file.

        Returns:
            Counter: dict[name, number]
        """
        kernel_name_counts = Counter()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                contents = f.readlines()
        except IOError:
            raise AssertionError(f"Unable to read from the specified impl_path file: {file_path}")

        # start to filter source file by current TORCH_VER
        current: CtrlState = CtrlState("global")
        current.record = True  # force set to record from beginning.
        def_symbol_stack: list[CtrlState] = [current]
        valid_content = ""
        for line in contents:
            stripped_line = line.strip()
            if stripped_line.startswith("#if"):
                assert not stripped_line.endswith("\\"), f"can't handle multi-line condition in {stripped_line}"
                current = CtrlState(stripped_line, current.record)
                def_symbol_stack.append(current)
            elif stripped_line.startswith("#else"):
                current.update_else()
            elif stripped_line.startswith("#elif"):
                assert not stripped_line.endswith("\\"), f"can't handle multi-line condition in {stripped_line}"
                current.update_elif(stripped_line)
            elif stripped_line.startswith("#endif"):
                def_symbol_stack.pop()
                current = def_symbol_stack[-1]
            elif current.record:
                valid_content += line

        # now, valid_content contains all effective source code. extract useful information..
        kernel_name_counts = Counter(self.pattern_unstruct_impl.findall(valid_content))
        supa_structs = Counter(self.pattern_supa_struct_impl.findall(valid_content))

        return kernel_name_counts, supa_structs

    def search_implements(self) -> None:
        """extract implement names from cpp source files.
        for structured implement, the name is OperatorName.unambiguous_name() of functional type.
        """
        for path in self.kernel_files_paths:
            for cur_dir, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(".cpp") or filename.endswith(".su"):
                        file_path = os.path.join(cur_dir, filename)
                        unstructured_counts, supa_structured_counts = self.__get_impl_names(
                            file_path
                        )
                        self.kernel_counter += unstructured_counts
                        self.supa_structs += supa_structured_counts
        for k, v in self.supa_structs.items():
            assert v == 1, f"multiple structured implement of '{k}'."

        self.struct_counter = self.supa_structs

    def checking_implements(
        self,
        grouped_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        structured_funcs: List[NativeFunctionsGroup],
        backend_indices: Dict[DispatchKey, BackendIndex],
        supported_functions: List[NativeFunction],
        custom_functions: List[NativeFunction],
    ) -> None:
        """check number of cpp implements of an OP with required in supa_native_functions.yaml.
           throw execption if number mismatch.
           self.kernel_counter : all implemented unstructured kernels.
           self.struct_counter : all implemented structured kerenels.
        Args:
            native_functions (List[NativeFunction]): _description_
            backend_indices (Dict[DispatchKey, BackendIndex]): _description_
            groups (List[NativeFucntionsGroup]): grouped functions, contains unstructured.
            supported_functions: include support, to_native and autograd
            custom_functions: include custom, customautograd.
        """
        print("checking existence")

        # getting all unstructured and structured declaration from yaml.
        expect_unstructured_name_counts: Dict[str, List[NativeFunction]] = defaultdict(list)
        for native_f in itertools.chain(supported_functions, custom_functions):
            if not native_f.structured:
                expect_unstructured_name_counts[NameHelper.unstructured(native_f)].append(native_f)

        expect_structured_names: List[str] = []
        for g in structured_funcs:
            if g.structured:
                kernel = NameHelper.structured(g)
                expect_structured_names.append(kernel)
        # check with actually found structured implements.
        struct_counter = set(self.struct_counter.keys())
        if len(struct_counter) != len(expect_structured_names):
            # number of structured implementation found is large than defined in yaml
            missing = struct_counter.symmetric_difference(expect_structured_names)
            assert (
                False
            ), f"mismatched structured IMPL for {missing}. check names in yaml and code.\n \
                FYI: must provides out kind, e.g 'add.out', in yaml.\n \
                must uses SUPA_IMPL_FUNC(xxx) where xxx matchs at::meta::structured_xxx"

        # check with actually found unstructured implements.
        for expected_name, funcs in expect_unstructured_name_counts.items():
            # count who has supa_kernel.
            expected_overload_count = sum(map(lambda x: getattr(x, "supa_kernel", False), funcs))
            actual_overload_count = self.kernel_counter.pop(expected_name, 0)

            # For structured op, only registering 'out' variant in yaml is sufficient.
            # The functional and inplace variants are automatically handled via structured_delegate.
            # Example: registering 'addmv.out' covers 'addmv' and 'addmv_' automatically.
            assert expected_overload_count != 0, (
                f"For '{expected_name}': structured op only needs 'out' variant in yaml. " +
                "Remove redundant functional/inplace variants.\n"
            )

            # op name is not defined as structured, so they are treat at unstructured implementation.
            assert actual_overload_count == expected_overload_count, (
                f"For '{expected_name}'\n found {actual_overload_count} implementations, but expect: {expected_overload_count}\n"
                + "\n".join(map(lambda x: NameHelper.declares(x), funcs))
            )
        assert len(self.kernel_counter) == 0, f"missing declaration in yaml for {list(self.kernel_counter.keys())}"

    def __get_metadata(
        self, g: NativeFunctionsGroup, backend_indices: Dict[DispatchKey, BackendIndex]
    ) -> Optional[BackendMetadata]:
        if (metadata := backend_indices[CustomYamlProcessor.dispatch_key].get_kernel(g)) is None:
            metadata = backend_indices[CustomYamlProcessor.autograd_dispatch_key].get_kernel(g)
        return metadata

    def check_group(
        self,
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        structured_functions: Sequence[NativeFunctionsGroup],
        backend_indices: Dict[DispatchKey, BackendIndex],
    ) -> None:
        """check group functions for those functions which not provide structured access api: TORCH_IMPL_FUNC()

        reason: 1. implement functional kind of a group function only, not the out kind.
                2. implement functional and out kind seperated, but not with structured format.
                during generating, it checks out structured code.
        """

        def part_of_structured_group(func):
            def inner(self):
                if hasattr(self, "force_structured"):
                    return self.force_structured
                return func.fget(self)

            return property(inner)

        NativeFunction.part_of_structured_group = part_of_structured_group(NativeFunction.part_of_structured_group)

        supported_structures: Sequence[NativeFunctionsGroup] = [
            g for g in structured_functions if (g.structured and NameHelper.structured(g) in self.struct_counter)
        ]
        unimplemented_group: Sequence[NativeFunctionsGroup] = [
            g for g in structured_functions if (g.structured and g not in supported_structures)
        ]
        for u in unimplemented_group:
            structured_functions.remove(u)
            # check structure flag
            metadata = backend_indices[CustomYamlProcessor.dispatch_key].get_kernel(u)
            if metadata and u.structured:
                print(
                    "regilster with SUPA_IMPL_FUNC rather than SUPANativeFunctions::xxxops \n"
                    f"HINT: {u.functional.func.name.name} is structured op."
                )
                assert False
            elif backend_indices[CustomYamlProcessor.autograd_dispatch_key].get_kernel(u) is not None:
                object.__setattr__(u.out, "structured", False)
                for f in u.functions():
                    object.__setattr__(f, "structured_delegate", None)
                    # force generating function declaration as structured in order to keep consistence with legacy SUPANativeFunctions
                    # cooperate with replaced property of NativeFunction.part_of_structured_group
                    object.__setattr__(f, "force_structured", True)
                    k = self.__get_metadata(f, backend_indices)

        def sort_method(f: Union[NativeFunction, NativeFunctionsGroup]):
            if isinstance(f, NativeFunction):
                return str(f.func.name.name)
            else:
                return str(f.functional.func.name.name)

        grouped_native_functions.sort(key=sort_method)

        # update tag in structured group : native struct or supa struct.
        for g in supported_structures:
            kernel_name = NameHelper.structured(g)
            supa_flag = kernel_name in self.supa_structs
            object.__setattr__(g, "has_supa_structs", supa_flag)
            if supa_flag:
                for f in g.functions():
                    if f.structured:
                        object.__setattr__(f, "supa_kernel", False)


class NameHelper:
    @staticmethod
    def structured(g: NativeFunctionsGroup) -> str:
        """return the structure name for FunctionGroup
        must match code under Target.ANONYMOUS_DEFINITION in register_dispatch_key.StructuredRegisterDispatchKey.gen_one()
        it must include overload_name. otherwise structure's name conflict. e.g. div.tensor vs div.tensor_mode
        """
        return meta.name(g)

    @staticmethod
    def unstructured(f: NativeFunction) -> str:
        """get the kernel name of unstructured function.
        it should be 'sysint_overload=f.func.has_symint()' but so far, not all.
        """
        return cpp.name(f.func, symint_overload=f.has_symint_implement)

    @staticmethod
    def supa_kernel(f: NativeFunction) -> str:
        if getattr(f, "is_custom_supa", None) is None:
            return ""
        return NameHelper.unstructured(f)

    @staticmethod
    def declares(f: NativeFunction) -> str:
        """generate cpp declaration for a function."""
        with native_function_manager(f):
            sig = DispatcherSignature.from_schema(f.func, symint=False)
            args_str = ", ".join(a.defn() for a in sig.arguments())
            name = NameHelper.unstructured(f)
            location = f"{os.path.basename(f.loc.file)}:{f.loc.line}"
            return f"{name}({args_str});  //{location}"

    @staticmethod
    def source_file_name(key: DispatchKey) -> str:
        if key == DispatchKey.PrivateUse1:
            return "SUPA"
        elif key == DispatchKey.AutogradPrivateUse1:
            return "AutogradSUPA"
        assert False, f"unsupported dispatch key {key}"


def gen_hint(f: NativeFunction) -> str:
    loc = f.loc
    return f"{os.path.basename(loc.file)}:{loc.line}  {str(f.func)}"
