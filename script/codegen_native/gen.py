import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from packaging import version

from torchgen.gen import get_torchgen_root
from .utils import TorchVersions, get_torch_version

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")


def run_cmd(cmd: list[str]) -> None:
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        capture_output=True,
    )
    stdout, stderr = (
        result.stdout.decode("utf-8").strip(),
        result.stderr.decode("utf-8").strip(),
    )
    # print(stdout, stderr)
    if result.returncode != 0:
        print(f"Failed to run {cmd}")
        print(stdout, stderr)
        sys.exit(1)


def resolve_autograd_codegen() -> tuple[str, str, str, dict[str, str]]:
    torchgen_root = get_torchgen_root()
    packaged_tags_yaml = os.path.join(
        torchgen_root, "packaged/ATen/native/tags.yaml"
    )
    packaged_autograd_dir = os.path.join(torchgen_root, "packaged/autograd")
    if not os.path.exists(packaged_tags_yaml) or not os.path.exists(
        os.path.join(packaged_autograd_dir, "gen_autograd.py")
    ):
        raise RuntimeError(
            "failed to find autograd codegen resources from installed torchgen package"
        )

    return (
        "torchgen.packaged.autograd.gen_autograd",
        packaged_tags_yaml,
        packaged_autograd_dir,
        os.environ.copy(),
    )


def process_includes(content: str, header_map: dict) -> str:
    lines = content.splitlines(keepends=True)
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#include"):
            matched = False
            for pattern, replacement in header_map.items():
                if re.search(pattern, line):
                    if replacement is not None:
                        result.append(re.sub(pattern, replacement, line))
                    matched = True
                    break
            if not matched:
                result.append(line)
        else:
            result.append(line)
    return "".join(result)


def strip_quantized_registration_blocks(lines: list[str], ops: set[str]) -> list[str]:
    if not ops:
        return lines

    result = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m)" not in line:
            result.append(line)
            idx += 1
            continue

        block = [line]
        idx += 1
        brace_depth = line.count("{") - line.count("}")
        while idx < len(lines):
            block.append(lines[idx])
            brace_depth += lines[idx].count("{") - lines[idx].count("}")
            idx += 1
            if brace_depth <= 0:
                break

        block_text = "".join(block)
        if any(f'm.impl("{op}"' in block_text for op in ops):
            continue
        result.extend(block)

    return result


def transform_cuda_to_privateuse1(files, dst_path, skip_quantized_registration_ops=None):
    os.makedirs(dst_path, exist_ok=True)
    skip_quantized_registration_ops = skip_quantized_registration_ops or set()

    # 定义头文件映射规则
    HEADER_MAP = {
        # c10 headers
        r"<c10/cuda/CUDAGuard\.h>": r'"torch_supa/csrc/core/supa/SUPAGuard.h"',
        r"<c10/cuda/CUDAMacros\.h>": r'"torch_supa/csrc/core/supa/SUPAMacros.h"',
        r"<c10/cuda/CUDAStream\.h>": r'"torch_supa/csrc/core/supa/SUPAStream.h"',
        r"<c10/cuda/CUDAFunctions\.h>": r'"torch_supa/csrc/core/supa/SUPAFunctions.h"',
        # ATen headers
        r"<ATen/cuda/ATenCUDAGeneral\.h>": None,
        r"<ATen/cuda/CUDADevice\.h>": None,
        r"<ATen/cuda/EmptyTensor.h>": r'"torch_supa/csrc/aten/common/EmptyTensor.h"',
        r"<ATen/cuda/CUDAContext\.h>": r'"torch_supa/csrc/core/supa/SUPAContext.h"',
        # change cuda api name in the future
        # r"<ATen/NativeMetaFunctions.h>": r'"torch_supa/csrc/aten/NativeMetaFunctions.h"',
        # r'<ATen/NativeFunctions\.h>': r'"torch_supa/csrc/aten/NativeFunctions.h"',
    }

    for src in files:
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                # replace headers
                if "#" in line:
                    matched = False
                    for pattern, replacement in HEADER_MAP.items():
                        if re.search(pattern, line):
                            matched = True
                            if replacement is not None:
                                line = re.sub(pattern, replacement, line)
                                new_lines.append(line)
                            break

                    if not matched:
                        new_lines.append(line)
                # replace definitions
                else:
                    # api name change in the future
                    # if "at::native" in line:
                    #     if "return" in line or "struct" in line:
                    #         line = line.replace("at::native", "at::supa::native")

                    line = line.replace(
                        "c10::cuda::OptionalCUDAGuard", "c10::supa::OptionalSUPAGuard"
                    )
                    if "at::detail" in line:
                        # empty_supa
                        line = line.replace("CUDA", "SUPA")
                        line = line.replace("cuda", "supa")

                    if ("globalContext().lazyInitCUDA()" in line):
                        line = line.replace("globalContext().lazyInitCUDA()",
                                            "c10::supa::SupaSysCtrl::GetInstance().supaInit();")

                    if (
                        "at::native" not in line
                        and "struct" not in line
                        and "namespace" not in line
                    ):
                        line = line.replace("CUDA", "PrivateUse1")
                        line = line.replace("cuda", "privateuse1")

                    new_lines.append(line)

            filename = os.path.basename(src)
            new_filename = filename.replace("CUDA", "SUPANative")
            dst = os.path.join(dst_path, new_filename)
            if new_filename.startswith("RegisterQuantizedSUPANative"):
                new_lines = strip_quantized_registration_blocks(
                    new_lines, skip_quantized_registration_ops
                )
            file = Path(dst)
            old_contents: str | None = None
            try:
                old_contents = file.read_text(encoding="utf-8")
            except OSError:
                pass
            if "".join(new_lines) != old_contents:
                with open(dst, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)


_AUTOGRAD_FOOTER = """
} // namespace
} // namespace VariableType

namespace {

{registrations}

} // namespace
} // namespace torch::autograd
"""


_AUTOGRAD_UNPACK_HELPERS = """
namespace {
const Variable& checked_cast_variable(
    const Tensor& t,
    const char* name,
    int pos) {
  if (!t.defined()) {
    TORCH_CHECK(
        false,
        "Expected a proper Tensor but got None (or an undefined Tensor in C++) ",
        "for argument #",
        pos,
        " '",
        name,
        "'");
  }
  return t;
}

Variable& checked_cast_variable(Tensor& t, const char* name, int pos) {
  if (!t.defined()) {
    TORCH_CHECK(
        false,
        "Expected a proper Tensor but got None (or an undefined Tensor in C++) ",
        "for argument #",
        pos,
        " '",
        name,
        "'");
  }
  return t;
}

const Tensor& unpack(const Tensor& t, const char* name, int pos) {
  return checked_cast_variable(t, name, pos);
}

Tensor& unpack(Tensor& t, const char* name, int pos) {
  return checked_cast_variable(t, name, pos);
}

Tensor unpack_opt(const Tensor& t, const char* name, int pos) {
  if (!t.defined()) {
    return Tensor();
  }
  return unpack(t, name, pos);
}

std::vector<at::Tensor> unpack(
    const at::ITensorListRef& tl,
    const char* name,
    int pos) {
  std::vector<at::Tensor> ret;
  ret.reserve(tl.size());
  for (const auto& t : tl) {
    ret.push_back(t);
  }
  return ret;
}
} // namespace

"""


def run_autograd_codegen(native_yaml: str, generated_dir: str) -> None:
    module, tags_yaml, autograd_dir, env = resolve_autograd_codegen()
    print(
        "Running:",
        [
            sys.executable,
            "-m",
            module,
            native_yaml,
            tags_yaml,
            generated_dir,
            autograd_dir,
        ],
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            native_yaml,
            tags_yaml,
            generated_dir,
            autograd_dir,
        ],
        capture_output=True,
        env=env,
    )
    stdout, stderr = (
        result.stdout.decode("utf-8").strip(),
        result.stderr.decode("utf-8").strip(),
    )
    if result.returncode != 0:
        print("Failed to run autograd codegen")
        print(stdout, stderr)
        sys.exit(1)


def _extract_balanced_block(text: str, start: int) -> tuple[str, int]:
    open_brace = text.find("{", start)
    if open_brace < 0:
        raise RuntimeError("failed to find block opening brace")
    depth = 0
    for idx in range(open_brace, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                while end < len(text) and text[end] in " \t\r\n;":
                    end += 1
                return text[start:end], end
    raise RuntimeError("failed to find block closing brace")


def _extract_function(text: str, name: str) -> str:
    match = re.search(rf"\n[^\n]*\b{re.escape(name)}\s*\(", text)
    if match is None:
        raise RuntimeError(f"failed to find generated autograd wrapper {name}")
    start = match.start() + 1
    block, _ = _extract_balanced_block(text, start)
    return block.strip() + "\n"


def _find_variable_type_helpers(text: str) -> re.Match[str]:
    match = re.search(r"namespace\s+VariableType\s*\{\s*(namespace\s*\{)", text)
    if match is None:
        raise RuntimeError("failed to find VariableType helper namespace")
    return match


def _extract_variable_type_prologue(text: str) -> str:
    match = _find_variable_type_helpers(text)
    return text[: match.start(1)].rstrip() + "\n"


def _extract_variable_type_helpers(text: str) -> str:
    match = _find_variable_type_helpers(text)
    block, _ = _extract_balanced_block(text, match.start(1))
    return block.strip() + "\n\n"


def transform_autograd_cuda_to_privateuse1(files: list[str], dst_path: str) -> None:
    os.makedirs(dst_path, exist_ok=True)
    for src in files:
        if not os.path.exists(src):
            continue
        text = Path(src).read_text(encoding="utf-8")
        reg_match = re.search(
            r"TORCH_LIBRARY_IMPL\(aten, AutogradCUDA, m\) \{.*?\n\}", text, re.S
        )
        if reg_match is None:
            continue
        registration = reg_match.group(0)
        wrapper_names = re.findall(r"VariableType::([A-Za-z0-9_]+_AutogradCUDA)", registration)
        if not wrapper_names:
            continue

        prologue = _extract_variable_type_prologue(text)
        helpers = _extract_variable_type_helpers(text)
        functions = []
        seen = set()
        for name in wrapper_names:
            if name in seen:
                continue
            seen.add(name)
            privateuse1_name = name.replace("AutogradCUDA", "AutogradPrivateUse1")
            functions.append(_extract_function(text, name).replace(name, privateuse1_name))

        registration = registration.replace("AutogradCUDA", "AutogradPrivateUse1")
        content = (
            prologue
            + helpers
            + _AUTOGRAD_UNPACK_HELPERS
            + "namespace {\n\n"
            + "\n".join(functions)
            + _AUTOGRAD_FOOTER.replace("{registrations}", registration)
        )
        filename = os.path.basename(src).replace("VariableType_", "RegisterAutogradSUPANative_")
        dst = Path(dst_path) / filename
        old_contents: str | None = None
        try:
            old_contents = dst.read_text(encoding="utf-8")
        except OSError:
            pass
        if content != old_contents:
            dst.write_text(content, encoding="utf-8")


def transform_native_functions(src, dst_path):
    os.makedirs(dst_path, exist_ok=True)

    HEADER_MAP = {
        r"<ATen/NativeMetaFunctions.h>": r'"torch_supa/csrc/aten/NativeMetaFunctions.h"',
    }

    if os.path.exists(src):
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()

        # replace namespace and api
        content = process_includes(content, HEADER_MAP)
        content = content.replace("namespace at", "namespace at::supa")
        content = content.replace("at::meta", "at::supa::meta")
        content = content.replace("CUDA", "PrivateUse1")
        content = content.replace("cuda", "privateuse1")

        filename = os.path.basename(src)
        new_filename = filename.replace("CUDA", "SUPA")
        dst = os.path.join(dst_path, new_filename)

        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)


def transform_native_meta_functions(src, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    if os.path.exists(src):
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()

        content = content.replace("namespace at", "namespace at::supa")

        filename = os.path.basename(src)
        new_filename = filename.replace("CUDA", "SUPA")
        dst = os.path.join(dst_path, new_filename)

        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)


def transform_native_yaml(src, dst):
    with open(src, "r", encoding="utf-8") as f:
        native_funcs = yaml.safe_load(f)

    br_src = os.path.join(BASE_DIR, "torch_supa/csrc/aten/supa_native_functions.yaml")
    with open(br_src, "r", encoding="utf-8") as f:
        supa_native_funcs = yaml.safe_load(f)

    FILTERED_KEYS = [
        "miopen",  # miopen op
        "fft",  # fft op
        "_cholesky_solve_helper",  # cusovler op
        "sparse",  # sparse op  !!!!
        "_cslt_compress",
        "_convert_indices_from_csr_to_coo",
        "_convert_indices_from_coo_to_csr",
    ]

    BR_FUNC_LISTS = []
    for tag in ["supported", "perf_supported", "autograd"]:
        tag_ops = supa_native_funcs.get(tag)
        if tag_ops is not None:
            BR_FUNC_LISTS += tag_ops

    PRESERVE_QUANTIZED_DISPATCH = set(supa_native_funcs.get("preserve_quantized_dispatch", []) or [])

    def keep_quantized_cuda_dispatch(entry: dict) -> dict | None:
        dispatch = entry.get("dispatch")
        if not isinstance(dispatch, dict):
            return None

        quantized_dispatch = {}
        for dispatch_keys, kernel in dispatch.items():
            keys = [key.strip() for key in dispatch_keys.split(",")]
            if "QuantizedCUDA" in keys:
                quantized_dispatch["QuantizedCUDA"] = kernel

        if not quantized_dispatch:
            return None

        preserved_entry = entry.copy()
        preserved_entry["dispatch"] = quantized_dispatch
        return preserved_entry

    def should_preserve_quantized_dispatch(func_str: str, structured_str: str) -> bool:
        return (func_str in PRESERVE_QUANTIZED_DISPATCH) or (structured_str in PRESERVE_QUANTIZED_DISPATCH)

    def should_keep(func_str: str, structured_str: str) -> bool:
        return (not any(key in func_str for key in FILTERED_KEYS)) and \
               (func_str not in BR_FUNC_LISTS) and \
               (structured_str not in BR_FUNC_LISTS)

    filtered_funcs = []
    for entry in native_funcs:
        if isinstance(entry, dict):
            """
            Filt function name and structured_delgate name. For example, if "sqrt_out" is in supa_native_functions.yaml,
            we need to remove "sqrt", "sqrt_", and "sqrt_out" from native_functions.yaml.
            Then we could use SUPA_IMPL_FUNC or SUPANativeFunctions to implement "sqrt_out".
            """
            func_str = entry.get("func", "").split("(")[0]
            structured_str = entry.get("structured_delegate", "")
            if should_keep(func_str, structured_str):
                filtered_funcs.append(entry)
            elif should_preserve_quantized_dispatch(func_str, structured_str):
                preserved_entry = keep_quantized_cuda_dispatch(entry)
                if preserved_entry is not None:
                    filtered_funcs.append(preserved_entry)
        elif isinstance(entry, list):
            kept_overloads = []
            for sub in entry:
                if not isinstance(sub, dict):
                    continue
                func_str = sub.get("func", "").split("(")[0]
                structured_str = sub.get("structured_delegate", "")
                if should_keep(func_str, structured_str):
                    kept_overloads.append(sub)
                elif should_preserve_quantized_dispatch(func_str, structured_str):
                    preserved_entry = keep_quantized_cuda_dispatch(sub)
                    if preserved_entry is not None:
                        kept_overloads.append(preserved_entry)
            if kept_overloads:
                filtered_funcs.append(kept_overloads)

    with open(dst, "w", encoding="utf-8") as f:
        yaml.dump(
            filtered_funcs,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def main():
    native_yaml = os.path.join(
        get_torchgen_root(), "packaged/ATen/native/native_functions.yaml"
    )
    src_dir = os.path.join(get_torchgen_root(), "packaged/ATen")
    dst_dir = os.path.join(BASE_DIR, "build/aten/ATen")

    # remove previous generated files
    generated_dir = os.path.join(BASE_DIR, "build/aten/generated")
    if os.path.exists(generated_dir):
        shutil.rmtree(generated_dir)
    autograd_generated_dir = os.path.join(BASE_DIR, "build/aten/autograd_generated")
    if os.path.exists(autograd_generated_dir):
        shutil.rmtree(autograd_generated_dir)

    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    transform_native_yaml(
        native_yaml, os.path.join(dst_dir, "native/native_functions.yaml")
    )

    run_cmd(
        [
            sys.executable,
            "-m",
            "torchgen.gen",
            "-s",
            dst_dir,
            "-d",
            generated_dir,
            "--backend-whitelist",
            "CUDA",
            "SparseCUDA",
            "SparseCsrCUDA",
            "QuantizedCUDA",
            "NestedTensorCUDA"
        ]
    )

    # native_functions = "build/aten/generated/NativeFunctions.h"
    # native_meta_functions = "build/aten/generated/NativeMetaFunctions.h"

    torch_version = get_torch_version()
    use_new_register = version.parse(torch_version) >= version.parse("2.8")
    if use_new_register:
        register_files = [
            "build/aten/generated/RegisterCUDA_0.cpp",
            # "build/aten/generated/RegisterSparseCUDA_0.cpp",
            # "build/aten/generated/RegisterSparseCsrCUDA_0.cpp",
            "build/aten/generated/RegisterQuantizedCUDA_0.cpp",
            "build/aten/generated/RegisterNestedTensorCUDA_0.cpp",
        ]
    else:
        register_files = [
            "build/aten/generated/RegisterCUDA.cpp",
            # "build/aten/generated/RegisterSparseCUDA.cpp",
            # "build/aten/generated/RegisterSparseCsrCUDA.cpp",
            "build/aten/generated/RegisterQuantizedCUDA.cpp",
            "build/aten/generated/RegisterNestedTensorCUDA.cpp",
        ]

    register_files = [os.path.join(BASE_DIR, rel_path) for rel_path in register_files]
    register_dst_dir = os.path.join(BASE_DIR, "torch_supa/csrc/aten/generated")

    # currently do not change the api name
    # transform_native_functions(native_functions, dst)
    # transform_native_meta_functions(native_meta_functions, dst)
    br_src = os.path.join(BASE_DIR, "torch_supa/csrc/aten/supa_native_functions.yaml")
    with open(br_src, "r", encoding="utf-8") as f:
        supa_native_funcs = yaml.safe_load(f)
    skip_quantized_registration_ops = set(
        supa_native_funcs.get("preserve_quantized_dispatch", []) or []
    )
    transform_cuda_to_privateuse1(
        register_files, register_dst_dir, skip_quantized_registration_ops
    )

    run_autograd_codegen(native_yaml, autograd_generated_dir)
    autograd_register_files = [
        str(path) for path in sorted(Path(autograd_generated_dir).glob("VariableType_*.cpp"))
    ]
    transform_autograd_cuda_to_privateuse1(autograd_register_files, register_dst_dir)

    # generate torch version file
    TorchVersions.init(torch_version)
    TorchVersions.generate_ver_file(os.path.join(os.path.join(BASE_DIR, "torch_supa/csrc/core/supa"), "TorchVersion.h"))


if __name__ == "__main__":
    main()
