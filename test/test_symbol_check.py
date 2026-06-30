# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import shutil
import subprocess
from pathlib import Path
from typing import Set

import pytest
import suda
import torch_supa


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        pytest.skip(f"required tool not found: {name}")


def _find_torch_lib(name) -> Path:
    torch_supa_dir = Path(torch_supa.__file__).resolve().parent
    candidate = torch_supa_dir / "lib" / name
    if candidate.exists():
        return candidate
    raise AssertionError(
        f"{name} not found under torch_supa package path {torch_supa_dir}"
    )

def _extract_symbols_from_lib(lib_path: Path) -> Set[str]:
    result = subprocess.check_output(
        ["nm", "-gP", "--defined-only", str(lib_path)],
        stderr=subprocess.PIPE,
        text=True,
    )

    symbols = set()
    for line in result.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 2:
            continue
        symbol = parts[0]
        type_info = parts[1]
        if symbol and "cu" in symbol and not symbol.startswith("_ZT") and type_info != "W" and "supa" not in symbol and "suda" not in symbol:
            symbols.add(symbol)
    return symbols


def _extract_strings_from_lib(lib_path: Path) -> Set[str]:
    result = subprocess.check_output(
        ["strings", "-n", "8", str(lib_path)],
        stderr=subprocess.PIPE,
        text=True,
    )
    return {line.strip() for line in result.splitlines() if line.strip()}


def _collect_suda_symbols() -> Set[str]:
    suda_root = Path(suda.__file__).resolve().parent
    suda_lib_dir = suda_root / "_cuda" / "lib"
    allowed_libs = {
        "libcublas_static.a",
        "libcublasLt_static.a",
        "libcudnn_static.a",
        "libcudart_static.a",
        "libnvrtc_static.a",
    }

    if not suda_lib_dir.exists():
        pytest.skip("suda lib directory not found: {}".format(suda_lib_dir))

    libs = sorted(lib for lib in suda_lib_dir.glob("*_static.a") if lib.name in allowed_libs)
    print(libs)
    if not libs:
        pytest.skip("no target static libs found under {}".format(suda_lib_dir))

    symbols = set()
    for lib in libs:
        symbols.update(_extract_symbols_from_lib(lib))
    return symbols


@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.single
def test_torch_suda_op_symbol_stripped():
    for tool in ("nm", "strings"):
        _require_tool(tool)

    torch_suda_op = _find_torch_lib("libtorch_suda_op.so")
    suda_symbols = _collect_suda_symbols()
    lib_strings = _extract_strings_from_lib(torch_suda_op)

    matched_symbols = sorted(
        symbol
        for symbol in suda_symbols
        if any(symbol in string and "at4cuda6detail6_stubs" not in string for string in lib_strings)
    )

    assert not matched_symbols, (
        "found cuda symbols in {}: {}".format(torch_suda_op, ", ".join(matched_symbols))
    )


@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.single
def test_torch_supa_symbol_stripped():
    for tool in ("nm", "strings"):
        _require_tool(tool)

    torch_suda_op = _find_torch_lib("libtorch_supa_op.so")
    suda_symbols = _collect_suda_symbols()
    lib_strings = _extract_strings_from_lib(torch_suda_op)

    matched_symbols = sorted(
        symbol
        for symbol in suda_symbols
        if any(symbol in string and "at4cuda6detail6_stubs" not in string for string in lib_strings)
    )

    assert not matched_symbols, (
        "found cuda symbols in {}: {}".format(torch_suda_op, ", ".join(matched_symbols))
    )
