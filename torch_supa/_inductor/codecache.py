# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import os
import contextlib
import hashlib
import json
from typing import (
    Any,
    Dict,
    List,
    Union,
    Optional
)

import torch
from torch._inductor import config
from torch._inductor.codecache import CacheBase, get_lock_dir, LOCK_TIMEOUT
from torch._inductor.graph import GraphLowering
import torch_supa

empty_json = "{}"


@contextlib.contextmanager
def lock_context(key):
    from filelock import FileLock
    lock_dir = get_lock_dir()
    lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
    with lock:
        yield

def patch_cache_base_get_system():
    # patch function CacheBase.get_system with get_system, add logic to support supa
    @staticmethod
    def get_system():
        from .triton_compat import HAS_TRITON, triton_key

        if HAS_TRITON:
            # Use triton_key instead of triton.__version__ as the version
            # is not updated with each code change
            triton_version = triton_key()
        else:
            triton_version = None

        try:
            system: Dict[str, Any] = {
                "device": {"name": None},
                "version": {
                    "triton": triton_version,
                },
            }
            device_properties = torch_supa.supa.get_device_properties(
                torch_supa.supa.current_device()
            )
            if torch.version.supa is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["supa"] = torch.version.supa
            elif torch.version.cuda is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["cuda"] = torch.version.cuda
            else:
                system["device"]["name"] = device_properties.gcnArchName
                system["version"]["hip"] = torch.version.hip
        except (AssertionError, RuntimeError):
            # If deivce is not installed, none of the above config is relevant.
            system = {}

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return system

    CacheBase.get_system = get_system


def patch_aot_code_compiler_compile():
    # In v2.6.0, aoti has bug when init oss_proxy_executor with default op_json,
    # which could not be skipped, so here we try to create a new supa op_json,
    # and clear the content of default op_json.
    from torch._inductor.codecache import AotCodeCompiler

    AotCodeCompiler.src_compile = AotCodeCompiler.compile

    @classmethod
    def compile_supa(
        cls,
        graph: GraphLowering,
        wrapper_code: str,
        kernel_code: str,
        serialized_extern_kernel_nodes: Optional[str],
        *,
        device_type: str,
        additional_files: list[str],
    ) -> Union[List[str], str]:
        result = cls.src_compile(
            graph,
            wrapper_code,
            kernel_code,
            serialized_extern_kernel_nodes,
            device_type=device_type,
            additional_files=additional_files,
        )
        generated_files = additional_files
        if not config.aot_inductor.package:
            return result

        output_so = [r for r in result if r.endswith(".so")]
        if len(output_so) > 1:
            raise RuntimeError(
                f"Could not generate supa op json, because there are"
                f"more than one so in generated files: {result}"
            )
        output_so = output_so[0]
        key = os.path.basename(output_so)[0].replace(".", "_")
        dir_basename = os.path.splitext(output_so)[0]
        with lock_context(key):
            if serialized_extern_kernel_nodes:
                extern_kernel_nodes_json = dir_basename + "_supa.json"
                with open(extern_kernel_nodes_json, "w") as f:
                    f.write(serialized_extern_kernel_nodes)
                generated_files.append(extern_kernel_nodes_json)

            if serialized_extern_kernel_nodes:
                source_json_file = dir_basename + ".json"
                with open(source_json_file, "w") as f:
                    f.write(empty_json)
        return generated_files

    AotCodeCompiler.compile = compile_supa
