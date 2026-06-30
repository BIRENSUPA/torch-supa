import re
import os
import sys
import traceback
import torch
from torchgen.utils import FileManager

def get_torch_version():
    try:
        return re.split(r"^([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}).*", torch.__version__)[1]
    except:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)

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
                "2.4.0",
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
