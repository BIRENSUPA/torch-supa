# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
import pytest

from torch_supa.testing.common_utils import assert_allclose, create_random_tensor, RandomMode
import itertools

shapes = [
    (1,),
    (8, 507),
    (133, 16, 16),
    (1, 3, 64, 64),
]

vector_cases = [[[128], torch.float32, torch.bfloat16], [[9126], torch.int32, torch.bfloat16]]

dtypes = ["bool", "int8", "int", "long", "uint8", "bfloat16", "half", "float32", "double"]

# double to int may cause mismatch
unsupported_dtypes = [
    ("double", "int8"),
    ("double", "int"),
    ("double", "long"),
    ("double", "uint8"),
]

def gen_src_dst_dtype(dtypes):
    src_dst_dtypes_list = []
    # kernel does not support double
    dtypes.remove("double")
    for src_dst_dtype in list(itertools.permutations(dtypes, 2)):
        if src_dst_dtype in unsupported_dtypes:
            continue
        else:
            src_dst_dtypes_list.append(list(src_dst_dtype))
    return src_dst_dtypes_list


src_dst_dtypes_list = gen_src_dst_dtype(dtypes)

class TestCast:
    def run_cast(self, shape, src_dtype, dst_dtype, perf):
        min_value = 0
        max_value = 0
        if src_dtype == torch.uint8 or dst_dtype == torch.uint8:
            min_value = 0
            max_value = 255
        elif src_dtype == torch.int8 or dst_dtype == torch.int8:
            min_value = -128
            max_value = 127
        else:
            min_value = -1000
            max_value = 1000

        cpu_input, supa_input = create_random_tensor(
            shape, min_value=min_value, max_value=max_value, dtype=src_dtype, mode=RandomMode.uniform
        )

        y_supa = supa_input.to(dst_dtype)
        # skip cpu golden test
        if not perf:
            y_cpu = cpu_input.to(dst_dtype)

            assert_allclose(y_cpu, y_supa, rtol=0, atol=0)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("src_dtype, dst_dtype", src_dst_dtypes_list)
    def test_cast(self, shape, src_dtype, dst_dtype):
        src_dtype = eval("torch." + src_dtype)
        dst_dtype = eval("torch." + dst_dtype)
        self.run_cast(shape, src_dtype, dst_dtype, False)


    @pytest.mark.parametrize("scalar", [3.0])
    @pytest.mark.parametrize("src_dtype, dst_dtype", src_dst_dtypes_list)
    def test_cast_scalar(self, src_dtype, dst_dtype, scalar):
        src_dtype = eval("torch." + src_dtype)
        dst_dtype = eval("torch." + dst_dtype)
        cpu_input = torch.tensor(scalar).to(src_dtype)
        supa_input = cpu_input.to("supa")

        y_cpu = cpu_input.to(dst_dtype)
        y_supa = supa_input.to(dst_dtype)

        assert_allclose(y_cpu, y_supa, rtol=0, atol=0)
