# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.

import copy
from contextlib import contextmanager
from enum import Enum

import numpy as np
import torch

cpu_device = torch.device("cpu")
supa_device = torch.device("supa")

precisionsDefault = {
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-4),
    torch.float32: (1.3e-6, 1e-5),
}


def getDefaultRtolAndAtol(dtype):
    rtol = precisionsDefault.get(dtype, (0, 0))[0]
    atol = precisionsDefault.get(dtype, (0, 0))[1]
    return rtol, atol


class RandomMode(Enum):
    norm = 1
    uniform = 2
    range = 3


def create_torch_tensor_from_np(x, requires_grad=False, format=None):
    if isinstance(x, np.ndarray):
        if format is not None:
            cpu_input = (
                torch.from_numpy(copy.deepcopy(x))
                .contiguous()
                .to(memory_format=format)
                .to(cpu_device)
            )
            supa_input = (
                torch.from_numpy(copy.deepcopy(x))
                .contiguous()
                .to(memory_format=format)
                .to(supa_device)
            )
        else:
            cpu_input = torch.from_numpy(copy.deepcopy(x)).contiguous().to(cpu_device)
            supa_input = torch.from_numpy(copy.deepcopy(x)).contiguous().to(supa_device)
    elif isinstance(x, float):
        cpu_input = torch.tensor(x).contiguous().to(cpu_device)
        supa_input = torch.tensor(x).contiguous().to(supa_device)
    else:
        assert "Not support instance type!"

    cpu_input.requires_grad_(requires_grad)
    supa_input.requires_grad_(requires_grad)

    return cpu_input, supa_input


def create_random_tensor(
    shape,
    dtype,
    format=None,
    min_value=-5,
    max_value=5,
    requires_grad=False,
    mode: RandomMode = RandomMode.norm,
):
    if dtype == torch.bool:
        if min_value == 0 and max_value == 0:
            x = np.random.randint(0, 1, size=shape).astype(bool)
        elif min_value == 1 and max_value == 1:
            x = np.random.randint(1, 2, size=shape).astype(bool)
        else:
            x = np.random.randint(0, 2, size=shape).astype(bool)
    elif dtype == torch.half:
        if mode is RandomMode.uniform:
            x = np.random.uniform(min_value, max_value, shape).astype(np.float16)
        elif mode is RandomMode.norm:
            x = np.random.randn(*shape)
            if len(shape) != 0:
                x = x.astype(np.float16)
        elif mode is RandomMode.range:
            x = np.arange(min_value, max_value + 1, dtype=np.float16)
            np.random.shuffle(x)
            x = x[0 : np.prod(shape)]
            x = x.reshape(shape)
    elif dtype == torch.float:
        if mode is RandomMode.uniform:
            x = np.random.uniform(min_value, max_value, shape).astype(np.float32)
        elif mode is RandomMode.norm:
            # print("Warning: Actually, the generated data range will exceed [min_value, max_value]!!!")
            x = np.random.randn(*shape)
            if len(shape) != 0:
                x = x.astype(np.float32)
        elif mode is RandomMode.range:
            x = np.arange(min_value, max_value + 1, dtype=np.float32)
            np.random.shuffle(x)
            x = x[0 : np.prod(shape)]
            x = x.reshape(shape)
    elif dtype == torch.float64:
        x = np.random.randn(*shape).astype(np.float64)
    elif dtype == torch.int64:
        if mode is RandomMode.range:
            x = np.arange(min_value, max_value + 1, dtype=np.int64)
            np.random.shuffle(x)
            x = x[0 : np.prod(shape)]
            x = x.reshape(shape)
        else:
            x = np.random.randint(min_value, max_value + 1, size=shape).astype(np.int64)
    elif dtype == torch.int32:
        x = np.random.randint(min_value, max_value + 1, size=shape).astype(np.int32)
    elif dtype == torch.uint8:
        if mode is RandomMode.uniform:
            x = np.random.uniform(min_value, max_value, shape).astype(np.uint8)
        elif mode is RandomMode.norm:
            x = np.random.randn(*shape).astype(np.uint8)
    elif dtype == torch.int8:
        if mode is RandomMode.uniform:
            x = np.random.uniform(min_value, max_value, shape).astype(np.int8)
        elif mode is RandomMode.norm:
            x = np.random.randn(*shape).astype(np.int8)
    elif dtype == torch.bfloat16:
        if mode is RandomMode.uniform:
            x = torch.empty(shape, dtype=dtype)
            x.uniform_(min_value, max_value)
        elif mode is RandomMode.norm:
            x = torch.randn(shape, dtype=dtype)
        elif mode is RandomMode.range:
            x = np.arange(min_value, max_value + 1, dtype=np.float32)
            np.random.shuffle(x)
            x = x[0 : np.prod(shape)]
            x = x.reshape(shape)
            x = torch.tensor(x, dtype=dtype)
        x.requires_grad_(requires_grad)
        supa_input = torch.tensor(x, device=supa_device, requires_grad=requires_grad)
        return x, supa_input
    elif dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        x_cpu, x_supa = create_random_tensor(shape, torch.bfloat16, format, min_value, max_value,
                                             requires_grad, mode)
        return x_cpu.to(dtype), x_supa.to(dtype)
    else:
        raise TypeError("Unsuppported type!")
    return create_torch_tensor_from_np(x, requires_grad, format)


def assert_allclose(tensor_golden, tensor_compare, rtol, atol, equal_nan=False):
    if tensor_golden.dtype == tensor_compare.dtype and (tensor_compare.dtype == torch.bfloat16 or
       tensor_compare.dtype == torch.float8_e4m3fn or tensor_compare.dtype == torch.float8_e5m2):
        torch.testing.assert_close(
            tensor_compare.cpu(),
            tensor_golden.cpu(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
    else:
        try:
            golden_data = tensor_golden.cpu().detach().numpy()
            compare_data = tensor_compare.cpu().detach().numpy()
            np.testing.assert_allclose(golden_data, compare_data, rtol, atol, equal_nan)
        except AssertionError as e:
            data_compare(tensor_golden, tensor_compare)
            raise e


def assert_allclose_inds_set(tensor_golden, tensor_compare, rtol, atol):
    golden_data_set = set(tensor_golden.cpu().flatten().detach().tolist())
    compare_data_set = set(tensor_compare.cpu().flatten().detach().tolist())
    if tensor_golden.numel() != len(golden_data_set):
        print(
            "Warning: assert_allclose_inds_set func should be given distinct inds as input, found golden has repeated elements"
        )
    if tensor_compare.numel() != len(compare_data_set):
        print(
            "Warning: assert_allclose_inds_set func should be given distinct inds as input, found compare data has repeated elements"
        )
    set_union = golden_data_set | compare_data_set
    set_intersection = golden_data_set & compare_data_set
    jaccard_index = len(set_intersection) / len(set_union)
    jaccard_index_thr = (1 - rtol) - atol / len(set_union)
    assert (
        jaccard_index >= jaccard_index_thr
    ), f"Assert Fail !!!, golden_length: {tensor_golden.numel()}, compare_length: {tensor_compare.numel()}, rtol: {rtol}, atol: {atol}, jaccard_index: {jaccard_index}, jaccard_index_thr: {jaccard_index_thr}"


def data_compare(tensor_golden, tensor_supa):
    golden_data = tensor_golden.cpu().detach().numpy()
    supa_data = tensor_supa.cpu().detach().numpy()

    golden_data = golden_data.flatten()
    supa_data = supa_data.flatten()
    abs_data = np.abs(golden_data - supa_data)
    index1 = np.argsort(abs_data)[::-1]
    index2 = np.argsort(abs_data / np.abs(golden_data))[::-1]
    index_range = 10 if len(index1) > 10 else len(index1)
    print("A Top10 Flatten Index And Data:")
    for i in range(index_range):
        print(
            index1[i], golden_data[index1[i]], supa_data[index1[i]], abs_data[index1[i]]
        )
    print("R Top10 Flatten Index And Data:")
    for i in range(index_range):
        print(
            "{}, {:.6f}, {:.6f}, {:.6f}".format(
                index2[i],
                golden_data[index2[i]],
                supa_data[index2[i]],
                abs_data[index2[i]] / np.abs(golden_data)[index2[i]],
            )
        )


def assert_equal(tensor_golden, tensor_compare):
    golden_data = tensor_golden.cpu().detach().numpy()
    compare_data = tensor_compare.cpu().detach().numpy()
    try:
        np.testing.assert_equal(golden_data, compare_data)
    except AssertionError as e:
        data_compare(tensor_golden, tensor_compare)
        raise e


def _fail(msg):
    assert False, msg


def assertRtolEqual(x, y, prec=None, prec16=None):
    def compare_res(pre, minimum):
        result = np.abs(y - x)
        deno = np.maximum(np.abs(x), np.abs(y))
        result_atol = np.less_equal(result, pre)
        result_rtol = np.less_equal(result / np.add(deno, minimum), pre)
        if not result_rtol.all() and not result_atol.all():
            if (
                np.sum(not result_rtol) > size * pre
                and np.sum(not result_atol) > size * pre
            ):
                _fail("result error")

    threshold = 1.0e-4
    threshold2 = 1.0e-3
    minimum16 = 6e-8
    minimum = 10e-10
    if prec is None:
        prec = threshold
    if prec16 is None:
        prec16 = threshold2
    if torch.is_tensor(x) and torch.is_tensor(y):
        x = x.numpy()
        y = y.numpy()
    size = x.size
    if x.shape != y.shape:
        _fail("shape error")
    if x.dtype != y.dtype:
        _fail("dtype error")
    dtype_list = [
        np.bool_,
        np.uint16,
        np.int16,
        np.int32,
        np.float16,
        np.float32,
        np.int8,
        np.uint8,
        np.int64,
        np.float64,
    ]
    if x.dtype not in dtype_list:
        _fail(
            "required dtype in [np.bool, np.uint16, np.int16, "
            + "np.int32, np.float16, np.float32, np.int8, np.uint8, np.int64]"
        )
    if x.dtype == np.bool_:
        result = np.equal(x, y)
        if result.all() is False:
            _fail("result error")
    elif x.dtype == np.float16:
        compare_res(prec16, minimum16)
    elif x.dtype in [
        np.float32,
        np.int8,
        np.uint8,
        np.uint16,
        np.int16,
        np.int32,
        np.int64,
        np.float64,
    ]:
        compare_res(prec, minimum)
    else:
        _fail("required numpy object")


@contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    yield
    torch.set_rng_state(rng_state)


def get_transposed_shape(shape):
    a = torch.rand(shape, dtype=torch.float)
    a = a.transpose(0, 1)
    return a.shape


def generate_uncontiguous_tensor(shape, dtype):
    if isinstance(shape, int):
        shape *= 2
    else:
        shape = get_transposed_shape(shape)

    result, result_supa = create_random_tensor(shape, dtype)

    if isinstance(shape, int):
        result = result[-(shape // 2) :]
        result_supa = result_supa[-(shape // 2) :]
    else:
        result = result.transpose(0, 1)
        result_supa = result_supa.transpose(0, 1)

    return result, result_supa
