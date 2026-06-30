# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch.nn.functional as F
from torch_supa.testing.common_utils import assert_allclose, create_random_tensor

RTOL = {torch.float32: 5e-5, torch.bfloat16: 0.016}
ATOL = {torch.float32: 5e-5, torch.bfloat16: 1e-3}

dtype = [torch.float32, torch.bfloat16]
input_shape = [(2, 3)]

class TestBatchNorm:
    @pytest.mark.sanity
    @pytest.mark.regression
    @pytest.mark.parametrize("dtype", dtype)
    @pytest.mark.parametrize("input_shape", input_shape)
    def test_native_batch_norm(self, dtype, input_shape):
        param_shape = (3,)
        input_cpu, input_supa = create_random_tensor(input_shape, dtype=dtype)
        running_mean_cpu = torch.zeros(param_shape, dtype=dtype)
        running_mean_supa = torch.zeros(param_shape, dtype=dtype).cuda()

        running_var_cpu = torch.ones(param_shape, dtype=dtype)
        running_var_supa = torch.ones(param_shape, dtype=dtype).cuda()

        output_cpu = F.batch_norm(input_cpu, running_mean=running_mean_cpu, running_var=running_var_cpu)
        output_supa = F.batch_norm(input_supa, running_mean=running_mean_supa, running_var=running_var_supa)

        assert_allclose(output_cpu, output_supa, atol=ATOL[dtype], rtol=RTOL[dtype])


    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_batch_norm(self):
        input_shape = [1, 64, 112, 112]
        param_shape = [64]
        training = False
        momentum = 0.10000000000000001
        eps = 1.0000000000000001e-05
        cudnn_enabled = True
        dtype = torch.float32
        input_cpu, input_supa = create_random_tensor(input_shape, dtype=dtype)

        weight = torch.ones(param_shape, dtype=dtype)
        weight_supa = torch.ones(param_shape, dtype=dtype).cuda()

        bias = torch.zeros(param_shape, dtype=dtype)
        bias_supa = torch.zeros(param_shape, dtype=dtype).cuda()

        running_mean = torch.zeros(param_shape, dtype=dtype)
        running_mean_supa = torch.zeros(param_shape, dtype=dtype).cuda()

        running_var = torch.ones(param_shape, dtype=dtype)
        running_var_supa = torch.ones(param_shape, dtype=dtype).cuda()

        output_cpu = torch.ops.aten.batch_norm(
            input_cpu,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            cudnn_enabled,
        )

        output_supa = torch.ops.aten.batch_norm(
            input_supa,
            weight_supa,
            bias_supa,
            running_mean_supa,
            running_var_supa,
            training,
            momentum,
            eps,
            cudnn_enabled,
        )
        assert_allclose(output_cpu, output_supa, atol=ATOL[dtype], rtol=RTOL[dtype])
