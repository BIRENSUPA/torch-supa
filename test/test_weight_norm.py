# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import numpy as np
import pytest
import torch
import torch.nn as nn

from torch_supa.testing.common_utils import assert_allclose

dtype2prec_DONTUSE = {torch.float: 1e-5, torch.bfloat16: 1e-1}


class TestWeightNorm:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_weight_norm(self):
        for dtype in [torch.float, torch.bfloat16]:
            input_cpu = torch.randn(1, 2, 3, 4, dtype=dtype)
            input_supa = input_cpu.clone().supa()
            m_cpu = nn.BatchNorm2d(2).to(dtype=dtype)
            m_supa = nn.BatchNorm2d(2).to(dtype=dtype).supa()
            expected_output_cpu = m_cpu(input_cpu)
            expected_output_supa = m_supa(input_supa)

            # add weight normalization
            m_cpu = torch.nn.utils.weight_norm(m_cpu)
            np.testing.assert_equal(m_cpu.weight_v.size(), m_cpu.weight.size())
            np.testing.assert_equal(m_cpu.weight_g.size(), (2,))
            assert_allclose(
                m_cpu(input_cpu),
                expected_output_cpu,
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )

            m_supa = torch.nn.utils.weight_norm(m_supa)
            np.testing.assert_equal(m_supa.weight_v.size(), m_supa.weight.size())
            np.testing.assert_equal(m_supa.weight_g.size(), (2,))
            assert_allclose(
                m_supa(input_supa),
                expected_output_supa,
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
