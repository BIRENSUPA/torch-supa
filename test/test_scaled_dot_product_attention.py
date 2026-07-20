# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import pytest
import torch
import torch_supa

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_supa.testing.common_utils import (
    assert_allclose,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def generator(batch, q_len, kv_len, num_heads, head_size, num_heads_kv, head_size_v, dtype=torch.float16):
    torch.manual_seed(0)
    query = torch.randn(batch, q_len, num_heads, head_size, requires_grad=True, dtype=torch.float).to(dtype)
    key = torch.randn(batch, kv_len, num_heads_kv, head_size, requires_grad=True, dtype=torch.float).to(dtype)
    value = torch.randn(batch, kv_len, num_heads_kv, head_size_v, requires_grad=True, dtype=torch.float).to(dtype)
    dO = torch.randn(batch, q_len, num_heads, head_size_v, requires_grad=True, dtype=torch.float).to(dtype)
    return query, key, value, dO

class TestSDPAMethod:
    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_sdpa(self):
        torch_supa
        torch.nn

        query, key, value = (
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
        )
        query_device, key_deivce, value_device = (
            query.clone().to(device),
            key.clone().to(device),
            value.clone().to(device),
        )
        out = F.scaled_dot_product_attention(query, key, value)
        out_device = F.scaled_dot_product_attention(
            query_device, key_deivce, value_device
        )
        assert_allclose(out, out_device, atol=1e-2, rtol=0.016, equal_nan=True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_attention(self):
        query, key, value = (
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
        )
        query_device, key_deivce, value_device = (
            query.clone().to(device),
            key.clone().to(device),
            value.clone().to(device),
        )
        out = F.scaled_dot_product_attention(query, key, value)
        assert torch.backends.cuda.is_flash_attention_available() is True
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            assert torch._fused_sdp_choice(query_device, key_deivce, value_device) == SDPBackend.FLASH_ATTENTION.value
            flash_ref = F.scaled_dot_product_attention(query_device, key_deivce, value_device)
        assert_allclose(out, flash_ref, atol=1e-2, rtol=0.016, equal_nan=True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_attention_headdim512(self):
        query, key, value = (
            torch.randn(1, 1, 128, 512, dtype=torch.float16),
            torch.randn(1, 1, 256, 512, dtype=torch.float16),
            torch.randn(1, 1, 256, 512, dtype=torch.float16),
        )
        query_device, key_deivce, value_device = (
            query.clone().to(device),
            key.clone().to(device),
            value.clone().to(device),
        )
        out = F.scaled_dot_product_attention(query, key, value)
        assert torch.backends.cuda.is_flash_attention_available() is True
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            assert torch._fused_sdp_choice(query_device, key_deivce, value_device) == SDPBackend.FLASH_ATTENTION.value
            flash_ref = F.scaled_dot_product_attention(query_device, key_deivce, value_device)
        assert_allclose(out, flash_ref, atol=1e-2, rtol=0.016, equal_nan=True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize(
        "batch_size, q_len, kv_len, num_heads, head_size, num_heads_kv, head_size_v, is_causal, window_left, window_right",
        [
            (1, 128, 128, 1, 128, 1, 128, False, -1, -1)
        ],
    )
    def test_attention_bwd(self, batch_size, q_len, kv_len, num_heads, head_size, num_heads_kv, head_size_v, is_causal, window_left, window_right):
        query, key, value, dO = generator(
            batch_size,
            q_len,
            kv_len,
            num_heads,
            head_size,
            num_heads_kv,
            head_size_v
        )
        attn_bias = None
        cu_seqlens_q, cu_seqlens_k = None, None
        O = torch.zeros_like(dO)
        L = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(query, key, value, 0.0, False, scale=None)[1]
        golden = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(dO, query, key, value, O,
                                                                                     L, dropout_p=0.0, is_causal=is_causal, attn_mask=None, scale=None)
        query, key, value, dO = query.cuda(), key.cuda(), value.cuda(), dO.cuda()
        O, L = O.cuda(), L.cuda()
        dq, dk, dv = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

        d_query, d_key, d_value = torch.ops.aten._scaled_dot_product_flash_attention_backward(dO, query, key, value, O,
                                                                                              L, cum_seq_q=cu_seqlens_q, cum_seq_k=cu_seqlens_k, max_q=0, max_k=0, dropout_p=0.0, is_causal=False, philox_seed=None, philox_offset=None, scale=None)
        assert_allclose(d_query.cpu(), golden[0], atol=1e-2, rtol=0.016, equal_nan=True)
        assert_allclose(d_key.cpu(), golden[1], atol=1e-2, rtol=0.016, equal_nan=True)
        assert_allclose(d_value.cpu(), golden[2], atol=1e-2, rtol=0.016, equal_nan=True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_efficient_attention(self):
        query, key, value = (
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
        )
        query_device, key_deivce, value_device = (
            query.clone().to(device),
            key.clone().to(device),
            value.clone().to(device),
        )
        out = F.scaled_dot_product_attention(query, key, value)
        flash_ref = torch.ops.aten._scaled_dot_product_efficient_attention(query_device, key_deivce, value_device, compute_log_sumexp=False, attn_bias=None, scale=None)
        assert_allclose(out, flash_ref[0], atol=1e-2, rtol=0.016, equal_nan=True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize(
        "batch_size, q_len, kv_len, num_heads, head_size, num_heads_kv, head_size_v, is_causal, window_left, window_right",
        [
            (1, 128, 128, 1, 128, 1, 128, False, -1, -1)
        ],
    )
    def test_efficient_attention_bwd(self, batch_size, q_len, kv_len, num_heads, head_size, num_heads_kv, head_size_v, is_causal, window_left, window_right):
        query, key, value, dO = generator(
            batch_size,
            q_len,
            kv_len,
            num_heads,
            head_size,
            num_heads_kv,
            head_size_v
        )
        attn_bias = None
        cu_seqlens_q, cu_seqlens_k = None, None
        O = torch.zeros_like(dO)
        L = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(query, key, value, 0.0, False, scale=None)[1]
        golden = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(dO, query, key, value, O,
                                                                                     L, dropout_p=0.0, is_causal=False, attn_mask=None, scale=None)
        query, key, value, dO = query.cuda(), key.cuda(), value.cuda(), dO.cuda()
        O, L = O.cuda(), L.cuda()
        dq, dk, dv = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

        grad_input_mask = [True, False, True, False]
        d_query, d_key, d_value, d_bias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(dO, query, key, value, attn_bias, O,
                                                                                                          L, philox_seed=None, philox_offset=None, dropout_p=0.0, grad_input_mask=grad_input_mask, is_causal=False, scale=None)
        assert_allclose(d_query.cpu(), golden[0], atol=1e-2, rtol=0.016, equal_nan=True)
        assert_allclose(d_key.cpu(), golden[1], atol=1e-2, rtol=0.016, equal_nan=True)
        assert_allclose(d_value.cpu(), golden[2], atol=1e-2, rtol=0.016, equal_nan=True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    def test_cudnn_attention(self):
        query, key, value = (
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
            torch.randn(1, 128, 1, 64, dtype=torch.float16),
        )
        query_device, key_deivce, value_device = (
            query.clone().to(device),
            key.clone().to(device),
            value.clone().to(device),
        )
        out = F.scaled_dot_product_attention(query, key, value)
        out_supa = torch.ops.aten._scaled_dot_product_cudnn_attention(
            query_device, key_deivce, value_device,
            attn_bias=None, compute_log_sumexp=False, dropout_p=0.0, is_causal=False, return_debug_mask=False, scale=None
        )
        assert_allclose(out, out_supa[0], atol=1e-2, rtol=0.016, equal_nan=True)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize(
        "batch_size, num_heads, q_len, kv_len, head_size",
        [
            (1, 1, 128, 128, 64)
        ],
    )
    def test_cudnn_attention_bwd(self, batch_size, num_heads, q_len, kv_len, head_size):
        shape_q = (batch_size, q_len, num_heads, head_size)
        shape_kv = (batch_size, kv_len, num_heads, head_size)
        query = torch.randn(*shape_q, requires_grad=True, dtype=torch.float16)
        key = torch.randn(*shape_kv, requires_grad=True, dtype=torch.float16)
        value = torch.randn(*shape_kv, requires_grad=True, dtype=torch.float16)
        dO = torch.randn(*shape_q, dtype=torch.float16)

        query_device = query.clone().to(device)
        key_device = key.clone().to(device)
        value_device = value.clone().to(device)
        dO_device = dO.clone().to(device)

        out_cpu = F.scaled_dot_product_attention(query, key, value)
        out_cpu.backward(dO)

        out_device = torch.ops.aten._scaled_dot_product_cudnn_attention(
            query_device, key_device, value_device,
            attn_bias=None, compute_log_sumexp=True, dropout_p=0.0, is_causal=False, return_debug_mask=False, scale=None
        )
        # out_device =  attention, log_sumexp, Tensor(), Tensor(), max_seqlen_batch_q, 
        # max_seqlen_batch_kv, sudnn_seed, sudnn_offset, Tensor();

        d_query, d_key, d_value = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
            dO_device, query_device, key_device, value_device,
            out_device[0], out_device[1], out_device[6], out_device[7],
            attn_bias=None, cum_seq_q=None, cum_seq_k=None,
            max_q=q_len, max_k=kv_len, dropout_p=0.0, is_causal=False, scale=None
        )

        assert_allclose(d_query.cpu(), query.grad, atol=1e-2, rtol=0.016, equal_nan=True)
        assert_allclose(d_key.cpu(), key.grad, atol=1e-2, rtol=0.016, equal_nan=True)
        assert_allclose(d_value.cpu(), value.grad, atol=1e-2, rtol=0.016, equal_nan=True)
