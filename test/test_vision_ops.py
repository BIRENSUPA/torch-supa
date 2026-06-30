# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

import torch
from torch_supa.testing.common_utils import assert_allclose
from torchvision import ops
import pytest

class TestNMS:
    def _create_tensors_with_iou(self, N, iou_thresh):
        # force last box to have a pre-defined iou with the first box
        # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
        # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
        # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
        # Adjust the threshold upward a bit with the intent of creating
        # at least one box that exceeds (barely) the threshold and so
        # should be suppressed.
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        boxes[-1, :] = boxes[0, :]
        x0, y0, x1, y1 = boxes[-1].tolist()
        iou_thresh += 1e-5
        boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
        scores = torch.rand(N)
        return boxes, scores


    def test_nms_input_errors(self):
        with pytest.raises(RuntimeError):
            ops.nms(torch.rand(4), torch.rand(3), 0.5)
        with pytest.raises(RuntimeError):
            ops.nms(torch.rand(3, 5), torch.rand(3), 0.5)
        with pytest.raises(RuntimeError):
            ops.nms(torch.rand(3, 4), torch.rand(3, 2), 0.5)
        with pytest.raises(RuntimeError):
            ops.nms(torch.rand(3, 4), torch.rand(4), 0.5)

    @pytest.mark.sanity
    @pytest.mark.gcuSmoke
    @pytest.mark.regression
    @pytest.mark.gcuSanity
    @pytest.mark.gcuStress
    @pytest.mark.parametrize("iou", (0.2, 0.5, 0.8))
    def test_nms_gpu(self, iou, dtype=torch.float32):
        device = "supa"
        dtype = torch.float32 if device == "mps" else dtype
        tol = 1e-3 if dtype is torch.half else 1e-5
        err_msg = "NMS incompatible between CPU and SUPA for IoU={}"

        boxes, scores = self._create_tensors_with_iou(1000, iou)
        r_cpu = ops.nms(boxes, scores, iou)
        r_gpu = ops.nms(boxes.to(device), scores.to(device), iou)

        is_eq = torch.allclose(r_cpu, r_gpu.cpu())
        if not is_eq:
            # if the indices are not the same, ensure that it's because the scores
            # are duplicate
            is_eq = torch.allclose(scores[r_cpu], scores[r_gpu.cpu()], rtol=tol, atol=tol)
        assert is_eq, err_msg.format(iou)


    def test_nms_float16(self):
        device = "supa"
        boxes = torch.tensor(
            [
                [285.3538, 185.5758, 1193.5110, 851.4551],
                [285.1472, 188.7374, 1192.4984, 851.0669],
                [279.2440, 197.9812, 1189.4746, 849.2019],
            ]
        ).to(device)
        scores = torch.tensor([0.6370, 0.7569, 0.3966]).to(device)

        iou_thres = 0.2
        keep32 = ops.nms(boxes, scores, iou_thres)
        keep16 = ops.nms(boxes.to(torch.float16), scores.to(torch.float16), iou_thres)
        assert_allclose(keep32, keep16, atol=0, rtol=0)

    @pytest.mark.parametrize("seed", range(10))
    def test_batched_nms_implementations(self, seed):
        """Make sure that both implementations of batched_nms yield identical results"""
        torch.random.manual_seed(seed)

        num_boxes = 1000
        iou_threshold = 0.9

        boxes = torch.cat((torch.rand(num_boxes, 2), torch.rand(num_boxes, 2) + 10), dim=1)
        assert max(boxes[:, 0]) < min(boxes[:, 2])  # x1 < x2
        assert max(boxes[:, 1]) < min(boxes[:, 3])  # y1 < y2

        scores = torch.rand(num_boxes)
        idxs = torch.randint(0, 4, size=(num_boxes,))
        keep_vanilla = ops.boxes._batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
        keep_trick = ops.boxes._batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)

        torch.testing.assert_close(
            keep_vanilla, keep_trick, msg="The vanilla and the trick implementation yield different nms outputs."
        )

        # Also make sure an empty tensor is returned if boxes is empty
        empty = torch.empty((0,), dtype=torch.int64)
        torch.testing.assert_close(empty, ops.batched_nms(empty, None, None, None))


class TestROIAlign:
    @pytest.mark.sanity
    @pytest.mark.regression
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
    def test_roi_align_basic(self, dtype):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors on CPU
        input_cpu = torch.arange(25, dtype=dtype, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=dtype, device="cpu")

        # Create input tensors on CUDA
        input_supa = torch.arange(25, dtype=dtype, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=dtype, device="supa")

        # Apply roi_align on both devices
        output_cpu = ops.roi_align(input_cpu, rois_cpu, output_size=(2, 2))
        output_supa = ops.roi_align(input_supa, rois_supa, output_size=(2, 2))

        # Compare results
        assert_allclose(output_cpu, output_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_roi_align_multiple_rois(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors on CPU
        input_cpu = torch.arange(25, dtype=torch.float32, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 1, 1, 3, 3]
        ], dtype=torch.float32, device="cpu")

        # Create input tensors on CUDA
        input_supa = torch.arange(25, dtype=torch.float32, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 1, 1, 3, 3]
        ], dtype=torch.float32, device="supa")

        # Apply roi_align on both devices
        output_cpu = ops.roi_align(input_cpu, rois_cpu, output_size=(2, 2))
        output_supa = ops.roi_align(input_supa, rois_supa, output_size=(2, 2))

        # Compare results
        assert_allclose(output_cpu, output_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_roi_align_different_output_sizes(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors on CPU
        input_cpu = torch.arange(25, dtype=torch.float32, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cpu")

        # Create input tensors on CUDA
        input_supa = torch.arange(25, dtype=torch.float32, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="supa")

        # Test different output sizes on both devices
        output_1x1_cpu = ops.roi_align(input_cpu, rois_cpu, output_size=(1, 1))
        output_1x1_supa = ops.roi_align(input_supa, rois_supa, output_size=(1, 1))
        assert output_1x1_cpu.shape == (1, 1, 1, 1)
        assert output_1x1_supa.shape == (1, 1, 1, 1)
        assert_allclose(output_1x1_cpu, output_1x1_supa.cpu(), atol=1e-3, rtol=1e-3)

        output_3x3_cpu = ops.roi_align(input_cpu, rois_cpu, output_size=(3, 3))
        output_3x3_supa = ops.roi_align(input_supa, rois_supa, output_size=(3, 3))
        assert output_3x3_cpu.shape == (1, 1, 3, 3)
        assert output_3x3_supa.shape == (1, 1, 3, 3)
        assert_allclose(output_3x3_cpu, output_3x3_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_roi_align_spatial_scale(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors on CPU
        input_cpu = torch.arange(25, dtype=torch.float32, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cpu")

        # Create input tensors on CUDA
        input_supa = torch.arange(25, dtype=torch.float32, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="supa")

        # Apply roi_align with different spatial scales on both devices
        output_default_cpu = ops.roi_align(input_cpu, rois_cpu, output_size=(2, 2))
        output_scaled_cpu = ops.roi_align(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=0.5)

        output_default_supa = ops.roi_align(input_supa, rois_supa, output_size=(2, 2))
        output_scaled_supa = ops.roi_align(input_supa, rois_supa, output_size=(2, 2), spatial_scale=0.5)

        # Check that outputs are different on both devices
        assert output_default_cpu.shape == output_scaled_cpu.shape
        assert output_default_supa.shape == output_scaled_supa.shape
        assert not torch.allclose(output_default_cpu, output_scaled_cpu, atol=1e-5)
        assert not torch.allclose(output_default_supa, output_scaled_supa, atol=1e-5)

        # Compare CPU and CUDA results
        assert_allclose(output_default_cpu, output_default_supa.cpu(), atol=1e-3, rtol=1e-3)
        assert_allclose(output_scaled_cpu, output_scaled_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_roi_align_sampling_ratio(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors on CPU
        input_cpu = torch.arange(25, dtype=torch.float32, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cpu")

        # Create input tensors on CUDA
        input_supa = torch.arange(25, dtype=torch.float32, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="supa")

        # Apply roi_align with different sampling ratios on both devices
        output_default_cpu = ops.roi_align(input_cpu, rois_cpu, output_size=(2, 2))
        output_ratio_2_cpu = ops.roi_align(input_cpu, rois_cpu, output_size=(2, 2), sampling_ratio=2)

        output_default_supa = ops.roi_align(input_supa, rois_supa, output_size=(2, 2))
        output_ratio_2_supa = ops.roi_align(input_supa, rois_supa, output_size=(2, 2), sampling_ratio=2)

        # Check that outputs have the same shape on both devices
        assert output_default_cpu.shape == output_ratio_2_cpu.shape
        assert output_default_supa.shape == output_ratio_2_supa.shape

        # Compare CPU and CUDA results
        assert_allclose(output_default_cpu, output_default_supa.cpu(), atol=1e-3, rtol=1e-3)
        assert_allclose(output_ratio_2_cpu, output_ratio_2_supa.cpu(), atol=1e-3, rtol=1e-3)


class TestROIPool:
    @pytest.mark.sanity
    @pytest.mark.regression
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
    def test_roi_pool_basic(self, dtype):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors on CPU
        input_cpu = torch.arange(25, dtype=dtype, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=dtype, device="cpu")

        # Create input tensors on CUDA
        input_supa = torch.arange(25, dtype=dtype, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=dtype, device="supa")

        # Apply roi_pool on both devices
        output_cpu = ops.roi_pool(input_cpu, rois_cpu, output_size=(2, 2))
        output_supa = ops.roi_pool(input_supa, rois_supa, output_size=(2, 2))

        # Compare results
        assert_allclose(output_cpu, output_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_roi_pool_multiple_rois(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors on CPU
        input_cpu = torch.arange(25, dtype=torch.float32, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 1, 1, 3, 3]
        ], dtype=torch.float32, device="cpu")

        # Create input tensors on CUDA
        input_supa = torch.arange(25, dtype=torch.float32, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 1, 1, 3, 3]
        ], dtype=torch.float32, device="supa")

        # Apply roi_pool on both devices
        output_cpu = ops.roi_pool(input_cpu, rois_cpu, output_size=(2, 2))
        output_supa = ops.roi_pool(input_supa, rois_supa, output_size=(2, 2))

        # Compare results
        assert_allclose(output_cpu, output_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_roi_pool_different_output_sizes(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors on CPU
        input_cpu = torch.arange(25, dtype=torch.float32, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cpu")

        # Create input tensors on CUDA
        input_supa = torch.arange(25, dtype=torch.float32, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="supa")

        # Test different output sizes on both devices
        output_1x1_cpu = ops.roi_pool(input_cpu, rois_cpu, output_size=(1, 1))
        output_1x1_supa = ops.roi_pool(input_supa, rois_supa, output_size=(1, 1))
        assert output_1x1_cpu.shape == (1, 1, 1, 1)
        assert output_1x1_supa.shape == (1, 1, 1, 1)
        assert_allclose(output_1x1_cpu, output_1x1_supa.cpu(), atol=1e-3, rtol=1e-3)

        output_3x3_cpu = ops.roi_pool(input_cpu, rois_cpu, output_size=(3, 3))
        output_3x3_supa = ops.roi_pool(input_supa, rois_supa, output_size=(3, 3))
        assert output_3x3_cpu.shape == (1, 1, 3, 3)
        assert output_3x3_supa.shape == (1, 1, 3, 3)
        assert_allclose(output_3x3_cpu, output_3x3_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_roi_pool_spatial_scale(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors on CPU
        input_cpu = torch.arange(25, dtype=torch.float32, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cpu")

        # Create input tensors on CUDA
        input_supa = torch.arange(25, dtype=torch.float32, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="supa")

        # Apply roi_pool with different spatial scales on both devices
        output_default_cpu = ops.roi_pool(input_cpu, rois_cpu, output_size=(2, 2))
        output_scaled_cpu = ops.roi_pool(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=0.5)

        output_default_supa = ops.roi_pool(input_supa, rois_supa, output_size=(2, 2))
        output_scaled_supa = ops.roi_pool(input_supa, rois_supa, output_size=(2, 2), spatial_scale=0.5)

        # Check that outputs are different on both devices
        assert output_default_cpu.shape == output_scaled_cpu.shape
        assert output_default_supa.shape == output_scaled_supa.shape
        assert not torch.allclose(output_default_cpu, output_scaled_cpu, atol=1e-5)
        assert not torch.allclose(output_default_supa, output_scaled_supa, atol=1e-5)

        # Compare CPU and CUDA results
        assert_allclose(output_default_cpu, output_default_supa.cpu(), atol=1e-3, rtol=1e-3)
        assert_allclose(output_scaled_cpu, output_scaled_supa.cpu(), atol=1e-3, rtol=1e-3)


class TestPSROIAlign:
    @pytest.mark.sanity
    @pytest.mark.regression
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
    def test_ps_roi_align_basic(self, dtype):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors with 4 channels (multiple of 2*2) on CPU
        input_cpu = torch.arange(100, dtype=dtype, device="cpu").reshape(1, 4, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=dtype, device="cpu")

        # Create input tensors with 4 channels (multiple of 2*2) on CUDA
        input_supa = torch.arange(100, dtype=dtype, device="supa").reshape(1, 4, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=dtype, device="supa")

        # Apply ps_roi_align on both devices
        output_cpu = ops.ps_roi_align(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=1.0, sampling_ratio=-1)
        output_supa = ops.ps_roi_align(input_supa, rois_supa, output_size=(2, 2), spatial_scale=1.0, sampling_ratio=-1)

        # Compare results
        assert_allclose(output_cpu, output_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_ps_roi_align_multiple_rois(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors with 4 channels (multiple of 2*2) on CPU
        input_cpu = torch.arange(100, dtype=torch.float32, device="cpu").reshape(1, 4, 5, 5)
        rois_cpu = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 1, 1, 3, 3]
        ], dtype=torch.float32, device="cpu")

        # Create input tensors with 4 channels (multiple of 2*2) on CUDA
        input_supa = torch.arange(100, dtype=torch.float32, device="supa").reshape(1, 4, 5, 5)
        rois_supa = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 1, 1, 3, 3]
        ], dtype=torch.float32, device="supa")

        # Apply ps_roi_align on both devices
        output_cpu = ops.ps_roi_align(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=1.0, sampling_ratio=-1)
        output_supa = ops.ps_roi_align(input_supa, rois_supa, output_size=(2, 2), spatial_scale=1.0, sampling_ratio=-1)

        # Compare results
        assert_allclose(output_cpu, output_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_ps_roi_align_different_output_sizes(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # For different output sizes, we need to adjust the number of input channels
        # For 1x1 output, we need 1 input channel (1*1)
        # Create input tensors on CPU
        input_1x1_cpu = torch.arange(25, dtype=torch.float32, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cpu")

        # Create input tensors on CUDA
        input_1x1_supa = torch.arange(25, dtype=torch.float32, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="supa")

        output_1x1_cpu = ops.ps_roi_align(input_1x1_cpu, rois_cpu, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=-1)
        output_1x1_supa = ops.ps_roi_align(input_1x1_supa, rois_supa, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=-1)
        assert output_1x1_cpu.shape == (1, 1, 1, 1)
        assert output_1x1_supa.shape == (1, 1, 1, 1)
        assert_allclose(output_1x1_cpu, output_1x1_supa.cpu(), atol=1e-3, rtol=1e-3)

        # For 3x3 output, we need 9 input channels (3*3)
        # Create input tensors on CPU
        input_3x3_cpu = torch.arange(225, dtype=torch.float32, device="cpu").reshape(1, 9, 5, 5)

        # Create input tensors on CUDA
        input_3x3_supa = torch.arange(225, dtype=torch.float32, device="supa").reshape(1, 9, 5, 5)

        output_3x3_cpu = ops.ps_roi_align(input_3x3_cpu, rois_cpu, output_size=(3, 3), spatial_scale=1.0, sampling_ratio=-1)
        output_3x3_supa = ops.ps_roi_align(input_3x3_supa, rois_supa, output_size=(3, 3), spatial_scale=1.0, sampling_ratio=-1)
        assert output_3x3_cpu.shape == (1, 1, 3, 3)
        assert output_3x3_supa.shape == (1, 1, 3, 3)
        assert_allclose(output_3x3_cpu, output_3x3_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_ps_roi_align_spatial_scale(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors with 4 channels (multiple of 2*2) on CPU
        input_cpu = torch.arange(100, dtype=torch.float32, device="cpu").reshape(1, 4, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cpu")

        # Create input tensors with 4 channels (multiple of 2*2) on CUDA
        input_supa = torch.arange(100, dtype=torch.float32, device="supa").reshape(1, 4, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="supa")

        # Apply ps_roi_align with different spatial scales on both devices
        output_default_cpu = ops.ps_roi_align(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=1.0, sampling_ratio=-1)
        output_scaled_cpu = ops.ps_roi_align(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=0.5, sampling_ratio=-1)

        output_default_supa = ops.ps_roi_align(input_supa, rois_supa, output_size=(2, 2), spatial_scale=1.0, sampling_ratio=-1)
        output_scaled_supa = ops.ps_roi_align(input_supa, rois_supa, output_size=(2, 2), spatial_scale=0.5, sampling_ratio=-1)

        # Check that outputs are different on both devices
        assert output_default_cpu.shape == output_scaled_cpu.shape
        assert output_default_supa.shape == output_scaled_supa.shape
        assert not torch.allclose(output_default_cpu, output_scaled_cpu, atol=1e-5)
        assert not torch.allclose(output_default_supa, output_scaled_supa, atol=1e-5)

        # Compare CPU and CUDA results
        assert_allclose(output_default_cpu, output_default_supa.cpu(), atol=1e-3, rtol=1e-3)
        assert_allclose(output_scaled_cpu, output_scaled_supa.cpu(), atol=1e-3, rtol=1e-3)


class TestPSROIPool:
    @pytest.mark.sanity
    @pytest.mark.regression
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
    def test_ps_roi_pool_basic(self, dtype):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors with 4 channels (multiple of 2*2) on CPU
        input_cpu = torch.arange(100, dtype=dtype, device="cpu").reshape(1, 4, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=dtype, device="cpu")

        # Create input tensors with 4 channels (multiple of 2*2) on CUDA
        input_supa = torch.arange(100, dtype=dtype, device="supa").reshape(1, 4, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=dtype, device="supa")

        # Apply ps_roi_pool on both devices
        output_cpu = ops.ps_roi_pool(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=1.0)
        output_supa = ops.ps_roi_pool(input_supa, rois_supa, output_size=(2, 2), spatial_scale=1.0)

        # Compare results
        assert_allclose(output_cpu, output_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_ps_roi_pool_multiple_rois(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors with 4 channels (multiple of 2*2) on CPU
        input_cpu = torch.arange(100, dtype=torch.float32, device="cpu").reshape(1, 4, 5, 5)
        rois_cpu = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 1, 1, 3, 3]
        ], dtype=torch.float32, device="cpu")

        # Create input tensors with 4 channels (multiple of 2*2) on CUDA
        input_supa = torch.arange(100, dtype=torch.float32, device="supa").reshape(1, 4, 5, 5)
        rois_supa = torch.tensor([
            [0, 0, 0, 4, 4],
            [0, 1, 1, 3, 3]
        ], dtype=torch.float32, device="supa")

        # Apply ps_roi_pool on both devices
        output_cpu = ops.ps_roi_pool(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=1.0)
        output_supa = ops.ps_roi_pool(input_supa, rois_supa, output_size=(2, 2), spatial_scale=1.0)

        # Compare results
        assert_allclose(output_cpu, output_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_ps_roi_pool_different_output_sizes(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # For different output sizes, we need to adjust the number of input channels
        # For 1x1 output, we need 1 input channel (1*1)
        # Create input tensors on CPU
        input_1x1_cpu = torch.arange(25, dtype=torch.float32, device="cpu").reshape(1, 1, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cpu")

        # Create input tensors on CUDA
        input_1x1_supa = torch.arange(25, dtype=torch.float32, device="supa").reshape(1, 1, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="supa")

        output_1x1_cpu = ops.ps_roi_pool(input_1x1_cpu, rois_cpu, output_size=(1, 1), spatial_scale=1.0)
        output_1x1_supa = ops.ps_roi_pool(input_1x1_supa, rois_supa, output_size=(1, 1), spatial_scale=1.0)
        assert output_1x1_cpu.shape == (1, 1, 1, 1)
        assert output_1x1_supa.shape == (1, 1, 1, 1)
        assert_allclose(output_1x1_cpu, output_1x1_supa.cpu(), atol=1e-3, rtol=1e-3)

        # For 3x3 output, we need 9 input channels (3*3)
        # Create input tensors on CPU
        input_3x3_cpu = torch.arange(225, dtype=torch.float32, device="cpu").reshape(1, 9, 5, 5)

        # Create input tensors on CUDA
        input_3x3_supa = torch.arange(225, dtype=torch.float32, device="supa").reshape(1, 9, 5, 5)

        output_3x3_cpu = ops.ps_roi_pool(input_3x3_cpu, rois_cpu, output_size=(3, 3), spatial_scale=1.0)
        output_3x3_supa = ops.ps_roi_pool(input_3x3_supa, rois_supa, output_size=(3, 3), spatial_scale=1.0)
        assert output_3x3_cpu.shape == (1, 1, 3, 3)
        assert output_3x3_supa.shape == (1, 1, 3, 3)
        assert_allclose(output_3x3_cpu, output_3x3_supa.cpu(), atol=1e-3, rtol=1e-3)

    @pytest.mark.sanity
    @pytest.mark.regression
    def test_ps_roi_pool_spatial_scale(self):
        if not torch.supa.is_available():
            pytest.skip("SUPA not available")

        # Create input tensors with 4 channels (multiple of 2*2) on CPU
        input_cpu = torch.arange(100, dtype=torch.float32, device="cpu").reshape(1, 4, 5, 5)
        rois_cpu = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cpu")

        # Create input tensors with 4 channels (multiple of 2*2) on CUDA
        input_supa = torch.arange(100, dtype=torch.float32, device="supa").reshape(1, 4, 5, 5)
        rois_supa = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="supa")

        # Apply ps_roi_pool with different spatial scales on both devices
        output_default_cpu = ops.ps_roi_pool(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=1.0)
        output_scaled_cpu = ops.ps_roi_pool(input_cpu, rois_cpu, output_size=(2, 2), spatial_scale=0.5)

        output_default_supa = ops.ps_roi_pool(input_supa, rois_supa, output_size=(2, 2), spatial_scale=1.0)
        output_scaled_supa = ops.ps_roi_pool(input_supa, rois_supa, output_size=(2, 2), spatial_scale=0.5)

        # Check that outputs are different on both devices
        assert output_default_cpu.shape == output_scaled_cpu.shape
        assert output_default_supa.shape == output_scaled_supa.shape
        assert not torch.allclose(output_default_cpu, output_scaled_cpu, atol=1e-5)
        assert not torch.allclose(output_default_supa, output_scaled_supa, atol=1e-5)

        # Compare CPU and CUDA results
        assert_allclose(output_default_cpu, output_default_supa.cpu(), atol=1e-3, rtol=1e-3)
        assert_allclose(output_scaled_cpu, output_scaled_supa.cpu(), atol=1e-3, rtol=1e-3)
