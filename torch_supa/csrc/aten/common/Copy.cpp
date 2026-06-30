/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/ops/empty_like.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>
#include "supa_runtime.h"
#include "torch_supa/csrc/aten/core/SUPANativeFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"

#include <ATen/native/cuda/Copy.h>

#include "torch_supa/csrc/core/supa/CachingHostAllocator.h"
#include "torch_supa/csrc/core/supa/PeerToPeerAccess.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAEvent.h"
#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"
#include "torch_supa/csrc/core/supa/TorchVersion.h"

#ifdef TORCH_SUPA_OP_DIR
namespace at::native {
void neg_kernel_cuda(TensorIteratorBase& iter);
void conj_kernel_cuda(TensorIteratorBase& iter);
void neg_conj_kernel_cuda(TensorIteratorBase& iter);
} // namespace at::native
#endif

namespace at::supa {

// device-to-device copy, does type conversion
void copy_device_to_device(TensorIterator& iter, bool non_blocking, bool p2p_enabled) {
  int64_t numel = iter.numel();

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
  bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();
  bool memcpy_eligible = same_type && same_conj && same_neg && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  c10::supa::SUPAGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy. We have to explicitly do this
  // for non-contig copies. This mimics the behavior of cross-device
  // supaMemcpyAsync on the default stream.
  c10::supa::SUPAStream copy_stream = c10::supa::getCurrentSUPAStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    c10::supa::SUPAEvent dst_ready;
    device_guard.set_device(dst_device);
    dst_ready.record(c10::supa::getCurrentSUPAStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready.block(copy_stream);
  }

  if (memcpy_eligible) {
    void* dst = iter.data_ptr(0);
    void* src = iter.data_ptr(1);
    size_t size = numel * iter.element_size(0);
    if (src != dst || src_device != dst_device) {
      // Due to bizarre supa driver intricacies, copies of
      // supaMallocAsynced memory between devices that aren't
      // peer-to-peer-capable need "supaMemcpyPeerAsync".
      // So we let the allocator implement the correct call
      // (either supaMemcpyAsync or supaMemcpyPeerAsync)
      C10_SUPA_CHECK(c10::supa::SUPACachingAllocator::memcpyAsync(
          dst, dst_device.index(), src, src_device.index(), size, copy_stream, p2p_enabled));
    }
  } else {
#ifdef TORCH_SUPA_OP_DIR
    if (same_neg) {
      if (!same_conj) {
        at::native::conj_kernel_cuda(iter);
      } else {
        at::native::direct_copy_kernel_cuda(iter);
      }
    } else {
      if (!same_conj) {
        at::native::neg_conj_kernel_cuda(iter);
      } else {
        at::native::neg_kernel_cuda(iter);
      }
    }
#else
    TORCH_CHECK(false, "torch_supa did not support d2d typecast without torch-supa.");
#endif
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on src_device, record stream event
    c10::supa::SUPAEvent src_ready;
    src_ready.record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready.block(c10::supa::getCurrentSUPAStream(dst_device.index()));
  }

  C10_SUPA_CHECK(supaGetLastError());
}

static bool copy_requires_temporaries(at::Tensor& dst, const at::Tensor& src, bool p2p_enabled) {
  c10::Device dst_device = dst.device();
  c10::Device src_device = src.device();

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GPU.
    TORCH_INTERNAL_ASSERT(dst_device.is_privateuseone() && src_device.is_privateuseone());
    return false;
  }

  bool same_dtype = dst.dtype() == src.dtype();
  if (same_dtype && dst.is_contiguous() && src.is_contiguous()) {
    // Contiguous same-dtype copies can always use supaMemcpyAsync
    return false;
  }
  if (dst_device.is_privateuseone() && src_device.is_privateuseone()) {
    // Copies between GPUs can use the copy kernel if P2P is supported
    return !p2p_enabled;
  }
  // The remaining cases require temporaries. For example, this includes
  // non-contiguous copies between CPU and GPU.
  return true;
}

static bool maybe_enable_p2p_access(c10::Device dst_device, c10::Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }
  return at::supa::get_p2p_access(src_device.index(), dst_device.index());
}

at::Tensor& SUPANativeFunctions::copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  auto dst_device = dst.device();
  auto src_device = src.device();
  if (dst_device.is_privateuseone()) {
    if (dst.scalar_type() == ScalarType::Double) {
      const char* error_msg = "SUPA device does not support double (float64). Please use float32 instead.";
      if (src.is_cpu() && !torch_supa::utils::EnvConfig::IsEnableDtypeDemotion()) {
        TORCH_CHECK(false, error_msg);
      }
      TORCH_WARN(error_msg);
      dst = dst.to(at::kFloat);
      return dst.copy_(src, non_blocking);
    }
  }

  // Enable p2p access between devices. (No-op if it involves the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  if (copy_requires_temporaries(dst, src, p2p_enabled)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    Tensor dst_contig;
    Tensor src_contig;

    // If non_blocking is true - type conversions are performed on the GPU
    // For blocking transfers conversions are performed on CPU to avoid
    // allocating extra GPU memory for GPU-GPU transfers conversions are
    // performed on the source device
    auto conversion_device = non_blocking ? kPrivateUse1 : kCPU;
    if (src.device() == conversion_device) {
      dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = src.to(dst.dtype()).expand_as(dst).contiguous();
    } else {
      bool same_type = dst.dtype() == src.dtype();
      dst_contig =
          (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, src.dtype(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = src.expand_as(dst).contiguous();
    }

    // propagate the correct conjugate bit
    dst_contig._set_conj(dst.is_conj());
    src_contig._set_conj(src.is_conj());

    dst_contig._set_neg(dst.is_neg());
    src_contig._set_neg(src.is_neg());

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return dst;
  }

  if (dst_device.is_privateuseone() && src_device.is_privateuseone()) {
    auto iter = TensorIteratorConfig()
                    .add_output(dst)
#if TORCH_VER >= TORCH_2_3_0
                    .add_const_input(src)
#else
                    .add_input(src)
#endif
                    .resize_outputs(false)
                    .check_all_same_dtype(false)
                    .check_all_same_device(false)
                    .build();
    copy_device_to_device(iter, non_blocking, p2p_enabled);
    ;
    return dst;
  }

  supaMemcpyKind kind = supaMemcpyDefault;
  if (dst_device.is_privateuseone() && src_device.is_cpu()) {
    kind = supaMemcpyHostToDevice;
  } else if (dst_device.is_cpu() && src_device.is_privateuseone()) {
    kind = supaMemcpyDeviceToHost;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  void* dst_ptr = dst.data_ptr();
  void* src_ptr = src.data_ptr();
  auto nbytes = src.numel() * src.dtype().itemsize();
  c10::supa::SUPAStream stream = c10::supa::getCurrentSUPAStream();

  if (non_blocking) {
    C10_SUPA_CHECK(supaMemcpyAsync(dst_ptr, src_ptr, nbytes, kind, stream));
    const auto& host_tensor = (dst_device == kCPU ? dst : src);
    auto* ptr = (dst_device == kCPU ? dst_ptr : src_ptr);
    auto* ctx = host_tensor.storage().data_ptr().get_context();
    // TODO: warn on the return value.
    at::supa::getCachingHostAllocator()->record_event(ptr, ctx, stream.unwrap());
  } else {
    c10::supa::memcpy_and_sync(dst_ptr, src_ptr, nbytes, kind, stream);
  }
  return dst;
}

at::Tensor SUPANativeFunctions::_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  TORCH_CHECK(self.sizes() == dst.sizes(), "_copy_from_and_resize now only support copy with same size");
  TORCH_CHECK(
      self.is_cpu() && dst.device().is_privateuseone(),
      "_copy_from_and_resize now only support copy from cpu tensor to "
      "supa tensor, but got src tensor device is ",
      self.device(),
      " and dst device is ",
      dst.device());
  dst.copy_(self);
  return dst;
}

at::Tensor SUPANativeFunctions::_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  dst.copy_(self, non_blocking);
  return dst;
}

} // namespace at::supa

namespace {

at::Tensor quantized_privateuse1_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(
      self.sizes() == dst.sizes(),
      "_copy_from for QuantizedPrivateUse1 expects source and destination tensors to have the same size");
  TORCH_CHECK(
      ((self.is_cpu() || self.device().is_privateuseone()) && (dst.is_cpu() || dst.device().is_privateuseone())) ||
          (self.device().is_privateuseone() && dst.device().is_privateuseone()),
      "_copy_from for QuantizedPrivateUse1 only supports copies between CPU and SUPA quantized tensors");

  if (dst.is_quantized() && !self.is_quantized()) {
    auto dst_mut = const_cast<at::Tensor&>(dst);
    TORCH_CHECK(self.scalar_type() == at::kFloat, "Quantized copy only works with kFloat as source Tensor");
    TORCH_CHECK(
        (dst.is_contiguous() && self.is_contiguous()) ||
            (dst.is_contiguous(at::MemoryFormat::ChannelsLast) && self.is_contiguous(at::MemoryFormat::ChannelsLast)),
        "Quantized copy only works with contiguous and NHWC Tensors");
    if (dst.qscheme() == at::kPerChannelAffine || dst.qscheme() == at::kPerChannelAffineFloatQParams ||
        dst.qscheme() == at::kPerChannelSymmetric) {
      at::native::quantize_tensor_per_channel_affine(
          self, dst_mut, dst.q_per_channel_scales(), dst.q_per_channel_zero_points(), dst.q_per_channel_axis());
    } else {
      at::native::quantize_tensor_per_tensor_affine(self, dst_mut, dst.q_scale(), dst.q_zero_point());
    }
    return dst;
  }

  TORCH_CHECK(
      dst.is_quantized() && self.is_quantized(),
      "Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor");
  TORCH_CHECK(dst.qscheme() == self.qscheme(), "Quantized Copy only works with same qscheme");
  TORCH_CHECK(dst.scalar_type() == self.scalar_type(), "Quantized Copy only works with same dtype");
  at::set_quantizer_(dst, self.quantizer());

  auto dst_mut = const_cast<at::Tensor&>(dst);
  at::supa::SUPANativeFunctions::copy_(dst_mut, self, non_blocking);
  return dst;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("_copy_from", TORCH_FN(quantized_privateuse1_copy_from));
}

} // namespace
