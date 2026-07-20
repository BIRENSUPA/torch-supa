/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <algorithm>
#include <chrono>
#include <future>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <ATen/native/ConvUtils.h>
#include "torch_supa/csrc/aten/SUPAGeneratorImpl.h"
#include "torch_supa/csrc/aten/common/Sleep.h"
#include "torch_supa/csrc/aten/ops/kernels/transformers/SdpUtils.h"
#include "torch_supa/csrc/core/supa/CachingHostAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAAllocatorConfig.h"
#include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
#include "torch_supa/csrc/core/supa/SUPAContext.h"
#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAGraphsUtils.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"
#include "torch_supa/csrc/core/supa/SublasContext.h"
#include "torch_supa/csrc/core/supa/TorchVersion.h"
#include "torch_supa/csrc/ipc/SupaIPCTypes.h"
#include "torch_supa/csrc/supa/Module.h"
#include "torch_supa/csrc/supa/SUPAPluggableAllocator.h"
#include "torch_supa/csrc/supa/memory_snapshot.h"
#include "torch_supa/csrc/supa/utils.h"
#include "torch_supa/csrc/utils/LazyInit.h"

#ifdef USE_BCCL
#include "torch_supa/csrc/supa/python_bccl.h"
#endif

#if TORCH_VER >= TORCH_2_2_0
#include "torch_supa/csrc/supa/CombinedTraceback.h"
#endif

#include <ATen/ATen.h>
#include <ATen/CachedTensorUtils.h>
#include <ATen/Context.h>
#include <ATen/WrapDimUtils.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_flatten.h>
#include "torch_supa/csrc/core/supa/SUPAVersion.h"

static bool in_bad_fork = false; // True for children forked after xpu init

namespace {

struct SupaGatherInfo {
  std::vector<int64_t> expected_size;
  int64_t dim;
};

using tensor_list2d = std::vector<std::vector<at::Tensor>>;

struct UniqueTypeChecker {
  void show(size_t type_id) {
    if (!unique) {
      return;
    }
    if (!type_id_) {
      type_id_ = type_id;
    }
    unique = type_id_.value() == type_id;
  }

  std::optional<size_t> type_id_;
  bool unique = true;
};

at::Device supa_device_from_index(int64_t device) {
  TORCH_CHECK(device >= 0, "Expected non-negative device index, but got ", device);
  return at::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(device));
}

std::vector<at::Tensor>& supa_broadcast_out(const at::Tensor& tensor, std::vector<at::Tensor>& out_tensors) {
  for (const auto i : c10::irange(out_tensors.size())) {
    TORCH_CHECK(
        out_tensors[i].device().type() == c10::DeviceType::PrivateUse1,
        "Expected all output tensors to be SUPA tensors, but output tensor at index ",
        i,
        " has device '",
        out_tensors[i].device(),
        "'");
    TORCH_CHECK(
        out_tensors[i].sizes() == tensor.sizes(),
        "Expected all output tensors to have same shape as the source tensor ",
        tensor.sizes(),
        ", but output tensor at index ",
        i,
        " has shape ",
        out_tensors[i].sizes());
  }

  for (auto& out_tensor : out_tensors) {
    out_tensor.copy_(tensor, /*non_blocking=*/true);
  }
  return out_tensors;
}

std::vector<at::Tensor> supa_broadcast(const at::Tensor& tensor, at::IntArrayRef devices) {
  std::vector<at::Tensor> diff_device_dst_tensors;
  diff_device_dst_tensors.reserve(devices.size());
  for (auto device : devices) {
    auto target_device = supa_device_from_index(device);
    if (tensor.device() != target_device) {
      diff_device_dst_tensors.emplace_back(
          at::empty(tensor.sizes(), tensor.options().device(target_device), tensor.suggest_memory_format()));
    }
  }
  supa_broadcast_out(tensor, diff_device_dst_tensors);

  std::vector<at::Tensor> dst_tensors;
  dst_tensors.reserve(devices.size());
  auto it = diff_device_dst_tensors.begin();
  for (auto device : devices) {
    auto target_device = supa_device_from_index(device);
    if (tensor.device() != target_device) {
      dst_tensors.emplace_back(*it++);
    } else {
      dst_tensors.emplace_back(tensor);
    }
  }
  TORCH_INTERNAL_ASSERT(it == diff_device_dst_tensors.end());
  return dst_tensors;
}

tensor_list2d supa_broadcast_coalesced(at::TensorList tensors, at::IntArrayRef devices, size_t buffer_size) {
  TORCH_CHECK(!devices.empty(), "Expected at least one device to broadcast to");
  TORCH_CHECK(
      std::all_of(
          tensors.begin(),
          tensors.end(),
          [&](const at::Tensor& t) { return t.device() == supa_device_from_index(devices[0]); }),
      "All tensors must be on devices[0]: ",
      devices[0]);

  tensor_list2d outputs(devices.size());
  outputs[0] = tensors.vec();
  for (auto& output : outputs) {
    output.reserve(tensors.size());
  }

  UniqueTypeChecker type_checker;
  c10::supa::SUPAGuard device_guard(static_cast<c10::DeviceIndex>(devices[0]));
  for (auto& chunk : torch::utils::take_tensors(tensors, buffer_size)) {
    type_checker.show(chunk.type_id());
    if (chunk.options().is_sparse()) {
      auto flat_tuple = torch::utils::flatten_sparse_tensors(chunk.tensors);
      auto broadcast_indices = supa_broadcast(flat_tuple.first, devices);
      auto broadcast_values = supa_broadcast(flat_tuple.second, devices);
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.set_index(static_cast<c10::DeviceIndex>(devices[i]));
        auto& device_outputs = outputs[i];
        auto& inds = broadcast_indices[i];
        auto& vals = broadcast_values[i];
        for (const auto& var : torch::utils::unflatten_sparse_tensors(inds, vals, chunk.tensors)) {
          device_outputs.emplace_back(torch::autograd::make_variable(var.tensor_data(), false));
        }
      }
    } else {
      auto results = supa_broadcast(torch::utils::flatten_dense_tensors(chunk.tensors), devices);
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.set_index(static_cast<c10::DeviceIndex>(devices[i]));
        auto& device_outputs = outputs[i];
        for (auto& var : torch::utils::unflatten_dense_tensors(results[i], chunk.tensors)) {
          device_outputs.emplace_back(torch::autograd::make_variable(var.tensor_data(), false));
        }
      }
    }
  }

  if (!type_checker.unique) {
    for (auto& output : outputs) {
      torch::utils::reorder_tensors_like(output, tensors);
    }
  }
  return outputs;
}

std::vector<at::Tensor> supa_scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const std::optional<std::vector<int64_t>>& chunk_sizes,
    int64_t dim,
    const std::optional<std::vector<std::optional<c10::supa::SUPAStream>>>& streams) {
  TORCH_CHECK(!devices.empty(), "Expected at least one device to scatter to");
  if (chunk_sizes.has_value()) {
    TORCH_CHECK(
        chunk_sizes->size() == devices.size(),
        "Expected devices and chunk_sizes to be of same length, but got "
        "len(devices) = ",
        devices.size(),
        " and len(chunk_sizes) = ",
        chunk_sizes->size());
  }

  dim = at::maybe_wrap_dim(dim, tensor);
  std::vector<at::Tensor> chunks = chunk_sizes
      ? tensor.split_with_sizes(/*split_sizes=*/*chunk_sizes, /*dim=*/dim)
      : tensor.chunk(/*chunks=*/static_cast<int64_t>(devices.size()), /*dim=*/dim);

  c10::supa::OptionalSUPAStreamGuard supa_guard;
  for (const auto i : c10::irange(chunks.size())) {
    auto target_device = supa_device_from_index(devices[i]);
    if (chunks[i].device() != target_device) {
      if (i < (streams ? streams->size() : 0U) && (*streams)[i]) {
        TORCH_CHECK(
            (*streams)[i]->device_index() == target_device.index(),
            "Expected the device associated with the stream at index ",
            i,
            " (was ",
            (*streams)[i]->device_index(),
            ") to match the device supplied at that index (expected ",
            target_device.index(),
            ")");
        supa_guard.reset_stream((*streams)[i]->unwrap());
      }
      chunks[i] = chunks[i].to(
          target_device,
          /*non_blocking=*/true,
          /*copy=*/false,
          /*memory_format=*/at::MemoryFormat::Preserve);
    }
  }
  return chunks;
}

std::vector<at::Tensor> supa_scatter_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors,
    int64_t dim,
    const std::optional<std::vector<std::optional<c10::supa::SUPAStream>>>& streams) {
  TORCH_CHECK(!out_tensors.empty(), "Expected at least one output tensor to scatter to");

  dim = at::maybe_wrap_dim(dim, tensor);
  std::vector<int64_t> chunk_sizes;
  chunk_sizes.reserve(out_tensors.size());
  int64_t total_size = 0;
  const auto tensor_sizes = tensor.sizes();
  for (const auto i : c10::irange(out_tensors.size())) {
    const auto& out_tensor = out_tensors[i];
    TORCH_CHECK(
        out_tensor.device().type() == c10::DeviceType::PrivateUse1,
        "Expected all output tensors to be SUPA tensors, but output tensor at index ",
        i,
        " has device ",
        out_tensor.device());
    TORCH_CHECK(
        out_tensor.dim() == tensor.dim(),
        "Expected output tensor at index ",
        i,
        " to have ",
        tensor.dim(),
        " dimensions, but got ",
        out_tensor.dim());
    for (const auto d : c10::irange(tensor.dim())) {
      if (d == dim) {
        continue;
      }
      TORCH_CHECK(
          out_tensor.size(d) == tensor_sizes[d],
          "Output tensor at index ",
          i,
          " has invalid shape ",
          out_tensor.sizes(),
          ", but expected to match input shape except for scatter dim ",
          dim);
    }
    chunk_sizes.emplace_back(out_tensor.size(dim));
    total_size += out_tensor.size(dim);
  }
  TORCH_CHECK(
      total_size == tensor.size(dim),
      "Total size for output tensors along scatter dim ",
      dim,
      " does not match input size; got ",
      total_size,
      ", but expected ",
      tensor.size(dim));

  auto chunks = tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim);
  c10::supa::OptionalSUPAStreamGuard supa_guard;
  for (const auto i : c10::irange(out_tensors.size())) {
    if (i < (streams ? streams->size() : 0U) && (*streams)[i]) {
      const auto device_index = out_tensors[i].device().index();
      TORCH_CHECK(
          (*streams)[i]->device_index() == device_index,
          "Expected the device associated with the stream at index ",
          i,
          " (was ",
          (*streams)[i]->device_index(),
          ") to match the device supplied at that index (expected ",
          device_index,
          ")");
      supa_guard.reset_stream((*streams)[i]->unwrap());
    }
    out_tensors[i].copy_(chunks[i], /*non_blocking=*/true);
  }
  return out_tensors;
}

at::Tensor& supa_gather_out_impl(at::TensorList tensors, at::Tensor& out_tensor, int64_t dim) {
  std::vector<int64_t> chunk_sizes;
  chunk_sizes.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    chunk_sizes.emplace_back(tensor.size(dim));
  }
  auto chunks = out_tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim);
  for (const auto i : c10::irange(tensors.size())) {
    chunks[i].copy_(tensors[i], /*non_blocking=*/true);
  }
  return out_tensor;
}

SupaGatherInfo supa_gather_info(at::TensorList tensors, int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
  int64_t total_size = 0;
  const auto& first = tensors.front();
  const auto first_size = first.sizes();
  dim = at::maybe_wrap_dim(dim, first);
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "Expected all input tensors to be SUPA tensors, but tensor at index ",
        i,
        " has device ",
        tensor.device());
    TORCH_CHECK(
        tensor.dim() == static_cast<int64_t>(expected_size.size()),
        "Expected all input tensors to have the same number of dimensions, but tensor at index ",
        i,
        " has ",
        tensor.dim(),
        " dimensions, (expected ",
        expected_size.size(),
        ")");
    expected_size[dim] = tensor.size(dim);
    for (const auto dimension : c10::irange(expected_size.size())) {
      TORCH_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Input tensor at index ",
          i,
          " has invalid shape ",
          tensor.sizes(),
          ", but expected ",
          at::IntArrayRef(expected_size));
    }
    total_size += tensor.size(dim);
  }
  expected_size[dim] = total_size;
  return {std::move(expected_size), dim};
}

at::Tensor& supa_gather_out(at::TensorList tensors, at::Tensor& out_tensor, int64_t dim) {
  const auto gather_info = supa_gather_info(tensors, dim);
  TORCH_CHECK(
      out_tensor.sizes() == gather_info.expected_size,
      "Expected out tensor to have shape ",
      at::IntArrayRef(gather_info.expected_size),
      ", but got ",
      out_tensor.sizes());
  return supa_gather_out_impl(tensors, out_tensor, gather_info.dim);
}

at::Tensor supa_gather(at::TensorList tensors, int64_t dim, std::optional<int32_t> destination_index) {
  const auto gather_info = supa_gather_info(tensors, dim);
  at::Device device(c10::DeviceType::CPU);
  if (!destination_index || *destination_index != -1) {
    device = at::Device(
        c10::DeviceType::PrivateUse1,
        destination_index ? static_cast<c10::DeviceIndex>(*destination_index) : c10::DeviceIndex(-1));
  }

  auto result = at::empty(
      gather_info.expected_size, tensors.front().options().device(device), tensors.front().suggest_memory_format());
  return supa_gather_out_impl(tensors, result, gather_info.dim);
}

} // namespace

inline c10::DeviceIndex THBPUtils_unpackDeviceIndex(PyObject* obj) {
  int overflow = 0;
  long value = PyLong_AsLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  if (overflow != 0) {
    throw std::runtime_error("Overflow when unpacking DeviceIndex");
  }
  if (value > std::numeric_limits<c10::DeviceIndex>::max() || value < std::numeric_limits<c10::DeviceIndex>::min()) {
    throw std::runtime_error("Overflow when unpacking DeviceIndex");
  }
  return (c10::DeviceIndex)value;
}

inline PyObject* THBPUtils_packDeviceIndex(c10::DeviceIndex value) {
  return PyLong_FromLong(value);
}

static PyObject* THBPModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level

  c10::supa::SupaSysCtrl::GetInstance().supaInit();

  auto m = THPObjectPtr(PyImport_ImportModule("torch.supa"));
  if (!m) {
    throw python_error();
  }

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  auto num_gpus = c10::supa::device_count();
  auto* default_supa_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));

  for (int i = 0; i < num_gpus; i++) {
    auto gen = at::supa::detail::getDefaultSUPAGenerator(static_cast<c10::DeviceIndex>(i));
    auto* cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    PyTuple_SetItem(default_supa_generators, i, (PyObject*)cast_gen);
  }

  set_module_attr("default_generators", default_supa_generators);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_getCurrentBlasHandle_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  sublasHandle_t handle = at::supa::getCurrentSuBlasHandle();
  return PyLong_FromVoidPtr(handle);
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPModule_clearBlasWorkspaces_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::supa::clearSublasWorkspaces();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_supaSynchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    c10::supa::device_synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackLong(arg);

  torch_supa::utils::supa_lazy_init();
  c10::supa::set_device(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THBPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch_supa::utils::supa_lazy_init();
  auto current_device = c10::supa::ExchangeDevice(device_index);

  return THBPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_maybeExchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch_supa::utils::supa_lazy_init();
  auto current_device = c10::supa::MaybeExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch_supa::utils::supa_lazy_init();
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int32_t>(c10::supa::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_canDeviceAccessPeer_wrap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    THPUtils_invalidArguments(args, nullptr, "can_device_peer_access", 1, "(int device, int peer_device);");
    return nullptr;
  }
  TORCH_CHECK(THPUtils_checkLong(arg1), "invalid argument to canDeviceAccessPeer");
  TORCH_CHECK(THPUtils_checkLong(arg2), "invalid argument to canDeviceAccessPeer");
  auto device = THPUtils_unpackDeviceIndex(arg1);
  auto peer_device = THPUtils_unpackDeviceIndex(arg2);

  torch_supa::utils::supa_lazy_init();
  auto can_access = c10::supa::canDeviceAccessPeer(device, peer_device);
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyLong_FromLong(c10::supa::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_getCurrentStream_wrap(PyObject* /*unused*/, PyObject* device_index) {
  HANDLE_TH_ERRORS
  int64_t device = THPUtils_unpackLong(device_index);

  auto stream = c10::supa::getCurrentSUPAStream(static_cast<c10::DeviceIndex>(device));
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(output_tuple, 1, THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(output_tuple, 2, THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_getCurrentStream_raw(PyObject* /* unused */, PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  auto c10_device_index = THBPUtils_unpackDeviceIndex(device_index);
  return PyLong_FromVoidPtr(c10::supa::getCurrentSUPAStream(c10_device_index).stream());
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_getDefaultStream_wrap(PyObject* /* unused */, PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  auto c10_device_index = THBPUtils_unpackDeviceIndex(device_index);
  auto stream = c10::supa::getDefaultSUPAStream(c10_device_index);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(output_tuple, 1, THBPUtils_packDeviceIndex(stream.device_index()));
  PyTuple_SetItem(output_tuple, 2, THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_setStream_wrap(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {"stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = c10::supa::SUPAStream::unpack3(
      stream_id, static_cast<c10::DeviceIndex>(device_index), static_cast<c10::DeviceType>(device_type));

  auto device = c10::supa::current_device();
  if (device != stream.device_index()) {
    c10::supa::set_device(stream.device_index());
  }
  c10::supa::setCurrentSUPAStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_getCompiledVersion(PyObject* self, PyObject* noargs) {
#if defined(FAKE_SUPA_VERSION)
  return THPUtils_packInt64((int64_t)FAKE_SUPA_VERSION);
#else
  return THPUtils_packInt64((int64_t)SUPA_VERSION);
#endif
}

static PyObject* THBPModule_isCurrentStreamCapturing_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // If there's no cuda supa, c10::supa::currentStreamCaptureStatus returns
  // CaptureStatus::None without initializing a context.
  if (c10::supa::currentStreamCaptureStatus() == c10::supa::CaptureStatus::supaStreamCaptureStatusNone) {
    Py_RETURN_FALSE;
  }
  Py_RETURN_TRUE;

  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_supaHostAllocator(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::Allocator* allocator = at::supa::getCachingHostAllocator();
  return PyLong_FromVoidPtr(allocator);
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_supaCachingAllocator_raw_alloc(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    THPUtils_invalidArguments(args, nullptr, "caching_allocator_alloc", 1, "(ssize_t size, intptr_t stream);");
    return nullptr;
  }
  auto size = PyLong_AsSsize_t(size_o);
  supaStream_t stream = static_cast<supaStream_t>(PyLong_AsVoidPtr(stream_o));
  void* mem = nullptr;
  {
    pybind11::gil_scoped_release no_gil;
    mem = c10::supa::SUPACachingAllocator::raw_alloc_with_stream(size, stream);
  }
  return PyLong_FromVoidPtr(mem);
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_supaCachingAllocator_raw_delete(PyObject* _unused, PyObject* obj) {
  HANDLE_TH_ERRORS
  void* mem_ptr = PyLong_AsVoidPtr(obj);
  {
    pybind11::gil_scoped_release no_gil;
    c10::supa::SUPACachingAllocator::raw_delete(mem_ptr);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_supaCachingAllocator_enable(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkBool(arg), "supaCachingAllocator_enable expects a bool, but got ", THPUtils_typename(arg));
  c10::supa::SUPACachingAllocator::enable(THPUtils_unpackBool(arg));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_supaCachingAllocator_set_allocator_settings(PyObject* _unused, PyObject* env) {
  HANDLE_TH_ERRORS
  c10::supa::SUPACachingAllocator::setAllocatorSettings(THPUtils_unpackString(env));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_getAllocatorBackend(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(c10::supa::SUPACachingAllocator::name());
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_supaSleep(PyObject* _unused, PyObject* cycles) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(cycles), "torch.supa._sleep(): expected 'int'");
  int64_t unpacked_cycles = THPUtils_unpackLong(cycles);
  {
    pybind11::gil_scoped_release no_gil;
    at::supa::sleep(unpacked_cycles);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_hasPrimaryContext(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to has_primary_context");
  auto device_index = THBPUtils_unpackDeviceIndex(arg);
  if (c10::supa::hasPrimaryContext(device_index)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;

  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_getMemoryFraction(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* device_o = nullptr;
  if (!PyArg_ParseTuple(args, "O", &device_o)) {
    THPUtils_invalidArguments(args, nullptr, "get_memory_fraction", 1, "(int device);");
    return nullptr;
  }
  auto device_index = THBPUtils_unpackDeviceIndex(device_o);
  return PyFloat_FromDouble(c10::supa::SUPACachingAllocator::getMemoryFraction(device_index));
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_setMemoryFraction(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* fraction_o = nullptr;
  PyObject* device_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
    THPUtils_invalidArguments(args, nullptr, "set_memory_fraction", 1, "(double fraction, int device);");
    return nullptr;
  }
  double fraction = PyFloat_AsDouble(fraction_o);
  auto device_index = THBPUtils_unpackDeviceIndex(device_o);

  c10::supa::SUPACachingAllocator::setMemoryFraction(fraction, device_index);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THBPModule_hostEmptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    at::supa::CachingHostAllocator_emptyCache();
  }
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THBPModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    c10::supa::SUPACachingAllocator::emptyCache();
  }
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THBPModule_ipcCollect(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch_supa::supa::SupaIPCCollect();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_memoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const auto device_index = THBPUtils_unpackDeviceIndex(arg);

#if TORCH_VER >= TORCH_2_5_0
#if TORCH_VER >= TORCH_2_7_0
  using namespace c10::CachingAllocator;
#endif
  using namespace c10::CachingDeviceAllocator;
#else
  using c10::supa::SUPACachingAllocator::DeviceStats;
  using c10::supa::SUPACachingAllocator::Stat;
  using c10::supa::SUPACachingAllocator::StatArray;
  using c10::supa::SUPACachingAllocator::StatType;
#endif

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)> statTypeNames = {
        "all", "small_pool", "large_pool"};
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames.at(i)] = statToDict(statArray.at(i));
    }
    return dict;
  };

  const DeviceStats stats = c10::supa::SUPACachingAllocator::getDeviceStats(device_index);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["num_sync_all_streams"] = stats.num_sync_all_streams;
  result["num_device_alloc"] = stats.num_device_alloc;
  result["num_device_free"] = stats.num_device_free;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["requested_bytes"] = statArrayToDict(stats.requested_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_resetAccumulatedMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to reset_accumulated_memory_stats");
  const auto device_index = THBPUtils_unpackDeviceIndex(arg);
  c10::supa::SUPACachingAllocator::resetAccumulatedStats(device_index);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THBPModule_resetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const auto device_index = THBPUtils_unpackDeviceIndex(arg);
  c10::supa::SUPACachingAllocator::resetPeakStats(device_index);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

torch::CapturedTraceback* getFromContext(const std::shared_ptr<c10::GatheredContext>& x) {
  if (torch::CapturedTraceback* sc = dynamic_cast<torch::CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(false, "attempting to gather stack context from the wrong StackContext type.");
}

PyObject* THBPModule_memorySnapshot(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  using c10::supa::SUPACachingAllocator::BlockInfo;
  using c10::supa::SUPACachingAllocator::SegmentInfo;

  py::str device_s = "device";
  py::str address_s = "address";
  py::str total_size_s = "total_size";
  py::str allocated_size_s = "allocated_size";
  py::str active_size_s = "active_size";
  py::str requested_size_s = "requested_size";
  py::str stream_s = "stream";
  py::str segment_type_s = "segment_type";
  py::str segment_pool_id = "segment_pool_id";
  py::str large_s = "large";
  py::str small_s = "small";
  py::str size_s = "size";
  py::str state_s = "state";
  py::str active_allocated_s = "active_allocated";
  py::str active_pending_free_s = "active_pending_free";
  py::str inactive_s = "inactive";
  py::str addr_s = "addr";
  py::str cpp_frames_s = "cpp_frames";
  py::str blocks_s = "blocks";
  py::str is_expandable_s = "is_expandable";
  py::str frames_s = "frames";
  py::str time_us_s = "time_us";

  py::list empty_frames;
  std::vector<torch::CapturedTraceback*> to_gather_frames;
  std::vector<py::dict> to_gather_dest;

  auto add_frame_key = [&](const py::dict& d, const std::shared_ptr<c10::GatheredContext>& ctx) {
    if (ctx) {
      auto* sc = getFromContext(ctx);
      to_gather_frames.emplace_back(sc);
      to_gather_dest.emplace_back(d);
    } else {
      d[frames_s] = empty_frames;
    }
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict[device_s] = segmentInfo.device;
    segmentDict[address_s] = segmentInfo.address;
    segmentDict[total_size_s] = segmentInfo.total_size;
    segmentDict[allocated_size_s] = segmentInfo.allocated_size;
    segmentDict[active_size_s] = segmentInfo.active_size;
    segmentDict[requested_size_s] = segmentInfo.requested_size;
    // we want the python objects to pickle easily so use an int to
    // represent the stream rather than a torch.supa.stream object
    segmentDict[stream_s] = int64_t(segmentInfo.stream);
    segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);
    segmentDict[segment_pool_id] = segmentInfo.owner_private_pool_id;
    segmentDict[is_expandable_s] = segmentInfo.is_expandable;
    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    auto address = segmentInfo.address;
    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict[address_s] = address;
      blockDict[size_s] = blockInfo.size;
      blockDict[requested_size_s] = blockInfo.requested_size;
      blockDict[state_s] =
          (blockInfo.allocated ? active_allocated_s : (blockInfo.active ? active_pending_free_s : inactive_s));
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      blocks.append(blockDict);
      address += blockInfo.size;
    }
    segmentDict[blocks_s] = blocks;

    return segmentDict;
  };

  auto snapshot = c10::supa::SUPACachingAllocator::snapshot();

  py::list segments;

  for (const auto& segmentInfo : snapshot.segments) {
    segments.append(segmentInfoToDict(segmentInfo));
  }

  py::list traces;
  py::str action_s = "action";
  py::str alloc_s = "alloc";
  py::str free_requested_s = "free_requested";
  py::str free_completed_s = "free_completed";
  py::str segment_alloc_s = "segment_alloc";
  py::str segment_free_s = "segment_free";
  py::str segment_map_s = "segment_map";
  py::str segment_unmap_s = "segment_unmap";

  py::str snapshot_s = "snapshot";
  py::str oom_s = "oom";
  py::str device_free_s = "device_free";

  using namespace c10::supa::SUPACachingAllocator;

  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
      default:
        break;
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    py::list trace;
    for (const auto& te : traceInfo) {
      py::dict trace_entry;
      if (te.context_) {
        // without further compression frames can get really large on dump
        auto* sc = getFromContext(te.context_);
        to_gather_frames.emplace_back(sc);
        to_gather_dest.emplace_back(trace_entry);
      }
      trace_entry[action_s] = action_to_str(te.action_);
      trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] = te.addr_;
      trace_entry[size_s] = te.size_;
      trace_entry[stream_s] = int64_t(te.stream_);
      trace_entry[time_us_s] = te.time_.t_;
      trace.append(trace_entry);
    }
    traces.append(trace);
  }

  // py::list external_annotations;
  // for (const auto& ae : snapshot.external_annotations) {
  //   py::dict annotation_entry;
  //   for (const auto& md : ae.metadata_) {
  //     annotation_entry[(py::str)md.first] = md.second;
  //   }
  //   annotation_entry[device_s] = ae.device_;
  //   annotation_entry[time_us_s] = ae.time_.t_;
  //   external_annotations.append(annotation_entry);
  // }

  py::dict allocator_settings;
  // py::str last_allocator_settings_s = "PYTORCH_SUPA_ALLOC_CONF";
  py::str max_split_size_s = "max_split_size";
  py::str garbage_collection_threshold_s = "garbage_collection_threshold";
  py::str expandable_segments_s = "expandable_segments";
  py::str pinned_num_register_threads_s = "pinned_num_register_threads";
  py::str release_lock_on_malloc_s = "release_lock_on_supamalloc";
  py::str pinned_use_host_register_s = "pinned_use_supa_host_register";
  // py::str roundup_power2_divisions_s = "roundup_power2_divisions";

  // allocator_settings[last_allocator_settings_s] =
  //     snapshot.config_metadata.last_allocator_settings;
  allocator_settings[max_split_size_s] = int64_t(snapshot.config_metadata.max_split_size);
  allocator_settings[garbage_collection_threshold_s] = snapshot.config_metadata.garbage_collection_threshold;
  allocator_settings[expandable_segments_s] = snapshot.config_metadata.expandable_segments;
  allocator_settings[pinned_num_register_threads_s] = int64_t(snapshot.config_metadata.pinned_num_register_threads);
  allocator_settings[release_lock_on_malloc_s] = snapshot.config_metadata.release_lock_on_malloc;
  allocator_settings[pinned_use_host_register_s] = snapshot.config_metadata.pinned_use_host_register;
  // unsigned int roundup_key = 1;
  // py::dict roundup_settings;
  // for (const auto& v : snapshot.config_metadata.roundup_power2_divisions) {
  //   py::str roundup_key_s = std::to_string(roundup_key);
  //   roundup_settings[roundup_key_s] = int64_t(v);
  //   roundup_key *= 2;
  // }
  // allocator_settings[roundup_power2_divisions_s] = roundup_settings;

  py::dict result;
  result["segments"] = segments;
  result["device_traces"] = traces;
  result["allocator_settings"] = allocator_settings;
  // result["external_annotations"] = external_annotations;

  auto frames = py_symbolize(to_gather_frames);
  for (auto i : c10::irange(frames.size())) {
    to_gather_dest.at(i)[frames_s] = frames.at(i);
  }

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THBPModule_attachOutOfMemoryObserver(PyObject* _unused, PyObject* observer) {
  HANDLE_TH_ERRORS
  Py_XINCREF(observer);
  auto obs = [observer](int64_t device, int64_t alloc, int64_t device_allocated, int64_t device_free) {
    py::gil_scoped_acquire g;
    PyObject* result = PyObject_CallFunction(observer, "LLLL", device, alloc, device_allocated, device_free);
    if (!result) {
      throw py::error_already_set();
    }
    Py_XDECREF(result);
  };
  c10::supa::SupaSysCtrl::GetInstance().supaInit();
  c10::supa::SUPACachingAllocator::attachOutOfMemoryObserver(std::move(obs));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static void DLPack_Capsule_Destructor(PyObject* data) {
  if (C10_LIKELY(!PyCapsule_IsValid(data, "dltensor"))) {
    // early out, see DLPack spec: if a consuming library sets the capsule
    // name to something else, they own it and we don't need to do anything
    return;
  }
  HANDLE_TH_ERRORS
  // Causes overheads for validity checks again, but this case is rare
  // since consuming libraries should rename the capsule according to spec.
  // Note that this cannot set a python error (we checked validity above),
  // so we don't need to handle python error state here.
  DLManagedTensor* tensor = (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  // the dlMTensor has not been consumed, call deleter ourselves.
  // DLPack spec mentions that deleter may be NULL, but deleter from
  // `at::toDLPack` is never NULL, so no need for an additional check here.
  tensor->deleter(tensor);
  END_HANDLE_TH_ERRORS_RET()
}

static PyObject* THBPModule_toDLPack(PyObject* _unused, PyObject* data) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPVariable_Check(data), "data must be a Tensor");
  DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_Unpack(data));
  return PyCapsule_New(dlMTensor, "dltensor", DLPack_Capsule_Destructor);
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPModule_fromDLPack(PyObject* _unused, PyObject* data) {
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  DLManagedTensor* dlMTensor = (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  TORCH_CHECK(
      dlMTensor,
      "from_dlpack received an invalid capsule. "
      "Note that DLTensor capsules can be consumed only once, "
      "so you might have already constructed a tensor from it once.");

  // atensor steals the ownership of the underlying storage. It also passes a
  // destructor function that will be called when the underlying storage goes
  // out of scope. When the destructor is called, the dlMTensor is destructed
  // too.
  // HACK: Ensure that we hold the GIL here just in case the
  // managed tensor originating from a buggy NumPy build.
  if ((dlMTensor->dl_tensor).device.device_type == DLDeviceType::kDLCUDA) {
    // replace the cuda devices using privateuse1
    (dlMTensor->dl_tensor).device.device_type = DLDeviceType::kDLExtDev;
  }
  auto atensor = at::fromDLPack(dlMTensor);

  // Make sure this capsule will never be used again.
  PyCapsule_SetName(data, "used_dltensor");
  return THPVariable_Wrap(atensor);
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THBPModule_methods[] = {
    {"_supa_init", (PyCFunction)THBPModule_initExtension, METH_NOARGS, nullptr},
    {"_supa_synchronize", (PyCFunction)THBPModule_supaSynchronize, METH_NOARGS, nullptr},
    {"_supa_setDevice", (PyCFunction)THBPModule_setDevice_wrap, METH_O, nullptr},
    {"_supa_exchangeDevice", THBPModule_exchangeDevice, METH_O, nullptr},
    {"_supa_maybeExchangeDevice", THBPModule_maybeExchangeDevice, METH_O, nullptr},
    {"_supa_getDevice", (PyCFunction)THBPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_supa_getDeviceCount", (PyCFunction)THBPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
    {"_supa_canDeviceAccessPeer", THBPModule_canDeviceAccessPeer_wrap, METH_VARARGS, nullptr},
    {"_supa_getCurrentStream", (PyCFunction)THBPModule_getCurrentStream_wrap, METH_O, nullptr},
    {"_supa_getCurrentRawStream", THBPModule_getCurrentStream_raw, METH_O, nullptr},
    {"_supa_getDefaultStream", (PyCFunction)THBPModule_getDefaultStream_wrap, METH_O, nullptr},
    {"_supa_getCurrentBlasHandle", THBPModule_getCurrentBlasHandle_wrap, METH_NOARGS, nullptr},
    {"_supa_clearSublasWorkspaces", THBPModule_clearBlasWorkspaces_wrap, METH_NOARGS, nullptr},
    {"_supa_isCurrentStreamCapturing", (PyCFunction)THBPModule_isCurrentStreamCapturing_wrap, METH_NOARGS, nullptr},
    {"_supa_setStream", castPyCFunctionWithKeywords(THBPModule_setStream_wrap), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_supa_getCompiledVersion", THBPModule_getCompiledVersion, METH_NOARGS, nullptr},
    {"_supa_sleep", THBPModule_supaSleep, METH_O, nullptr},
    {"_supa_hasPrimaryContext", THBPModule_hasPrimaryContext, METH_O, nullptr},
    {"_supa_getMemoryFraction", THBPModule_getMemoryFraction, METH_VARARGS, nullptr},
    {"_supa_setMemoryFraction", THBPModule_setMemoryFraction, METH_VARARGS, nullptr},
    {"_supa_emptyCache", THBPModule_emptyCache, METH_NOARGS, nullptr},
    {"_supa_ipc_collect", THBPModule_ipcCollect, METH_NOARGS, nullptr},
    {"_supa_memoryStats", THBPModule_memoryStats, METH_O, nullptr},
    {"_supa_resetAccumulatedMemoryStats", THBPModule_resetAccumulatedMemoryStats, METH_O, nullptr},
    {"_supa_resetPeakMemoryStats", THBPModule_resetPeakMemoryStats, METH_O, nullptr},
    {"_supa_memorySnapshot", THBPModule_memorySnapshot, METH_NOARGS, nullptr},
    {"_supa_attach_out_of_memory_observer", THBPModule_attachOutOfMemoryObserver, METH_O, nullptr},
    {"_supa_supaHostAllocator", THBPModule_supaHostAllocator, METH_NOARGS, nullptr},
    {"_host_emptyCache", THBPModule_hostEmptyCache, METH_NOARGS, nullptr},
    {"_supa_supaCachingAllocator_raw_alloc", THBPModule_supaCachingAllocator_raw_alloc, METH_VARARGS, nullptr},
    {"_supa_supaCachingAllocator_raw_delete", THBPModule_supaCachingAllocator_raw_delete, METH_O, nullptr},
    {"_supa_supaCachingAllocator_enable", THBPModule_supaCachingAllocator_enable, METH_O, nullptr},
    {"_supa_supaCachingAllocator_set_allocator_settings",
     THBPModule_supaCachingAllocator_set_allocator_settings,
     METH_O,
     nullptr},
    {"_supa_getAllocatorBackend", THBPModule_getAllocatorBackend, METH_NOARGS, nullptr},
#ifdef USE_BCCL
    {"_bccl_version", THBPModule_bccl_version, METH_NOARGS, nullptr},
    {"_bccl_version_suffix", THBPModule_bccl_version_suffix, METH_NOARGS, nullptr},
    {"_bccl_unique_id", THBPModule_bccl_unique_id, METH_NOARGS, nullptr},
    {"_bccl_init_rank", THBPModule_bccl_init_rank, METH_VARARGS, nullptr},
    {"_bccl_reduce", THBPModule_bccl_reduce, METH_VARARGS, nullptr},
    {"_bccl_all_reduce", THBPModule_bccl_all_reduce, METH_VARARGS, nullptr},
    {"_bccl_broadcast", THBPModule_bccl_broadcast, METH_VARARGS, nullptr},
    {"_bccl_all_gather", THBPModule_bccl_all_gather, METH_VARARGS, nullptr},
    {"_bccl_reduce_scatter", THBPModule_bccl_reduce_scatter, METH_VARARGS, nullptr},
#endif
    {"_supa_to_dlpack", THBPModule_toDLPack, METH_O, nullptr},
    {"_supa_from_dlpack", THBPModule_fromDLPack, METH_O, nullptr},
    {nullptr}};

TORCH_SUPA_API PyMethodDef* THBPModule_get_methods() {
  return THBPModule_methods;
}

std::string uuid_to_string(const char* uuid_bytes) {
  // UUIDs are a 128-bit label. CUDA and HIP store this as char[16].
  // For string representation, the code here expands this to
  // 8-4-4-4-12 hex format, so each byte becomes 2 hex characters.
  return fmt::format(
      "{:02x}{:02x}{:02x}{:02x}-"
      "{:02x}{:02x}-"
      "{:02x}{:02x}-"
      "{:02x}{:02x}-"
      "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
      (uint8_t)uuid_bytes[0],
      (uint8_t)uuid_bytes[1],
      (uint8_t)uuid_bytes[2],
      (uint8_t)uuid_bytes[3],
      (uint8_t)uuid_bytes[4],
      (uint8_t)uuid_bytes[5],
      (uint8_t)uuid_bytes[6],
      (uint8_t)uuid_bytes[7],
      (uint8_t)uuid_bytes[8],
      (uint8_t)uuid_bytes[9],
      (uint8_t)uuid_bytes[10],
      (uint8_t)uuid_bytes[11],
      (uint8_t)uuid_bytes[12],
      (uint8_t)uuid_bytes[13],
      (uint8_t)uuid_bytes[14],
      (uint8_t)uuid_bytes[15]);
}

void RegisterSupaDeviceProperties(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<SUuuid_st>(m, "_SUuuid")
      .def_property_readonly(
          "bytes", [](const SUuuid_st& uuid) { return std::vector<uint8_t>(uuid.bytes, uuid.bytes + 16); })
      .def("__str__", [](const SUuuid_st& uuid) { return uuid_to_string(uuid.bytes); });

  py::class_<supaDeviceProp>(m, "_SupaDeviceProperties")
      .def_readonly("name", &supaDeviceProp::name)
      .def_readonly("major", &supaDeviceProp::major)
      .def_readonly("minor", &supaDeviceProp::minor)
      .def_readonly("is_multi_gpu_board", &supaDeviceProp::isMultiGpuBoard)
      .def_readonly("is_integrated", &supaDeviceProp::integrated)
      .def_readonly("multi_processor_count", &supaDeviceProp::multiProcessorCount)
      .def_readonly("total_memory", &supaDeviceProp::totalGlobalMem)
      .def_readonly("max_threads_per_multi_processor", &supaDeviceProp::maxThreadsPerMultiProcessor)
      .def_readonly("max_threads_per_block", &supaDeviceProp::maxThreadsPerBlock)
      .def_readonly("warp_size", &supaDeviceProp::warpSize)
      .def_readonly("shared_memory_per_block", &supaDeviceProp::sharedMemPerBlock)
      .def_property_readonly(
          "clock_rate",
          [](const supaDeviceProp&) {
            int clk = 0;
            C10_SUPA_CHECK(supaDeviceGetAttribute(&clk, supaDevAttrClockRate, c10::supa::current_device()));
            return clk;
          })
      .def_property_readonly(
          "memory_clock_rate",
          [](const supaDeviceProp&) {
            int mem_clk = 0;
            C10_SUPA_CHECK(supaDeviceGetAttribute(&mem_clk, supaDevAttrMemoryClockRate, c10::supa::current_device()));
            return mem_clk;
          })
      .def_readonly("memory_bus_width", &supaDeviceProp::memoryBusWidth)
      .def_readonly("shared_memory_per_block", &supaDeviceProp::sharedMemPerBlock)
      .def_readonly("shared_memory_per_block_optin", &supaDeviceProp::sharedMemPerBlockOptin)
      .def_readonly("shared_memory_per_multiprocessor", &supaDeviceProp::sharedMemPerMultiprocessor)
      .def_readonly("regs_per_multiprocessor", &supaDeviceProp::regsPerMultiprocessor)
      .def_readonly("gcnArchName", &supaDeviceProp::name)
      .def_readonly("uuid", &supaDeviceProp::uuid)
      .def_readonly("pci_bus_id", &supaDeviceProp::pciBusID)
      .def_readonly("pci_device_id", &supaDeviceProp::pciDeviceID)
      .def_readonly("pci_domain_id", &supaDeviceProp::pciDomainID)
      .def_readonly("L2_cache_size", &supaDeviceProp::l2CacheSize)
      .def("__repr__", [](const supaDeviceProp& prop) {
        std::ostringstream stream;
        stream << "_SupaDeviceProperties(name='" << prop.name << "', major=" << prop.major << ", minor=" << prop.minor
               << ", total_memory=" << prop.totalGlobalMem / (1024ULL * 1024)
               << "MB, multi_processor_count=" << prop.multiProcessorCount
               << ", uuid=" << uuid_to_string(prop.uuid.bytes) << ", pci_bus_id=" << prop.pciBusID
               << ", pci_device_id=" << prop.pciDeviceID << ", pci_domain_id=" << prop.pciDomainID
               << ", L2_cache_size=" << prop.l2CacheSize / (1024ULL * 1024) << "MB)";
        return stream.str();
      });

  m.def(
      "_supa_record_memory_history_legacy",
      static_cast<void (*)(bool, bool, int64_t, bool, bool)>(torch_supa::supa::_record_memory_history));

  m.def(
      "_supa_record_memory_history",
      static_cast<void (*)(std::optional<std::string>, std::optional<std::string>, const std::string&, size_t)>(
          torch_supa::supa::_record_memory_history));
  m.def("_supa_get_conv_benchmark_empty_cache", []() { return at::native::_cudnn_get_conv_benchmark_empty_cache(); });

  m.def("_sudnn_set_conv_benchmark_empty_cache", [](bool enable) {
    return at::native::_cudnn_set_conv_benchmark_empty_cache(enable);
  });
}

// We choose to ignore certain blocks that are currently allocated
// when we set the pool to its checkpoint. For those blocks, we need
// to swap out the deleter function of their corresponding blocks
// so that a deallocation is not triggered when they die.
void removeStorageDeleterFns(
    const std::vector<c10::StorageImpl*>& stale_live_storages,
    std::unordered_set<void*> definitely_stale_pointers) {
  for (c10::StorageImpl* stale_storage : stale_live_storages) {
    auto* ptr = stale_storage->data_ptr().get();
    auto allocated_pointer = definitely_stale_pointers.find(ptr);
    TORCH_CHECK(allocated_pointer != definitely_stale_pointers.end());
    auto* t = c10::supa::SUPACachingAllocator::get();
    bool succeeded =
        stale_storage->mutable_data_ptr().compare_exchange_deleter(t->raw_deleter(), &c10::detail::deleteNothing);

    TORCH_CHECK(succeeded, "Unexpected deleter function on storage, could not swap function");
  }
}

void addStorageDeleterFns(
    std::vector<c10::StorageImpl*>& storages_to_add_deleters_to,
    c10::supa::SUPACachingAllocator::CheckpointDelta& delta) {
  std::unordered_map<void*, c10::StorageImpl*> storages;
  for (auto& storage : storages_to_add_deleters_to) {
    storages[storage->data_ptr().get()] = storage;
  }

  for (auto& data_ptr : delta.dataptrs_allocd) {
    auto storage_pair = storages.find(data_ptr.get());
    if (storage_pair != storages.end()) {
      auto* ctx = storage_pair->second->data_ptr().get_context();
      TORCH_CHECK(ctx == nullptr, " Not expecting deleter function");
      storage_pair->second->set_data_ptr_noswap(std::move(data_ptr));
    } else {
      data_ptr.release_context();
    }
  }
}

void RegisterSupaPluggableAllocator(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<
      c10::supa::SUPACachingAllocator::SUPAAllocator,
      std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator>>(m, "_supa_SUPAAllocator");
  m.def(
      "_supa_getAllocator", []() { return py::cast(torch_supa::supa::SUPAPluggableAllocator::getCurrentAllocator()); });

  m.def(
      "_supa_changeCurrentAllocator",
      [](const std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator>& allocator) {
        torch_supa::supa::SUPAPluggableAllocator::changeCurrentAllocator(allocator);
      });
  py::class_<
      torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator,
      c10::supa::SUPACachingAllocator::SUPAAllocator,
      std::shared_ptr<torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator>>(m, "_SUPAPluggableAllocator")
      .def(
          "set_init_fn",
          [](torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator& self, uint64_t func_ptr) {
            using FuncType = void(int);
            std::function<FuncType> func =
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_init_fn(func);
          })
      .def(
          "set_reset_fn",
          [](torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator& self, uint64_t func_ptr) {
            using FuncType = void();
            std::function<FuncType> func =
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_reset_fn(func);
          })
      .def(
          "set_memory_fraction_fn",
          [](torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator& self, uint64_t func_ptr) {
            using FuncType = void(double, int);
            std::function<FuncType> func =
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_memory_fraction_fn(func);
          })
      .def(
          "set_base_alloc_fn",
          [](torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator& self, uint64_t func_ptr) {
            using FuncType = void*(void*, size_t*);
            std::function<FuncType> func =
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_base_alloc_fn(func);
          })
      .def(
          "set_record_stream_fn",
          [](torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator& self, uint64_t func_ptr) {
            using FuncType = void(void*, supaStream_t);
            std::function<FuncType> func =
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_record_stream_fn(func);
          })
      .def(
          "set_begin_allocate_to_pool",
          [](torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator& self, uint64_t func_ptr) {
            using FuncType = void(int, c10::supa::MempoolId_t, std::function<bool(supaStream_t)>);
            std::function<FuncType> func =
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_begin_allocate_to_pool(func);
          })
      .def(
          "set_end_allocate_to_pool_fn",
          [](torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator& self, uint64_t func_ptr) {
            using FuncType = void(int, c10::supa::MempoolId_t);
            std::function<FuncType> func =
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_end_allocate_to_pool_fn(func);
          })
      .def(
          "set_release_pool",
          [](torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator& self, uint64_t func_ptr) {
            using FuncType = void(int, c10::supa::MempoolId_t);
            std::function<FuncType> func =
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_release_pool(func);
          });
  m.def("_supa_customAllocator", [](uint64_t malloc_ptr, uint64_t free_ptr) {
    using namespace torch_supa::supa::SUPAPluggableAllocator;
    std::function<MallocFuncType> malloc_fn =
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<MallocFuncType*>(malloc_ptr);
    std::function<FreeFuncType> free_fn =
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<FreeFuncType*>(free_ptr);
    return createCustomAllocator(malloc_fn, free_fn);
  });

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<
      c10::supa::SUPACachingAllocator::AllocatorState,
      std::shared_ptr<c10::supa::SUPACachingAllocator::AllocatorState>>(m, "_supa_SUPAAllocator_AllocatorState");

  m.def("_supa_getCheckpointState", [](c10::DeviceIndex device, c10::supa::MempoolId_t id) {
    return c10::supa::SUPACachingAllocator::getCheckpointState(device, id);
  });

  m.def("_supa_beginAllocateCurrentStreamToPool", [](c10::DeviceIndex device, c10::supa::MempoolId_t mempool_id) {
    auto stream = c10::supa::getCurrentSUPAStream(device);
    TORCH_CHECK(stream, "Expected stream capture to be under way");
    c10::supa::SUPACachingAllocator::beginAllocateToPool(
        device, mempool_id, [stream](supaStream_t target) { return target == stream; });
  });

  m.def("_supa_beginAllocateToPool", [](c10::DeviceIndex device, c10::supa::MempoolId_t mempool_id) {
    c10::supa::SUPACachingAllocator::beginAllocateToPool(device, mempool_id, [](supaStream_t) { return true; });
  });

  m.def("_supa_beginAllocateCurrentThreadToPool", [](c10::DeviceIndex device, c10::supa::MempoolId_t mempool_id) {
    auto tid = std::this_thread::get_id();

    c10::supa::SUPACachingAllocator::beginAllocateToPool(device, mempool_id, [=](supaStream_t) {
      auto current_tid = std::this_thread::get_id();
      return current_tid == tid;
    });
  });

  m.def("_supa_endAllocateCurrentStreamToPool", [](c10::DeviceIndex device, c10::supa::MempoolId_t mempool_id) {
    c10::supa::SUPACachingAllocator::endAllocateToPool(device, mempool_id);
  });

  m.def("_supa_endAllocateToPool", [](c10::DeviceIndex device, c10::supa::MempoolId_t mempool_id) {
    c10::supa::SUPACachingAllocator::endAllocateToPool(device, mempool_id);
  });

  m.def("_supa_releasePool", [](c10::DeviceIndex device, c10::supa::MempoolId_t mempool_id) {
    c10::supa::SUPACachingAllocator::releasePool(device, mempool_id);
  });

  m.def(
      "_supa_checkPoolLiveAllocations",
      [](c10::DeviceIndex device, c10::supa::MempoolId_t mempool_id, const py::set& expected_live_allocations) {
        std::unordered_set<void*> allocations;
        allocations.reserve(expected_live_allocations.size());
        for (const auto& elem : expected_live_allocations) {
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          allocations.insert(reinterpret_cast<void*>(py::cast<size_t>(elem)));
        }
        return c10::supa::SUPACachingAllocator::checkPoolLiveAllocations(device, mempool_id, allocations);
      });

  m.def(
      "_supa_setCheckpointPoolState",
      [](c10::DeviceIndex device,
         std::shared_ptr<c10::supa::SUPACachingAllocator::AllocatorState> pps,
         const std::vector<size_t>& stale_storages_ptr,
         const std::vector<size_t>& storages_to_add_deleters_to_ptr = {}) {
        std::unordered_set<c10::StorageImpl*> ptr_set;
        // iterate on std::vector for determinism
        std::vector<c10::StorageImpl*> ptrs;
        for (size_t ptr_int : stale_storages_ptr) {
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          c10::StorageImpl* ptr = (c10::StorageImpl*)ptr_int;
          if (!ptr_set.count(ptr)) {
            ptrs.push_back(ptr);
            ptr_set.insert(ptr);
          }
        }
        auto delta = c10::supa::SUPACachingAllocator::setCheckpointPoolState(device, std::move(pps));
        auto& freed_pointers = delta.ptrs_freed;

        std::unordered_set<void*> allocd_set;
        for (auto& data_ptr : delta.dataptrs_allocd) {
          allocd_set.insert(data_ptr.get());
        }
        std::unordered_set<void*> freed_pointer_set;
        size_t definite_freed_count = 0;
        for (void* ptr : freed_pointers) {
          if (!allocd_set.count(ptr)) {
            definite_freed_count += 1;
          }
          freed_pointer_set.insert((ptr));
        }
        // that block has already been freed,
        // so even those this will error, so too will the allocator
        // when the corresponding tensor dies because there is no
        // live tensor corresponding to it
        TORCH_CHECK(
            ptr_set.size() >= definite_freed_count,
            "Any stale tensors which are being manually freed"
            " must be passed to set checkpoint");

        removeStorageDeleterFns(ptrs, freed_pointer_set);
        std::vector<c10::StorageImpl*> storages_to_add_deleters_to;
        storages_to_add_deleters_to.reserve(storages_to_add_deleters_to_ptr.size());
        for (size_t ptr_int : storages_to_add_deleters_to_ptr) {
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          storages_to_add_deleters_to.push_back((c10::StorageImpl*)ptr_int);
        }

        addStorageDeleterFns(storages_to_add_deleters_to, delta);
      });
}

void InitSupaModuleBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_supa_getDeviceProperties",
      [](int device) -> supaDeviceProp* {
        return at::supa::getDeviceProperties(static_cast<c10::DeviceIndex>(device));
      },
      py::return_value_policy::reference);

  m.def("_is_flash_attention_available", []() { return sdp::is_flash_attention_available(); });
  m.def("_can_use_flash_attention", [](const sdp::sdp_params& params, bool debug) {
    return sdp::can_use_flash_attention(params, debug);
  });
  m.def("_can_use_mem_efficient_attention", [](const sdp::sdp_params& params, bool debug) {
    return sdp::can_use_mem_efficient_attention(params, debug);
  });
  m.def("_can_use_cudnn_attention", [](const sdp::sdp_params& params, bool debug) {
    return sdp::can_use_cudnn_attention(params, debug);
  });

  m.def(
      "_supa_broadcast_coalesced",
      [](std::vector<at::Tensor>& tensors, const std::vector<int64_t>& devices, size_t buffer_size) {
        py::gil_scoped_release no_gil;
        return supa_broadcast_coalesced(tensors, devices, buffer_size);
      },
      py::arg("tensors"),
      py::arg("devices"),
      py::arg("buffer_size"));
  m.def(
      "_supa_broadcast",
      [](at::Tensor& tensor, const std::vector<int64_t>& devices) {
        py::gil_scoped_release no_gil;
        return supa_broadcast(tensor, devices);
      },
      py::arg("tensor"),
      py::arg("devices"));
  m.def(
      "_supa_broadcast_out",
      [](at::Tensor& tensor, std::vector<at::Tensor>& out_tensors) -> std::vector<at::Tensor>& {
        py::gil_scoped_release no_gil;
        return supa_broadcast_out(tensor, out_tensors);
      },
      py::arg("tensor"),
      py::arg("out"));
  m.def(
      "_supa_scatter",
      [](const at::Tensor& tensor,
         std::vector<int64_t>& devices,
         const std::optional<std::vector<int64_t>>& chunk_sizes,
         int64_t dim,
         std::optional<py::object> py_streams) {
        std::optional<std::vector<std::optional<c10::supa::SUPAStream>>> streams;
        if (py_streams.has_value() && !py_streams->is_none()) {
          py::handle handle = *py_streams;
          streams = THPUtils_PySequence_to_SUPAStreamList(handle.ptr());
        }
        py::gil_scoped_release no_gil;
        return supa_scatter(tensor, devices, chunk_sizes, dim, streams);
      },
      py::arg("tensor"),
      py::arg("devices"),
      py::arg("chunk_sizes"),
      py::arg("dim"),
      py::arg("streams"));
  m.def(
      "_supa_scatter_out",
      [](const at::Tensor& tensor,
         std::vector<at::Tensor>& out_tensors,
         int64_t dim,
         std::optional<py::object> py_streams) {
        std::optional<std::vector<std::optional<c10::supa::SUPAStream>>> streams;
        if (py_streams.has_value() && !py_streams->is_none()) {
          py::handle handle = *py_streams;
          streams = THPUtils_PySequence_to_SUPAStreamList(handle.ptr());
        }
        py::gil_scoped_release no_gil;
        return supa_scatter_out(tensor, out_tensors, dim, streams);
      },
      py::arg("tensor"),
      py::arg("out"),
      py::arg("dim"),
      py::arg("streams"));
  m.def(
      "_supa_gather",
      [](std::vector<at::Tensor>& tensors, int64_t dim, std::optional<int32_t> destination_index) {
        py::gil_scoped_release no_gil;
        return supa_gather(tensors, dim, destination_index);
      },
      py::arg("tensors"),
      py::arg("dim"),
      py::arg("destination_index"));
  m.def(
      "_supa_gather_out",
      [](std::vector<at::Tensor>& tensors, at::Tensor& out_tensor, int64_t dim) -> at::Tensor& {
        py::gil_scoped_release no_gil;
        return supa_gather_out(tensors, out_tensor, dim);
      },
      py::arg("tensors"),
      py::arg("out"),
      py::arg("dim"),
      py::return_value_policy::reference_internal);
}
