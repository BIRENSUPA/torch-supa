/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <ATen/MapAllocator.h>
#include <Python.h>
#include <c10/util/flat_hash_map.h>
#include <libshm.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/python_numbers.h>
#include <vector>

#include <torch_supa/csrc/core/SUPAStorageImpl.h>
#include <torch_supa/csrc/core/supa/SUPAFunctions.h>
#include <torch_supa/csrc/core/supa/SUPAGuard.h>
#include <torch_supa/csrc/core/supa/TorchVersion.h>
#include <torch_supa/csrc/ipc/StorageSharing.h>
#include <torch_supa/csrc/ipc/SupaIPCTypes.h>

namespace torch_supa {
namespace reductions {
static PyObject* THBPStorage_shareSupa(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(args);
  TORCH_CHECK(storage.device_type() == at::kPrivateUse1, "_share_supa_: only available on SUPA");
  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();
  if (storage_impl->received_cuda()) {
    TORCH_CHECK(
        false,
        "Attempted to send SUPA tensor received from another process; this is not currently supported. Consider cloning before sending.");
  }

  at::DeviceGuard device_guard(storage.device());
  // c10::supa::LazySetDevice(storage.device().index());
  THPObjectPtr tuple(PyTuple_New(8));
  THPObjectPtr device(THPUtils_packInt32(storage.device().index()));
  THPObjectPtr _handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr size_bytes(THPUtils_packUInt64(storage.nbytes()));
  THPObjectPtr _offset_bytes(THPUtils_packInt32(0));
  THPObjectPtr _ref_counter(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _ref_counter_offset(THPUtils_packInt32(0));
  THPObjectPtr _event_handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _event_sync_required(Py_None);
  Py_INCREF(Py_None);
  if (storage.data()) {
    auto shandle = c10::supa::SUPACachingAllocator::shareIpcHandle(storage.mutable_data());
    _handle = PyBytes_FromStringAndSize(shandle.handle.c_str(), (Py_ssize_t)shandle.handle.size());
    _offset_bytes = PyLong_FromSsize_t((Py_ssize_t)shandle.offset);

    // Put Storage Data behind new ref counting context
    // See Note [SUPA IPC Refcounting implementation explained]
    at::DataPtr sent_data_ptr = torch_supa::supa::GetNewRefCountedSentData(storage.mutable_data(), storage.device());
    auto old_data_ptr = storage.set_data_ptr(std::move(sent_data_ptr));
    auto* sent_data = static_cast<torch_supa::supa::SupaIPCSentData*>(storage.data_ptr().get_context());
    sent_data->set_original_ptr(std::move(old_data_ptr));
    _ref_counter = PyBytes_FromString((sent_data->handle()).c_str());
    _ref_counter_offset = THPUtils_packUInt64(sent_data->offset());

    supaIpcEventHandle_t ipc_event_handle{};

    if (sent_data->event_sync_required_) {
      C10_SUPA_CHECK(supaIpcGetEventHandle(&ipc_event_handle, sent_data->event_));
    }

    _event_handle = PyBytes_FromStringAndSize((char*)&ipc_event_handle, SUPA_IPC_HANDLE_SIZE);
    _event_sync_required = PyBool_FromLong(sent_data->event_sync_required_);
  }

  if (!tuple || !device || !_handle || !size_bytes || !_offset_bytes || !_event_handle) {
    return nullptr;
  }
  PyTuple_SET_ITEM(tuple.get(), 0, device.release());
  PyTuple_SET_ITEM(tuple.get(), 1, _handle.release());
  // Size(in bytes) of the real storage, note this is not the size of basePtr
  // memory block.
  PyTuple_SET_ITEM(tuple.get(), 2, size_bytes.release());
  // Offset(in bytes) of the real storage in the basePtr memory block.
  // NB: this offset MUST be in bytes instead of numel, since we use
  // (storage_handle, offset)
  //     as key in shared_cache(multiprocessing/reduction.py).
  //     Offset in numel cannot uniquely represent a storage.
  PyTuple_SET_ITEM(tuple.get(), 3, _offset_bytes.release());
  PyTuple_SET_ITEM(tuple.get(), 4, _ref_counter.release());
  PyTuple_SET_ITEM(tuple.get(), 5, _ref_counter_offset.release());
  PyTuple_SET_ITEM(tuple.get(), 6, _event_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 7, _event_sync_required.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPStorage_releaseIPCCounter(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  PyObject* _ref_counter = PyTuple_GET_ITEM(args, 0);
  PyObject* _ref_counter_offset = PyTuple_GET_ITEM(args, 1);
  if (!(PyBytes_Check(_ref_counter) && THPUtils_checkLong(_ref_counter_offset))) {
    THPUtils_invalidArguments(
        args, nullptr, "_release_ipc_counter in SUPA mode", 1, "(bytes _ref_counter, int _ref_counter_offset)");
    return nullptr;
  }
  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  ptrdiff_t ref_counter_offset = (ptrdiff_t)THPUtils_unpackLong(_ref_counter_offset);
  // We don't want to break existing code, so resource deletion is best
  // effort basis. Exception expected if producer process terminated
  // before consumer released data.
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
  try {
    auto sptr = at::RefcountedMapAllocator::makeDataPtr(
        ref_counter_handle.c_str(), flags, sizeof(int64_t) * torch_supa::supa::SUPA_IPC_REF_COUNTER_FILE_SIZE, nullptr);
    *(static_cast<int64_t*>(sptr.get()) + ref_counter_offset) -= 1;
  } catch (c10::Error& err) {
    // Already warned inside of producer process
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static std::string THPStorage_bytesAsHandleString(PyObject* handle) {
  HANDLE_TH_ERRORS
  char* buffer = nullptr;
  Py_ssize_t handle_size = 0;
  if (PyBytes_AsStringAndSize(handle, &buffer, &handle_size) == -1) {
    TORCH_CHECK(handle_size == SUPA_IPC_HANDLE_SIZE, "incorrect handle");
  }
  return std::string(buffer, handle_size);
  END_HANDLE_TH_ERRORS_RET("")
}

static PyObject* THBPStorage_newSharedSupa(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 8, "tuple of 8 items expected");
  PyObject* _device = PyTuple_GET_ITEM(args, 0);
  PyObject* _handle = PyTuple_GET_ITEM(args, 1);
  PyObject* _size_bytes = PyTuple_GET_ITEM(args, 2);
  PyObject* _offset_bytes = PyTuple_GET_ITEM(args, 3);
  PyObject* _ref_counter = PyTuple_GET_ITEM(args, 4);
  PyObject* _ref_counter_offset = PyTuple_GET_ITEM(args, 5);
  PyObject* _event_handle = PyTuple_GET_ITEM(args, 6);
  PyObject* _event_sync_required = PyTuple_GET_ITEM(args, 7);
  if (!(THPUtils_checkLong(_device) && THPUtils_checkLong(_size_bytes) && PyBytes_Check(_handle) &&
        PyBytes_Check(_ref_counter) && PyBytes_Check(_event_handle) && THPUtils_checkLong(_offset_bytes) &&
        THPUtils_checkLong(_ref_counter_offset) && PyBool_Check(_event_sync_required))) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in SUPA mode",
        1,
        "(int device, bytes handle, int storage_size_bytes, int storage_offset_bytes, bytes _ref_counter, int _ref_counter_offset, bytes event_handle, bool event_sync_required)");
    return nullptr;
  }

  size_t storage_size = (size_t)THPUtils_unpackLong(_size_bytes) / sizeof(uint8_t);
  ptrdiff_t storage_offset_bytes = (ptrdiff_t)THPUtils_unpackLong(_offset_bytes);

  const auto device = c10::checked_convert<c10::DeviceIndex>(THPUtils_unpackLong(_device), "c10::DeviceIndex");
  c10::supa::SUPAGuard device_guard(device);

  if (PyObject_IsTrue(_event_sync_required)) {
    // Ensure that producer prepared all tensor's data
    std::string s_ipc_event_handle = THPStorage_bytesAsHandleString(_event_handle);
    if (s_ipc_event_handle.empty()) {
      return nullptr;
    }
    const auto* ipc_event_handle = reinterpret_cast<const supaIpcEventHandle_t*>(s_ipc_event_handle.c_str());
    supaEvent_t event = nullptr;
    C10_SUPA_CHECK(supaIpcOpenEventHandle(&event, *ipc_event_handle));
    C10_SUPA_CHECK(supaStreamWaitEvent(c10::supa::getCurrentSUPAStream(device), event, 0));
  }

  std::string s_handle = THPStorage_bytesAsHandleString(_handle);
  if (s_handle.empty()) {
    return nullptr;
  }
  std::shared_ptr<void> basePtr = c10::supa::SUPACachingAllocator::getIpcDevPtr(s_handle);

  // Offset the basePtr to reconstruct the real storage
  // devPtr = basePtr + storage_offset
  void* devPtr = basePtr.get();
  devPtr = (char*)devPtr + storage_offset_bytes;

  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  ptrdiff_t ref_counter_offset = (ptrdiff_t)THPUtils_unpackLong(_ref_counter_offset);

  struct IpcDeleterContext {
    std::string ref_counter_handle;
    ptrdiff_t ref_counter_offset{};
    c10::DeviceIndex device{-1};
    torch_supa::supa::SupaIPCReceivedData received_data;
  };

  auto ctx = std::make_unique<IpcDeleterContext>();
  ctx->ref_counter_handle = std::move(ref_counter_handle);
  ctx->ref_counter_offset = ref_counter_offset;
  ctx->device = device;
  ctx->received_data.shared_ptr_ = std::move(basePtr);

  auto cur_device = c10::supa::current_device();
  c10::DataPtr data_ptr(
      devPtr,
      ctx.release(),
      +[](void* ctx_) {
        std::unique_ptr<IpcDeleterContext> ctx(static_cast<IpcDeleterContext*>(ctx_));
        ctx->received_data.shared_ptr_.reset();
        try {
          c10::supa::stream_synchronize(c10::supa::getCurrentSUPAStream(ctx->device));

        } catch (c10::Error& err) {
          // Already warned inside of producer process
        }

        // We don't want to break existing code, so resource deletion is best
        // effort basis. Exception expected if producer process terminated
        // before consumer released data.
        int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
        try {
          auto sptr = at::RefcountedMapAllocator::makeDataPtr(
              ctx->ref_counter_handle.c_str(),
              flags,
              sizeof(int64_t) * torch_supa::supa::SUPA_IPC_REF_COUNTER_FILE_SIZE,
              nullptr);
          *(static_cast<int64_t*>(sptr.get()) + ctx->ref_counter_offset) -= 1;
        } catch (c10::Error& err) {
          // Already warned inside of producer process
        }
      },
      at::Device(at::DeviceType::PrivateUse1, cur_device));

  c10::intrusive_ptr<c10::StorageImpl> base = c10::make_intrusive<SUPAStorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      storage_size,
      std::move(data_ptr),
      /*allocator=*/nullptr,
      /*resizable=*/false);
  base->set_resizable(false);
  base->set_received_cuda(true);
#if TORCH_VER <= TORCH_2_8_0
  return THPStorage_NewWithStorage(THPStorageClass, std::move(base), c10::impl::PyInterpreterStatus::TAGGED_BY_US);
#else
  return THPStorage_NewWithStorage(THPStorageClass, std::move(base));
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPStorage_isShared(PyObject* self, PyObject* arg) {
  const auto& storage = THPStorage_Unpack(self);
  if (storage.device_type() == at::kPrivateUse1) {
    Py_RETURN_TRUE;
  }
  if (at::MapAllocator::fromDataPtr(storage.data_ptr()) || THManagedMapAllocator::fromDataPtr(storage.data_ptr())) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef methods[] = {
    {"_share_supa_", THBPStorage_shareSupa, METH_O, nullptr},
    {"_new_shared_supa", THBPStorage_newSharedSupa, METH_VARARGS, nullptr},
    {"_release_ipc_counter_supa", THBPStorage_releaseIPCCounter, METH_VARARGS, nullptr},
    {"is_shared", THBPStorage_isShared, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

TORCH_SUPA_API PyMethodDef* storage_ipc_functions() {
  return methods;
}
} // namespace reductions
} // namespace torch_supa
