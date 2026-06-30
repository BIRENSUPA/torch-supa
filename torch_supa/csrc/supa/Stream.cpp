/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <pybind11/pybind11.h>
#include <structmember.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>

#include "torch_supa/csrc/core/supa/SUPAFunctions.h"
#include "torch_supa/csrc/core/supa/SUPAGuard.h"
#include "torch_supa/csrc/core/supa/sys_ctrl/SupaSysCtrl.h"
#include "torch_supa/csrc/supa/Module.h"
#include "torch_supa/csrc/supa/Stream.h"

PyObject* THBPStreamClass = nullptr;

static PyObject* THBPStream_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS

  const auto current_device = c10::supa::current_device();

  int priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;
  uint64_t stream_ptr = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {"priority", "stream_id", "device_index", "device_type", "stream_ptr", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|iLLLK",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &priority,
          &stream_id,
          &device_index,
          &device_type,
          &stream_ptr)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  if (stream_ptr) {
    TORCH_CHECK(priority == 0, "Priority was explicitly set for a external stream")
  }
  at::supa::SUPAStream stream = (stream_id || device_index || device_type)
      ? at::supa::SUPAStream::unpack3(
            stream_id, static_cast<c10::DeviceIndex>(device_index), static_cast<c10::DeviceType>(device_type))
      : stream_ptr ? at::supa::getStreamFromExternal(
                         // NOLINTNEXTLINE(performance-no-int-to-ptr)
                         reinterpret_cast<supaStream_t>(stream_ptr),
                         current_device)
                   : at::supa::getStreamFromPool(priority);

  THBPStream* self = (THBPStream*)ptr.get();
  self->stream_id = static_cast<int64_t>(stream.id());
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  new (&self->supa_stream) at::supa::SUPAStream(stream);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THBPStream_dealloc(THBPStream* self) {
  self->supa_stream.~SUPAStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THBPStream_get_supa_stream(THBPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->supa_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPStream_get_priority(THBPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->supa_stream.priority());
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPStream_priority_range(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto [least_priority, greatest_priority] = at::supa::SUPAStream::priority_range();
  return Py_BuildValue("(ii)", least_priority, greatest_priority);
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPStream_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = (THBPStream*)_self;
  return PyBool_FromLong(self->supa_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPStream_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    auto* self = (THBPStream*)_self;
    self->supa_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THBPStream_eq(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto* self = (THBPStream*)_self;
  auto* other = (THBPStream*)_other;
  return PyBool_FromLong(self->supa_stream == other->supa_stream);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMemberDef THBPStream_members[] = {{nullptr}};

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyGetSetDef THBPStream_properties[] = {
    {"supa_stream", (getter)THBPStream_get_supa_stream, nullptr, nullptr, nullptr},
    {"priority", (getter)THBPStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static PyMethodDef THBPStream_methods[] = {
    {"query", THBPStream_query, METH_NOARGS, nullptr},
    {"synchronize", THBPStream_synchronize, METH_NOARGS, nullptr},
    {"priority_range", THBPStream_priority_range, METH_STATIC | METH_NOARGS, nullptr},
    {"__eq__", THBPStream_eq, METH_O, nullptr},
    {nullptr}};

PyTypeObject THBPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._SUPAStreamBase", /* tp_name */
    sizeof(THBPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THBPStream_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THBPStream_methods, /* tp_methods */
    THBPStream_members, /* tp_members */
    THBPStream_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THBPStream_pynew, /* tp_new */
};

void THBPStream_init(PyObject* module) {
  Py_INCREF(THPStreamClass);
  THBPStreamType.tp_base = THPStreamClass;
  THBPStreamClass = (PyObject*)&THBPStreamType;
  if (PyType_Ready(&THBPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THBPStreamType);
  if (PyModule_AddObject(module, "_SUPAStreamBase", (PyObject*)&THBPStreamType) < 0) {
    throw python_error();
  }
}