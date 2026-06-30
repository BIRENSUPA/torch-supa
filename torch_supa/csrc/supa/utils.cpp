/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>
#include <cstdarg>
#include <string>

#include "torch_supa/csrc/supa/THBP.h"

// NB: It's a list of *optional* SUPAStream; when nullopt, that means to use
// whatever the current stream of the device the input is associated with was.
std::vector<std::optional<c10::supa::SUPAStream>> THPUtils_PySequence_to_SUPAStreamList(PyObject* obj) {
  if (!PySequence_Check(obj)) {
    throw std::runtime_error("Expected a sequence in THPUtils_PySequence_to_SUPAStreamList");
  }
  THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, nullptr));
  if (seq.get() == nullptr) {
    throw std::runtime_error("expected PySequence, but got " + std::string(THPUtils_typename(obj)));
  }

  std::vector<std::optional<c10::supa::SUPAStream>> streams;
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject* stream = PySequence_Fast_GET_ITEM(seq.get(), i);

    if (PyObject_IsInstance(stream, (PyObject*)THPStreamClass)) {
      // Spicy hot reinterpret cast!!
      streams.emplace_back(c10::supa::SUPAStream::unpack3(
          (reinterpret_cast<THPStream*>(stream))->stream_id,
          static_cast<c10::DeviceIndex>(reinterpret_cast<THPStream*>(stream)->device_index),
          static_cast<c10::DeviceType>((reinterpret_cast<THPStream*>(stream))->device_type)));
    } else if (stream == Py_None) {
      streams.emplace_back();
    } else {
      throw std::runtime_error("Unknown data type found in stream list. Need torch.supa.Stream or None");
    }
  }
  return streams;
}
