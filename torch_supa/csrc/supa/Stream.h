/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#ifndef THBP_STREAM_INC
#define THBP_STREAM_INC

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"

struct THBPStream : THPStream {
  c10::supa::SUPAStream supa_stream;
};
extern PyObject *THBPStreamClass;

TORCH_SUPA_API void THBPStream_init(PyObject *module);

inline bool THBPStream_Check(PyObject *obj) {
  return THBPStreamClass && PyObject_IsInstance(obj, THBPStreamClass);
}

TORCH_SUPA_API std::vector<c10::optional<c10::supa::SUPAStream>>
THBPUtils_PySequence_to_SUPAStreamList(PyObject *obj);

c10::supa::SUPAStream THBPUtils_PyObject_to_SUPAStream(PyObject *py_stream);

#endif // THBP_STREAM_INC
