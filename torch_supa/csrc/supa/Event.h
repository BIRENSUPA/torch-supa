/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#ifndef THBP_EVENT_INC
#define THBP_EVENT_INC

#include "torch_supa/csrc/core/supa/SUPAEvent.h"
#include "torch_supa/csrc/core/supa/TorchVersion.h"
#include <torch/csrc/python_headers.h>
#if TORCH_VER >= TORCH_2_6_0
#include <torch/csrc/Event.h>
struct THBPEvent : THPEvent {
  c10::supa::SUPAEvent supa_event;
};
#else
struct THBPEvent {
  PyObject_HEAD c10::supa::SUPAEvent supa_event;
};
#endif

extern PyObject *THBPEventClass;

void THBPEvent_init(PyObject *module);

inline bool THBPEvent_Check(PyObject *obj) {
  return THBPEventClass && PyObject_IsInstance(obj, THBPEventClass);
}

#endif // THBP_EVENT_INC