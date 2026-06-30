/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <torch/csrc/python_headers.h>

PyObject* THBPModule_bccl_version(PyObject* self, PyObject* args);
PyObject* THBPModule_bccl_version_suffix(PyObject* self, PyObject* args);
PyObject* THBPModule_bccl_unique_id(PyObject* self, PyObject* args);
PyObject* THBPModule_bccl_init_rank(PyObject* self, PyObject* args);
PyObject* THBPModule_bccl_reduce(PyObject* self, PyObject* args);
PyObject* THBPModule_bccl_all_reduce(PyObject* self, PyObject* args);
PyObject* THBPModule_bccl_broadcast(PyObject* self, PyObject* args);
PyObject* THBPModule_bccl_all_gather(PyObject* self, PyObject* args);
PyObject* THBPModule_bccl_reduce_scatter(PyObject* self, PyObject* args);
