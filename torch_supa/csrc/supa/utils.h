/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

#include <torch/csrc/utils/python_numbers.h>
#include "torch_supa/csrc/core/supa/SUPAStream.h"

#include <vector>

std::vector<std::optional<c10::supa::SUPAStream>> THPUtils_PySequence_to_SUPAStreamList(PyObject* obj);
