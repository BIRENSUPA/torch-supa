/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once
#include <c10/macros/Macros.h>

// Marks a lambda as executable on both the host and device. The __host__
// attribute is important so that we can access static type information from
// the host, even if the function is typically only executed on the device.
#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

constexpr uint32_t num_threads() {
  return C10_WARP_SIZE * 4 * 2;
}

constexpr int thread_work_size() { return 32; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }
