/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

#pragma once

namespace torch_supa {
namespace profiler {
// enum class ExperConfigType {
//     TRACE_LEVEL = 0,
//     METRICS,
//     L2_CACHE,
//     RECORD_OP_ARGS,
//     MSPROF_TX,
//     OP_ATTR,
//     HOST_SYS,
//     MSTX_DOMAIN_INCLUDE,
//     MSTX_DOMAIN_EXCLUDE,
//     SYS_IO,
//     SYS_INTERCONNECTION,
//     CONFIG_TYPE_MAX_COUNT  // 表示枚举的总数，固定放在枚举的最后一个
// };
PyMethodDef *profiler_functions();
// TORCH_SUPA_API void initMstx(PyObject *module);
} // namespace profiler
} // namespace torch_supa
