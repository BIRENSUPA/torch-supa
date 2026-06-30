/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */


/* at::RecordFunction has different layout according to NDEBUG macro.
   if NDEBUG is defined in pytorch, define it here as well to keep consistency with at::RecordFunction
*/
#if defined(PYTORCH_HAS_NDEBUG) && !defined(NDEBUG)

#define NDEBUG
#include <ATen/record_function.h>
#undef NDEBUG

#elif !defined(PYTORCH_HAS_NDEBUG) && defined(NDEBUG)

#undef NDEBUG
#include <ATen/record_function.h>
#define NDEBUG

#else

#include <ATen/record_function.h>

#endif
