/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/**
 * Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once
// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#ifdef _WIN32
#if defined(C10_SUPA_BUILD_SHARED_LIBS)
#define C10_SUPA_EXPORT __declspec(dllexport)
#define C10_SUPA_IMPORT __declspec(dllimport)
#else
#define C10_SUPA_EXPORT
#define C10_SUPA_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_SUPA_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_SUPA_EXPORT
#endif // defined(__GNUC__)
#define C10_SUPA_IMPORT C10_SUPA_EXPORT
#endif // _WIN32

// This one is being used by libc10_cuda.so
#ifdef C10_SUPA_BUILD_MAIN_LIB
#define C10_SUPA_API C10_SUPA_EXPORT
#else
#define C10_SUPA_API C10_SUPA_IMPORT
#endif

#define TORCH_SUPA_API C10_SUPA_API

#define SUPA_API C10_SUPA_EXPORT
#define C10_COMPILE_TIME_MAX_SUPA_GPUS 16
