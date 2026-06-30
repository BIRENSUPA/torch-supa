/* Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd. */

/*
 * Copyright (c) 2026 Shanghai Biren Technology Co., Ltd. All rights reserved.
 */

#pragma once

#ifndef NO_SUPA_RT_HEADER
#include <supa_driver.h>
#include <runtime/supa_runtime_api.h>
#endif

static constexpr int64_t COMPUTE_MAJOR_VERSION_9 = 9;

#if defined(SUPA_VERSION)
#if (SUPA_VERSION == 2000)
#define FAKE_SUPA_VERSION 12090
#endif
#endif



