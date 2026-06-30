# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

FILE(GLOB _DIST_SRCS *.cpp symm_mem/*.cpp supa/*.cpp)
FILE(GLOB _DIST_KERNELS *.su symm_mem/*.su supa/*.su)

set(DIST_SRCS ${_DIST_SRCS} PARENT_SCOPE)
set(DIST_SUPA_SRCS ${_DIST_KERNELS} PARENT_SCOPE)
