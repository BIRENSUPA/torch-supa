# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

FILE(GLOB _ATEN_SRCS
    *.cpp
    common/*.cpp
    ops/*.cpp
    sudnn/*.cpp
    ops/*/*.cpp)

FILE(GLOB _REGISTER_SRCS
    generated/*.cpp)

FILE(GLOB _ATEN_SUPA_KERNELS
    ops/kernels/*.su)

FILE(GLOB _ATEN_SUDA_KERNELS
    ops/kernels/*.cu
    ops/kernels/reduce/*.cu
    ops/kernels/elementwise/*.cu)

LIST(APPEND ATEN_SRCS ${_ATEN_SRCS})

set(ATEN_SUPA_SRCS ${_ATEN_SUPA_KERNELS} PARENT_SCOPE)
set(ATEN_SUDA_SRCS ${_ATEN_SUDA_KERNELS} PARENT_SCOPE)
set(ATEN_SRCS ${ATEN_SRCS} PARENT_SCOPE)
set(REGISTER_SRCS ${_REGISTER_SRCS} PARENT_SCOPE)
