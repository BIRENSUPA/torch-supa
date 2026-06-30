# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

FILE(GLOB _PROFILER_SRCS
    *.cpp)

LIST(APPEND PROFILER_SRCS ${_PROFILER_SRCS})

# Pass to parent
set(PROFILER_SRCS ${PROFILER_SRCS} PARENT_SCOPE)
