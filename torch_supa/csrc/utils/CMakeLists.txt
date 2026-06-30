# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

FILE(GLOB _UTILS_SRCS *.cpp
logger/*.cpp)

LIST(APPEND UTILS_SRCS ${_UTILS_SRCS})

# Pass to parent
set(UTILS_SRCS ${UTILS_SRCS} PARENT_SCOPE)
