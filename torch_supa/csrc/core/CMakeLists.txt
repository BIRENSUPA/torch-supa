# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

FILE(GLOB _CORE_SRCS *.cpp supa/*.cpp supa/*/*.cpp)

LIST(APPEND CORE_SRCS ${_CORE_SRCS})

# Pass to parent
set(CORE_SRCS ${CORE_SRCS} PARENT_SCOPE)