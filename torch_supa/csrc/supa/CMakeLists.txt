# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

FILE(GLOB _SUPA_SRCS *.cpp)

LIST(APPEND SUPA_SRCS ${_SUPA_SRCS})

FILE(GLOB _SHARED_SRCS shared/*.cpp)
LIST(APPEND SUPA_SRCS ${_SHARED_SRCS})

# Pass to parent
set(SUPA_SRCS ${SUPA_SRCS} PARENT_SCOPE)
