# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

FILE(GLOB _TEST_SRCS
  *.cpp
  )

LIST(APPEND TEST_SRCS ${_TEST_SRCS} ${GTEST_ROOT}/src/gtest-all.cc)

# Pass to parent
set(TEST_SRCS ${TEST_SRCS} PARENT_SCOPE)
