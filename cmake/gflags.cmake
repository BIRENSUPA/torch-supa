# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

set(GFLAGS_VER           2.2.2)
set(GFLAGS_NAME          gflags)
set(GFLAGS_FOLDER        ${GFLAGS_NAME}-${GFLAGS_VER})
set(GFLAGS_SOURCE_DIR    ${TORCH_SUPA_THIRD_PARTY_ROOT}/gflags)
set(GFLAGS_ROOT          ${CMAKE_BINARY_DIR}/third-party/${GFLAGS_FOLDER})
set(GFLAGS_LIB           $<TARGET_FILE:${GFLAGS_NAME}_static>)

function(build_gflags)
  set(GFLAGS_IS_SUBPROJECT FALSE)
  set(BUILD_STATIC_LIBS ON)
  set(CMAKE_CXX_FLAGS "-fPIC")
  set(GFLAGS_BUILD_TESTING OFF)
  add_subdirectory(${GFLAGS_SOURCE_DIR} ${GFLAGS_ROOT})
endfunction()

build_gflags()
include_directories (${GFLAGS_INCLUDE_DIR})
