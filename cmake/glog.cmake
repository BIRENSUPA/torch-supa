# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

set(GLOG_VER           0.6.0)
set(GLOG_NAME          glog)
set(GLOG_FOLDER        ${GLOG_NAME}-${GLOG_VER})
set(GLOG_SOURCE_DIR    ${TORCH_SUPA_THIRD_PARTY_ROOT}/glog)
set(GLOG_ROOT          ${CMAKE_BINARY_DIR}/third-party/${GLOG_FOLDER})
set(GLOG_INCLUDE_DIR   ${GLOG_ROOT})
set(GLOG_LIB           $<TARGET_FILE:${GLOG_NAME}>)

function(build_glog)
  set(BUILD_SHARED_LIBS OFF)
  set(WITH_GFLAGS OFF)
  set(WITH_GTEST OFF)
  set(WITH_UNWIND OFF)
  add_subdirectory(${GLOG_SOURCE_DIR} ${GLOG_ROOT})
endfunction()

build_glog()
file(GLOB allcopyfiles "${GLOG_SOURCE_DIR}/src/glog/*.h")
file(COPY ${allcopyfiles} DESTINATION ${GLOG_INCLUDE_DIR}/glog)

include_directories (${GLOG_INCLUDE_DIR})
