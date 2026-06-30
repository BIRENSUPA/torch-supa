# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.


if(BCCL_FOUND)
    return()
endif()

if(DEFINED ENV{BIREN_HOME})
    # fullstack environment
    set(BCCL_PATH $ENV{BIREN_HOME}/bccl)
else()
    # SDK environment
    set(BCCL_PATH /usr/local/birensupa/sdk/latest/bccl)
endif()

find_path(BCCL_INCLUDE_PATH NAMES bccl.h HINTS ${BCCL_PATH}/include)
find_library(BCCL_LIB NAMES bccl HINTS ${BCCL_PATH}/lib)

if(BCCL_INCLUDE_PATH AND BCCL_LIB)
    set(BCCL_FOUND True CACHE INTERNAL "indicates whether BCCL is found")
endif()

message(STATUS "BCCL_INCLUDE_PATH: ${BCCL_INCLUDE_PATH}")
message(STATUS "BCCL_LIB: ${BCCL_LIB}")
if(NOT BCCL_FOUND)
    message(FATAL_ERROR "Cannot find BCCL, Please set env: BIREN_HOME or Install BCCL under /usr/")
endif()
