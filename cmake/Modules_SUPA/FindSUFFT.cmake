# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

set(SUFFT_FOUND FALSE)
if(SUFFT_FOUND)
    return()
endif()

if(DEFINED ENV{BIREN_HOME})
    # fullstack environment
    set(SUFFT_PATH $ENV{BIREN_HOME}/sufft)
else()
    # SDK environment
    set(SUFFT_PATH /usr/local/birensupa/sdk/latest/sufft)
endif()

find_path(SUFFT_INCLUDE_PATH NAMES sufft.h HINTS ${SUFFT_PATH}/include)
find_library(SUFFT_LIB NAMES sufft HINTS ${SUFFT_PATH}/lib ${SUFFT_PATH}/lib64)

message(STATUS "SUFFT_INCLUDE_PATH: ${SUFFT_INCLUDE_PATH}")
message(STATUS "SUFFT_LIB: ${SUFFT_LIB}")
if(SUFFT_INCLUDE_PATH AND SUFFT_LIB)
    set(SUFFT_FOUND True CACHE INTERNAL "indicates whether SUFFT is found")
endif()

if(NOT SUFFT_FOUND)
    message(FATAL_ERROR "Cannot find SUFFT, Please set env: BIREN_HOME or Install SUFFT under /usr/")
endif()
