# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

set(SUBLAS_FOUND FALSE)
if(SUBLAS_FOUND)
    return()
endif()

if(DEFINED ENV{BIREN_HOME})
    # fullstack environment
    set(SUBLAS_PATH $ENV{BIREN_HOME}/sublas)
else()
    # SDK environment
    set(SUBLAS_PATH /usr/local/birensupa/sdk/latest/sublas)
endif()

find_path(SUBLAS_INCLUDE_PATH NAMES sublas.h sublasLt.h sublasExport.h HINTS ${SUBLAS_PATH}/include)
find_library(SUBLAS_LIB NAMES sublas HINTS ${SUBLAS_PATH}/lib ${SUBLAS_PATH}/lib64)

if(SUBLAS_INCLUDE_PATH AND SUBLAS_LIB)
    set(SUBLAS_FOUND True CACHE INTERNAL "indicates whether SUBLAS is found")
endif()

message(STATUS "SUBLAS_INCLUDE_PATH: ${SUBLAS_INCLUDE_PATH}")
message(STATUS "SUBLAS_LIB: ${SUBLAS_LIB}")
if(NOT SUBLAS_FOUND)
    message(FATAL_ERROR "Cannot find SUBLAS, Please set env: BIREN_HOME or Install SUBLAS under /usr/")
endif()
