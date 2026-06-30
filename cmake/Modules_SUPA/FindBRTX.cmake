# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

set(BRTX_FOUND FALSE)
if(BRTX_FOUND)
    return()
endif()

if(DEFINED ENV{BIREN_HOME})
    set(BRTX_PATH $ENV{BIREN_HOME}/supti)
else()
    set(BRTX_PATH /usr/local/birensupa/sdk/latest/supti)
endif()

find_path(BRTX_INCLUDE_ROOT NAMES brtx/brToolsExt.h HINTS ${BRTX_PATH}/include)
if(BRTX_INCLUDE_ROOT)
    set(BRTX_INCLUDE_PATH ${BRTX_INCLUDE_ROOT} ${BRTX_INCLUDE_ROOT}/brtx)
endif()

if(BRTX_INCLUDE_PATH)
    set(BRTX_FOUND True CACHE INTERNAL "indicates whether BRTX is found")
endif()

message(STATUS "BRTX_INCLUDE_PATH: ${BRTX_INCLUDE_PATH}")
if(NOT BRTX_FOUND)
    message(WARNING "Cannot find BRTX, Please set env: BRTX_PATH, SUPTI_PATH, BIREN_HOME or Install BRTX under /usr/local/birensupa/sdk/latest/supti")
endif()
