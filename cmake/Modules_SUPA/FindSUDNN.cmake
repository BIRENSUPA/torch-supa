# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

set(SUDNN_FOUND FALSE)
if(CAFFE2_USE_SUDNN)
    if(SUDNN_FOUND)
        return()
    endif()

    if(DEFINED ENV{BIREN_HOME})
        # fullstack environment
        set(SUDNN_PATH $ENV{BIREN_HOME}/sudnn)
    else()
        # SDK environment
        set(SUDNN_PATH /usr/local/birensupa/sdk/latest/sudnn)
    endif()

    find_path(SUDNN_INCLUDE_PATH NAMES sudnn.h HINTS ${SUDNN_PATH}/include/sudnn)
    find_library(SUDNN_LIB NAMES sudnn HINTS ${SUDNN_PATH}/lib ${SUDNN_PATH}/lib64)

    if(SUDNN_INCLUDE_PATH AND SUDNN_LIB)
        set(SUDNN_FOUND True CACHE INTERNAL "indicates whether SUDNN is found")
    endif()

    message(STATUS "SUDNN_INCLUDE_PATH: ${SUDNN_INCLUDE_PATH}")
    message(STATUS "SUDNN_LIB: ${SUDNN_LIB}")
    if(NOT SUDNN_FOUND)
        message(FATAL_ERROR "Cannot find SUDNN, Please set env: BIREN_HOME or Install SUDNN under /usr/")
    endif()
endif()
