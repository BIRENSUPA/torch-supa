# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.


set(SUTLASS_FOUND FALSE)
if(SUTLASS_FOUND)
    return()
endif()

if(DEFINED ENV{BIREN_HOME})
    # fullstack environment
    #set(SUTLASS_PATH $ENV{BIREN_HOME}/sutlass)
    set(SUTLASS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third-party/sutlass)
else()
    # SDK environment
    set(SUTLASS_PATH /usr/local/birensupa/sdk/latest/sutlass)
endif()

find_path(SUTLASS_INCLUDE_PATH NAMES sutlass_header.hpp HINTS ${SUTLASS_PATH}/include/ NO_DEFAULT_PATH)
#find_library(SUTLASS_LIB NAMES sutlass HINTS ${SUTLASS_PATH}/lib ${SUTLASS_PATH}/lib64)

#if(SUTLASS_INCLUDE_PATH AND SUTLASS_LIB)
if(SUTLASS_INCLUDE_PATH)
    set(SUTLASS_FOUND True CACHE INTERNAL "indicates whether SUTLASS is found")
endif()

message(STATUS "SUTLASS_INCLUDE_PATH: ${SUTLASS_INCLUDE_PATH}")
#message(STATUS "SUTLASS_LIB: ${SUTLASS_LIB}")
if(NOT SUTLASS_FOUND)
    message(FATAL_ERROR "Cannot find SUTLASS, Please set env: BIREN_HOME or Install SUTLASS under /usr/")
endif()

# copy header to install folder for packaging whl
file(COPY ${SUTLASS_INCLUDE_PATH} DESTINATION ${TORCH_SUPA_INSTALL_INCLUDE}/sutlass/)
