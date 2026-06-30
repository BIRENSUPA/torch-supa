# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.


if(BESU_FOUND)
    return()
endif()

if(DEFINED ENV{BESU_PATH})
    set(BESU_PATH $ENV{BESU_PATH})
elseif(DEFINED ENV{BIREN_HOME})
    # fullstack environment
    set(BESU_PATH $ENV{BIREN_HOME}/brumd)
else()
    # SDK environment
    set(BESU_PATH /usr/local/birensupa/sdk/latest/brumd)
endif()

find_path(BESU_INCLUDE_PATH NAMES besu.h HINTS ${BESU_PATH}/include)
find_library(BESU_LIB NAMES besu HINTS ${BESU_PATH}/lib ${BESU_PATH}/lib64)
find_library(SUPA_LIB NAMES supa HINTS ${BESU_PATH}/lib ${BESU_PATH}/lib64)

if(BESU_INCLUDE_PATH AND BESU_LIB)
    set(BESU_HEADER_FOUND True CACHE INTERNAL "indicates whether BESU is found")
    set(BESU_FOUND True CACHE INTERNAL "indicates whether BESU is found")
endif()

message(STATUS "BESU_INCLUDE_PATH: ${BESU_INCLUDE_PATH}")
message(STATUS "BESU_LIB: ${BESU_LIB}")
message(STATUS "SUPA_LIB: ${SUPA_LIB}")
if(NOT BESU_HEADER_FOUND)
    message(WARNING " Cannot find BESU, Please set env: BESU_PATH or Install besu under $BIREN_HOME")
endif()
