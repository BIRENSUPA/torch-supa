# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# SUPA_PATH and BRCC_PATH
if(DEFINED ENV{BIREN_HOME})
  # fullstack environment
  set(SUPA_PATH $ENV{BIREN_HOME}/supa)
  set(BRCC_PATH $ENV{BIREN_HOME}/brcc)
else()
  # SDK environment
  set(SUPA_PATH /usr/local/birensupa/sdk/latest/supa)
  set(BRCC_PATH /usr/local/birensupa/sdk/latest/brcc)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/Modules_SUPA)
find_package(SUPABRCC)

if(NOT SUPA_FOUND)
    message(WARNING "Caffe2: SUPA cannot be found.")
    return()
endif()

if(SUPA_FOUND)
  if (NOT SUPA_ARCH)
    message(FATAL_ERROR "No SUPA ARCH found for building device code!!")
  else()
    message(STATUS "Build device code for ${SUPA_ARCH}")
  endif()

  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(SUPA_DEBUG_FLAGS "-O0 -g -D_DEBUG")
    if (ENABLE_COVERAGE)
      set(SUPA_DEBUG_FLAGS "${SUPA_DEBUG_FLAGS} --coverage")
    endif()
  else()
    set(SUPA_DEBUG_FLAGS "-O2 -DNDEBUG")
  endif()

  if ("${SUPA_ARCH}" MATCHES "br110")
    set(SUPA_DEBUG_FLAGS "${SUPA_DEBUG_FLAGS} -D__BR110_ARCH__")
  endif()

  set(SUPA_BRCC_FLAGS)
  # Use cxx17 standard
  append(SUPA_BRCC_FLAGS -std=c++17)
  # Open warning-as-error
  append(SUPA_BRCC_FLAGS -Werror)
  # Suppress unable to perform transformation warning
  append(SUPA_BRCC_FLAGS -Wno-pass-failed)
  # Propagate SUPA_DEBUG_FLAGS
  append(SUPA_BRCC_FLAGS "${SUPA_DEBUG_FLAGS}")
  # Disable relocatable device code generation
  append(SUPA_BRCC_FLAGS -fno-gpu-rdc)
  # Position Independent Code
  append(SUPA_BRCC_FLAGS -fPIC)
  # Specify device arch
  append(SUPA_BRCC_FLAGS "--supa-gpu-arch=${SUPA_ARCH}")
  # Propagate TORCH_CXX_FLAGS
  append(SUPA_BRCC_FLAGS "${TORCH_CXX_FLAGS}")
  # Propagate OpenMP Flags
  append(SUPA_BRCC_FLAGS "${OpenMP_CXX_FLAGS}")
  # Propagate mira Flags
  if ("$ENV{TORCH_USE_ISA}" STREQUAL "1")
    list(APPEND SUPA_BRCC_FLAGS "-use-mira=false")
  else()
    list(APPEND SUPA_BRCC_FLAGS "-use-mira=true")
  endif()
  list(APPEND SUPA_BRCC_FLAGS "-Wno-deprecated-builtins" "-fdeclspec" "-Wno-pass-failed" "-Wno-absolute-value")
  set(SUPA_INCLUDE_DIRS ${SUPA_PATH}/include)
  find_library(SUPA_SUPART_LIBRARY supart ${SUPA_PATH}/lib ${SUPA_PATH}/lib64)
  message("SUPA runtime lib: " ${SUPA_SUPART_LIBRARY})
endif()
