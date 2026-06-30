# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# FindSUDA.cmake
# -----------------
#
# Find SUDA library by querying the Python suda package for cmake_prefix_path.
#
# This module will:
#   1. Execute Python to get suda.cmake_prefix_path
#   2. Add the path to CMAKE_PREFIX_PATH
#   3. Call find_package(SUDA REQUIRED)
#
# Usage:
#   include(cmake/suda.cmake)
#   # After this, SUDA is available for use

# Check if SUDA was already found
if(NOT SUDA_FOUND)
  # Find Python first if not already found
  if(NOT Python_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(Python_EXECUTABLE ${Python3_EXECUTABLE})
  endif()

  # Query suda package for cmake_prefix_path
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import suda;print(suda.cmake_prefix_path)"
    OUTPUT_VARIABLE SUDA_PREFIX_PATH
    ERROR_QUIET
    RESULT_VARIABLE SUDA_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  if(NOT SUDA_RESULT EQUAL 0 OR SUDA_PREFIX_PATH STREQUAL "")
    message(FATAL_ERROR "Failed to get SUDA_PREFIX_PATH. "
      "Please ensure the suda package is installed. \n")
  endif()

  message(STATUS "SUDA prefix path: ${SUDA_PREFIX_PATH}")

  list(APPEND CMAKE_PREFIX_PATH ${SUDA_PREFIX_PATH})
endif()

# Find SUDA package
find_package(SUDA REQUIRED)

list(APPEND SUPA_BRCC_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
list(APPEND SUPA_BRCC_FLAGS "-Wno-sign-compare")
list(APPEND SUPA_BRCC_FLAGS "-Wno-attributes")
list(APPEND SUPA_BRCC_FLAGS "-Wno-error=attributes")
list(APPEND SUPA_BRCC_FLAGS "-Wno-unused-variable")
list(APPEND SUPA_BRCC_FLAGS "-Wno-unused-local-typedefs")
list(APPEND SUPA_BRCC_FLAGS "-Wno-absolute-value")

