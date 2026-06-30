# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

set(FMT_VER           8.1.1)
set(FMT_NAME          fmt)
set(FMT_FOLDER        ${FMT_NAME}-${FMT_VER})
set(FMT_SOURCE_DIR    ${TORCH_SUPA_THIRD_PARTY_ROOT}/fmt)
set(FMT_ROOT          ${CMAKE_BINARY_DIR}/third-party/${FMT_FOLDER})
set(FMT_INCLUDE_DIR   ${FMT_ROOT}/include)
set(FMT_LIB           $<TARGET_FILE:${FMT_NAME}>)

function(build_fmt)
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
  set(CMAKE_CXX_VISIBILITY_PRESET default)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
  set(CMAKE_CXX_FLAGS "${TORCH_CXX_FLAGS}")
  add_subdirectory(${FMT_SOURCE_DIR} ${FMT_ROOT})
endfunction()

build_fmt()
file(GLOB allcopyfiles "${FMT_SOURCE_DIR}/include/*")
file(COPY ${FMT_SOURCE_DIR}/include/fmt DESTINATION ${FMT_INCLUDE_DIR})
file(COPY ${allcopyfiles} DESTINATION ${TORCH_SUPA_INSTALL_INCLUDE}/fmt/include)

include_directories (${FMT_INCLUDE_DIR})
