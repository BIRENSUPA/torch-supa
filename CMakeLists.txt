cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
message(STATUS "CMake Version: ${CMAKE_VERSION}")

project(torch_supa CXX C)

if (POLICY CMP0135)
  cmake_policy(SET CMP0135 OLD)
endif()

set(LINUX TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(WERROR TRUE)
string(TIMESTAMP CURRENT_TIME "%Y-%m-%d-%H-%M-%S")

set(PLUGIN_NAME torch_supa)
set(CORE_SUPA c10_supa)
set(SUPA_FLASH_ATTENTION supa_flash_attention)
set(LIBSUPAOP supa_custom_ops)
set(LIBSUDAOP torch_suda_op)
set(SUPATARGET "")
set(BR_COMMON br_common)

set(RPATH_VALUE $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${RPATH_VALUE}/lib/:${RPATH_VALUE}/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TORCH_SUPA_INSTALL_LIBDIR})
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2 -s")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-terminate -Wno-error=terminate -Wno-attributes -Wno-error=attributes")
if(NOT TORCH_ABI_VER STREQUAL "0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=${TORCH_ABI_VER} -U__GXX_ABI_VERSION -D__GXX_ABI_VERSION=10${TORCH_ABI_VER}")
endif()
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++14. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BRCC_ENABLE_DEBUGGING_OPTIONS "Enable BRCC debugging options" OFF)
option(SUPA_ARCH_VERSION "BIRENSUPA Hardware Architecture Number. Also can set it by env 'VERSION'" OFF)
option(USE_SPLIT_SUPA "Seperate device code lib with host code lib." ON)
option(USE_PARALLEL_COMPILE "Compile supa kernel in parallel." OFF)
option(ENABLE_COVERAGE "Enable coverage reporting." OFF)
option(ENABLE_PERF_TOOLS "Enable Linux perf (sometimes called perf_events or perf tools)" OFF)
option(USE_BCCL "Build with BCCL" ON)
option(USE_FLASH_ATTENTION "Enable Flash Attention kernel for sdpa" ON)

set(CAFFE2_USE_SUDNN ON)

message(STATUS "PYTORCH_INSTALL_DIR: ${PYTORCH_INSTALL_DIR}")
set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
FIND_PACKAGE(Torch REQUIRED)

if (ENABLE_COVERAGE)
  message(STATUS "Coverage enabled.")
  add_compile_options(--coverage)
  add_link_options(--coverage)
endif()

set(SUPA_ARCH arch_20 CACHE STRING "Biren GPU Hardware Arch" FORCE)
if (DEFINED ENV{VERSION})
  set(SUPA_ARCH $ENV{VERSION})
endif()

if (SUPA_ARCH_VERSION)
  set(SUPA_ARCH ${SUPA_ARCH_VERSION})
endif()

set(TORCH_SUPA_ROOT "${PROJECT_SOURCE_DIR}/torch_supa/csrc")
set(TORCH_SUPA_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third-party")
set(TORCH_SUPA_INSTALL_INCLUDE "${TORCH_SUPA_INSTALL_LIBDIR}/../include")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY .)
include(cmake/utils.cmake)
include(cmake/fmt.cmake)
include(cmake/supa.cmake)
include(cmake/asan.cmake)
include(cmake/glog.cmake)
include(cmake/gflags.cmake)

set(LINUX TRUE)

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# must match existence of NDEBUG in installed pytorch, otherwise layout of class in ATen is mismatch.
IF("${PYTORCH_BUILD_CONFIG}" MATCHES ".* -DNDEBUG .*")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYTORCH_HAS_NDEBUG")
ENDIF()

# Define build type
IF(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
  IF(NOT "${ASAN}" STREQUAL "")
    message("asan: ${ASAN}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN}")
  ENDIF()
  IF(ENABLE_COVERAGE)
    message("Coverage Build with NDEBUG.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
  ENDIF()
ELSEIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("RelWithDebInfo build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ELSE()
  message("Release build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-overflow")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
if (CMAKE_COMPILER_IS_GNUCXX AND NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=redundant-decls")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=old-style-cast")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# These flags are not available in GCC-4.8.5. Set only when using clang.
# Compared against https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-partial-specialization")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-typedef-redefinition")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-aligned-allocation-unavailable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++17-extensions")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-constexpr-not-const")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qunused-arguments")
  if (${COLORIZE_OUTPUT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
  endif()
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
  if (${COLORIZE_OUTPUT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  endif()
endif()
if ((APPLE AND (NOT ("${CLANG_VERSION_STRING}" VERSION_LESS "9.0")))
  OR (CMAKE_COMPILER_IS_GNUCXX
  AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0 AND NOT APPLE)))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
endif()
if (NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finline-functions")
if (ENABLE_PERF_TOOLS)
  message(STATUS "option for Linux perf tools enabled.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
endif()

set(CMAKE_C_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
set(CXXFLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CXXFLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(SUDNN REQUIRED)
find_package(SUBLAS REQUIRED)
find_package(SUFFT REQUIRED)
find_package(SUTLASS REQUIRED)
find_package(BESU REQUIRED)
find_package(BRTX REQUIRED)

if (USE_BCCL)
  find_package(BCCL REQUIRED)
  add_definitions(-DUSE_BCCL)
else()
  set(USE_BCCL OFF)
  set(BCCL_LIB "")
endif()


# add more includes here
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PYTORCH_INSTALL_DIR}/include)
include_directories(${SUPA_INCLUDE_DIRS})
include_directories(${SUDNN_INCLUDE_PATH})
include_directories(${SUBLAS_INCLUDE_PATH})
include_directories(${SUFFT_INCLUDE_PATH})
include_directories(${SUTLASS_INCLUDE_PATH})
include_directories(${BRTX_INCLUDE_PATH})
if (BESU_HEADER_FOUND)
  include_directories(${BESU_INCLUDE_PATH})
endif()

# Set installed PyTorch dir
if(DEFINED PYTORCH_INSTALL_DIR)
  include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
else()
  message(FATAL_ERROR, "Cannot find installed PyTorch directory")
endif()

message(STATUS "Python Interpreter: ${PYTHON_EXECUTABLE}")
set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
find_package(Python3 COMPONENTS Interpreter Development)
if (NOT Python3_FOUND)
  message(FATAL_ERROR "No sutiable python library found.")
endif()
message(STATUS "Python lib: ${Python3_LIBRARIES};${Python3_LIBRARY_DIRS}")
message(STATUS "Python include: ${Python3_INCLUDE_DIRS}")
# drop libkineto.a from torch library to use the one in project third-party
list(FILTER TORCH_LIBRARIES EXCLUDE REGEX ".*libkineto\.a")
message(STATUS "Torch lib: ${TORCH_LIBRARIES}")
include_directories(${Python3_INCLUDE_DIRS})

# sources
set(CORE_SRCS)
set(ATEN_SRCS)
set(SUPA_SRCS)
set(UTILS_SRCS)
set(TEST_SRCS)
set(PROFILER_SRCS)
# Register files for torch_supa
set(REGISTER_SRCS)
set(ATEN_SUPA_SRCS)
set(DIST_SUPA_SRCS)
set(IPC_SRCS)

add_subdirectory(${TORCH_SUPA_ROOT}/aten)
add_subdirectory(${TORCH_SUPA_ROOT}/core)
add_subdirectory(${TORCH_SUPA_ROOT}/supa)
add_subdirectory(${TORCH_SUPA_ROOT}/utils)
add_subdirectory(${TORCH_SUPA_ROOT}/debugger)
if (USE_BCCL)
  add_subdirectory(${TORCH_SUPA_ROOT}/distributed)
endif()

add_subdirectory(${TORCH_SUPA_ROOT}/profiler)
add_subdirectory(${TORCH_SUPA_ROOT}/ipc)
add_subdirectory(${TORCH_SUPA_ROOT}/aten/ops/kernels/transformers)

if (USE_FLASH_ATTENTION)
  include_directories(${TORCH_SUPA_THIRD_PARTY_ROOT}/flash-attention/csrc/flash_attn)
  include_directories(${TORCH_SUPA_THIRD_PARTY_ROOT}/flash-attention/csrc/flash_attn/src)
  FILE(GLOB FLASH_ATTENTION_KERNELS_SU ${TORCH_SUPA_THIRD_PARTY_ROOT}/flash-attention/csrc/flash_attn/src/*.su)
  file(GLOB FLASH_ATTENTION_FP8_KERNELS "${TORCH_SUPA_THIRD_PARTY_ROOT}/flash-attention/csrc/flash_attn/src/*fp8*.su")
  foreach(fa_fp8_kernel IN LISTS FLASH_ATTENTION_FP8_KERNELS)
      list(REMOVE_ITEM FLASH_ATTENTION_KERNELS_SU ${fa_fp8_kernel})
  endforeach()

  FILE(GLOB FLASH_ATTENTION_KERNELS_CPP ${TORCH_SUPA_THIRD_PARTY_ROOT}/flash-attention/csrc/flash_attn/flash_api.cpp)
endif()

# supa kernels
if(USE_BCCL)
  list(APPEND ATEN_SUDA_SRCS ${DIST_SUPA_SRCS})
endif()
if ("$ENV{TORCH_USE_ISA}" STREQUAL "1")
  supa_add_library(${LIBSUPAOP} SHARED ${ATEN_SUPA_SRCS} BRCC_OPTIONS "-use-mira=false")
else()
  supa_add_library(${LIBSUPAOP} SHARED ${ATEN_SUPA_SRCS} BRCC_OPTIONS "-use-mira=true")
endif()
target_include_directories(${LIBSUPAOP} PRIVATE ${TORCH_SUPA_ROOT}/aten/ops/kernels/transformers/supa)

if (USE_FLASH_ATTENTION)
  set(FLASH_ATTENTION_BRCC_FLAGS
      "-use-mira=false"
      "-mllvm" "-br-preallocate-tensor-regs=0"
      "-mllvm" "-br-tcore-async-src-register=0"
      "-mllvm" "-vectorize-slp=false"
      "-mllvm" "-br-sfu-instr-count-per-sc-slot=2"
      "-mllvm" "-disable-licm=true"
      "-mllvm" "-br-allow-tensor-reg-spill=0"
      "-fgpu-rdc"
      "-fno-supa-prec-div")
  supa_add_library(${SUPA_FLASH_ATTENTION} SHARED ${FLASH_ATTENTION_KERNELS_SU} BRCC_OPTIONS ${FLASH_ATTENTION_BRCC_FLAGS})
  target_include_directories(${SUPA_FLASH_ATTENTION} PRIVATE ${TORCH_SUPA_ROOT}/aten/ops/kernels/transformers/supa)
  string(REPLACE ";" " " FLASH_ATTENTION_BRCC_FLAGS_STR "${FLASH_ATTENTION_BRCC_FLAGS}")
  set_property(TARGET ${SUPA_FLASH_ATTENTION} APPEND_STRING PROPERTY LINK_FLAGS " ${FLASH_ATTENTION_BRCC_FLAGS_STR}")
endif()

function(suda_target)
  include(cmake/suda.cmake)
  list(APPEND SUPA_BRCC_FLAGS "-Wno-macro-redefined" "-Wno-missing-exception-spec" "-Wno-unused-command-line-argument"
      "-Wno-ignored-pragmas" "-Wno-extra-tokens" "-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__" "-fvisibility=hidden" "-fvisibility-inlines-hidden")
  suda_add_library(${LIBSUDAOP} SHARED ${ATEN_SUDA_SRCS} OPTIONS WITH_SUPA)
  target_compile_definitions(${LIBSUDAOP} PRIVATE "-DHAVE_AVX512_CPU_DEFINITION")
  target_compile_definitions(${LIBSUDAOP} PRIVATE "-DHAVE_AVX2_CPU_DEFINITION")
  target_include_directories(${LIBSUDAOP} BEFORE PRIVATE ${FMT_INCLUDE_DIR} ${TORCH_SUPA_OP_DIR}/../include/)
  set_target_properties(${LIBSUDAOP} PROPERTIES LINKER_LANGUAGE SUPA)
  set_target_properties(${LIBSUDAOP} PROPERTIES LINK_FLAGS "--supa-link -O2 --supa-gpu-arch=${SUPA_ARCH}")
  target_link_options(${LIBSUDAOP} PRIVATE "-Wl,--exclude-libs,ALL")
  if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
    set(LIBSUDAOP_RODATA_TMP ${CMAKE_CURRENT_BINARY_DIR}/${LIBSUDAOP}.rodata.tmp)
    add_custom_command(TARGET ${LIBSUDAOP} POST_BUILD
      COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:${LIBSUDAOP}>
      COMMAND objcopy -O binary --only-section=.rodata $<TARGET_FILE:${LIBSUDAOP}> ${LIBSUDAOP_RODATA_TMP}
      COMMAND sed -i -f ${PROJECT_SOURCE_DIR}/cmake/rodata_replace.sed ${LIBSUDAOP_RODATA_TMP}
      COMMAND objcopy --update-section=.rodata=${LIBSUDAOP_RODATA_TMP} $<TARGET_FILE:${LIBSUDAOP}>
      VERBATIM)
  endif()
endfunction()
suda_target()

# Common so libs for all backends
set(COMMON_BE_DEPEND_LIBS ${SUDNN_LIB} ${SUBLAS_LIB} ${SUPA_LIB} ${SUPA_SUPART_LIBRARY} ${SUFFT_LIB} ${BCCL_LIB})

set(CPP_SRCS ${ATEN_SRCS} ${FLASH_ATTN_ADAPTATION_CPP_SRC} ${SUPA_SRCS} ${UTILS_SRCS} ${DIST_SRCS} ${PROFILER_SRCS} ${DEBUGGER_SRCS} ${FLASH_ATTENTION_KERNELS_CPP} ${IPC_SRCS})
if (DEFINED TORCH_SUPA_OP_DIR)
  set(CPP_SRCS ${CPP_SRCS} ${REGISTER_SRCS})
endif()

add_library(${PLUGIN_NAME} SHARED ${CPP_SRCS})
add_library(${CORE_SUPA} SHARED ${CORE_SRCS})
# replace by cmake config
link_directories(${PYTORCH_INSTALL_DIR}/lib)
find_library(C10_LIBRARY c10 PATHS ${PYTORCH_INSTALL_DIR}/lib)
find_library(TORCH_CPU_LIBRARY torch_cpu PATHS ${PYTORCH_INSTALL_DIR}/lib)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fvisibility=hidden" COMPILER_SUPPORTS_HIDDEN_VISIBILITY)
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(${CORE_SUPA} PRIVATE "-fvisibility=hidden")
  target_compile_options(${LIBSUPAOP} PRIVATE "-fvisibility=hidden")
  target_compile_options(${LIBSUDAOP} PRIVATE "-fvisibility=hidden" "-fvisibility-inlines-hidden")
  if (USE_FLASH_ATTENTION)
    target_compile_options(${SUPA_FLASH_ATTENTION} PRIVATE "-fvisibility=hidden")
  endif()
endif()

if (USE_FLASH_ATTENTION)
  # enable flash attention
  target_compile_definitions(${PLUGIN_NAME} PRIVATE "-DUSE_FLASH_ATTENTION")
endif()

if (USE_KINETO)
  # enable using kineto for profiler interface
  target_compile_definitions(${PLUGIN_NAME} PRIVATE "-DUSE_KINETO")
  add_subdirectory(${TORCH_SUPA_THIRD_PARTY_ROOT}/kineto/libkineto)
  include_directories(${LIBKINETO_INCLUDE})

  target_link_libraries(${PLUGIN_NAME} PUBLIC kineto)
  add_dependencies(${PLUGIN_NAME} kineto)
  file(COPY ${LIBKINETO_INCLUDE}/
        DESTINATION ${TORCH_SUPA_INSTALL_INCLUDE}/kineto/include)
endif()

target_link_libraries(${PLUGIN_NAME} PUBLIC ${LIBSUPAOP} ${LIBSUDAOP})
if (USE_FLASH_ATTENTION)
  target_link_libraries(${PLUGIN_NAME} PUBLIC ${SUPA_FLASH_ATTENTION})
endif()
target_link_libraries(${CORE_SUPA} PUBLIC ${C10_LIBRARY} ${LIBSUDAOP})
target_link_options(${PLUGIN_NAME} PRIVATE
    -Wl,--no-as-needed
    $<TARGET_FILE:${LIBSUDAOP}>
    -Wl,--as-needed
)

add_dependencies(${PLUGIN_NAME} ${LIBSUPAOP} ${CORE_SUPA})

target_link_libraries(${PLUGIN_NAME} PUBLIC ${FMT_NAME} ${GLOG_LIB} ${GFLAGS_LIB})

if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
  message(FATAL_ERROR "_GLIBCXX_USE_CXX11_ABI not defined")
endif()
string(APPEND CMAKE_CXX_FLAGS
       " -D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}")
if(${_GLIBCXX_USE_CXX11_ABI} EQUAL 1)
  set(CXX_STANDARD_REQUIRED ON)
else()
  # Please note this is required in order to ensure compatibility between gcc
  # 9 and gcc 7 This could be removed when all Linux PyTorch binary builds are
  # compiled by the same toolchain again
  append_cxx_flag_if_supported("-fabi-version=11" CMAKE_CXX_FLAGS)
endif()

# using same compile defintions with pytorch for call xxx_stub
target_compile_definitions(${PLUGIN_NAME} PRIVATE "-DHAVE_AVX512_CPU_DEFINITION")
target_compile_definitions(${PLUGIN_NAME} PRIVATE "-DHAVE_AVX2_CPU_DEFINITION")
if(DEFINED TORCH_SUPA_OP_DIR)
  target_compile_definitions(${PLUGIN_NAME} PRIVATE "-DTORCH_SUPA_OP_DIR")
  find_library(SUPAOP_LIBRARY torch_supa_op PATHS ${TORCH_SUPA_OP_DIR})
  message(STATUS "Torch-supa-op lib: ${TORCH_SUPA_OP_DIR} : ${SUPAOP_LIBRARY}")

  target_link_libraries(${PLUGIN_NAME} PUBLIC
    -Wl,--start-group
    ${CORE_SUPA}
    ${SUPAOP_LIBRARY}
    -Wl,--end-group
    ${TORCH_CPU_LIBRARY}
    ${Python3_LIBRARIES}
    ${SUPTI_LIBRARY_PATH}
    ${COMMON_BE_DEPEND_LIBS})
  file(GLOB_RECURSE HEADERS
      ${TORCH_SUPA_OP_DIR}/../include/*.h
      ${TORCH_SUPA_OP_DIR}/../include/*.cuh
  )
  foreach(HEADER  ${HEADERS})
    string(REPLACE "${TORCH_SUPA_OP_DIR}/../include/" "" HEADER_SUB ${HEADER})
    get_filename_component(DIR ${HEADER_SUB} DIRECTORY)
    file(COPY ${HEADER} DESTINATION "${TORCH_SUPA_INSTALL_INCLUDE}/${DIR}")
  endforeach()
else()
message(STATUS "build without torch_supa_op")
  target_link_libraries(${PLUGIN_NAME} PUBLIC ${CORE_SUPA} ${TORCH_LIBRARY} ${Python3_LIBRARIES} ${SUPTI_LIBRARY_PATH} ${COMMON_BE_DEPEND_LIBS})
endif()


# add c++ unit test
set(GTEST_ROOT ${TORCH_SUPA_THIRD_PARTY_ROOT}/gtest-1.10.0)

set(TEST_TARGET test_torch_supa)
add_subdirectory(${TORCH_SUPA_ROOT}/test)
add_executable(${TEST_TARGET} ${TEST_SRCS})
target_include_directories(${TEST_TARGET} PUBLIC ${GTEST_ROOT} ${GTEST_ROOT}/include)
add_dependencies(${TEST_TARGET} ${PLUGIN_NAME})
set_target_properties(${TEST_TARGET} PROPERTIES LINK_FLAGS "-O2")
target_link_libraries(${TEST_TARGET} PUBLIC ${PLUGIN_NAME} ${TORCH_LIBRARIES} pthread)
# enable for test the number of undefined symbols
# target_link_libraries(${TEST_TARGET} PUBLIC -Wl,--no-as-needed ${PLUGIN_NAME} ${TORCH_LIBRARIES})
