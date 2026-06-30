# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# Google cpplint
# Usage: make cpplint

set(LINT_FILTER)
set(LINT_FILTER ${LINT_FILTER}-,)
set(LINT_FILTER ${LINT_FILTER}+build,-build/c++17,-build/include_subdir,)
set(LINT_FILTER ${LINT_FILTER}+readability,)
set(LINT_FILTER ${LINT_FILTER}+runtime,-runtime/references,)
set(LINT_FILTER ${LINT_FILTER}+whitespace,)

find_package(Python3 COMPONENTS Interpreter Development)

if(Python3_Interpreter_FOUND)
    add_custom_target(
        cpplint
        COMMAND cmake ../..
        COMMAND mkdir -p ./cpplint/
        COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/third-party/cpplint.py"
        --exclude=../torch_supa/csrc/aten/Register*.cpp
        --exclude=../torch_supa/csrc/interface/python_custom_functions.cpp
        --linelength=120
        --filter=${LINT_FILTER}
        ${CPP_SRCS} > "./cpplint/cpplint-${CURRENT_TIME}.log" 2>&1
        VERBATIM
    )
endif()
