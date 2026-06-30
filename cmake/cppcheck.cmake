# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# additional target to perform cppcheck run, requires cppcheck

add_custom_target(
        cppcheck
        COMMENT "Running cppcheck, wait a few minutes..."
        COMMAND cmake ../..
        COMMAND mkdir -p ./cppcheck/
        COMMAND cppcheck
        --enable=all
        --std=c++17
        --language=c++
        --template="[{severity}][{id}] {message} {callstack} \(On {file}:{line}\)"
        --verbose
        --output-file="./cppcheck/cppcheck-${CURRENT_TIME}.log"
        --inline-suppr
        --quiet
        -D __GNUC__
        # header folders
        -I ${CMAKE_CURRENT_SOURCE_DIR}/torch_supa
        # exclude folders
        -i ${CMAKE_CURRENT_SOURCE_DIR}/third-party
        -i ${CMAKE_CURRENT_SOURCE_DIR}/build
        ${CPP_SRCS}
)
