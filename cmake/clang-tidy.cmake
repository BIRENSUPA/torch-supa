# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# additional target to perform clang-tidy run, requires clang-tidy-15

add_custom_target(
    clangtidy
    COMMENT "Running clangtidy, wait a few minutes..."
    COMMAND cmake ../..
    COMMAND mkdir -p ./clangtidy/
    COMMAND run-clang-tidy-10
    -config=''
    -export-fixes "./clangtidy/clangtidy-${CURRENT_TIME}.fix"
    -header-filter '.*torch_supa.*'
    -j 8
    ${CPP_SRCS} > "./clangtidy/clangtidy-${CURRENT_TIME}.log" 2>&1
)
