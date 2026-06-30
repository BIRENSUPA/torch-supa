# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

FILE(GLOB FLASH_ATTN_ADAPTATION_CPP_SRC
    *.cpp
    supa/flash_attn/*.cpp
    supa/sudnn_attn/*.cpp
    supa/*.cpp
    )

set(FLASH_ATTN_ADAPTATION_CPP_SRC ${FLASH_ATTN_ADAPTATION_CPP_SRC} PARENT_SCOPE)
