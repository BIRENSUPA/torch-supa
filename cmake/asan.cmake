# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# AddressSanitizer
# Usage: make asan
# See: https://en.wikipedia.org/wiki/AddressSanitizer
set(ASAN_FLAGS "-fsanitize=address -fsanitize-recover=address -fno-omit-frame-pointer")

add_custom_target(
  recover
  COMMAND cmake -DASAN="" ../..
)

add_custom_target(
  asan
  COMMAND make recover
  COMMAND cmake -DASAN=${ASAN_FLAGS} ../..
  COMMAND make -j 8
  COMMAND make install
  COMMAND make recover
  COMMAND mkdir -p ./asan/
  COMMAND bash -c "export ASAN_GTEST_FILTER=$(cat ${CMAKE_CURRENT_SOURCE_DIR}/cmake/filter.config); \
  ASAN_OPTIONS=halt_on_error=false:detect_leaks=1:log_path=./asan/asan-${CURRENT_TIME}.log ./test_brtorch --gtest_filter=$ASAN_GTEST_FILTER"
  VERBATIM
)
