#!/usr/bin/env bash

# This script is only to format the changed files you will commit

if [ -z "$1" ]; then
  # Default format staged
  ALL_COMMITED_FILES=$(git status -s | grep '^[AM]' | awk '{printf("%s\n", $2)}' | grep '\(\.h\|\.hpp\|\.cpp\|\.suh\|\.su\)$')
else
  # Or format diff with specific commit
  ALL_COMMITED_FILES=$(git diff --stat $1 | grep '|' | awk '{printf("%s\n", $1)}' | grep '\(\.h\|\.hpp\|\.cpp\|\.suh\|\.su\)$')
fi

if [ ! -z "$ALL_COMMITED_FILES" ]; then
  clang-format -style=file -i ${ALL_COMMITED_FILES}
fi
