#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -f "$BUILD_DIR/generation_test"; rmdir "$BUILD_DIR"' EXIT
g++ -std=c++17 -O1 -g -pthread -fsanitize=address -fsanitize-address-use-after-scope \
  -I"$ROOT/tests/native/stubs" "$ROOT/tests/native/generation_test.cpp" -o "$BUILD_DIR/generation_test"
"$BUILD_DIR/generation_test" "${1:-tokens}"
