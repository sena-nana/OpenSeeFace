#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cargo build --release --manifest-path "$ROOT/runtime-ort/Cargo.toml" --bin facetracker
mkdir -p "$ROOT/dist/facetracker"
BIN="$ROOT/runtime-ort/target/release/facetracker"
if [[ -f "$BIN.exe" ]]; then
  BIN="$BIN.exe"
fi
cp "$BIN" "$ROOT/dist/facetracker/"
cp -R "$ROOT/models" "$ROOT/dist/facetracker/"
if [[ -f "$ROOT/run.bat" ]]; then
  cp "$ROOT/run.bat" "$ROOT/dist/facetracker/"
fi
echo "Files are in dist/facetracker/"
