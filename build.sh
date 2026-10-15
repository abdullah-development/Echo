#!/bin/sh
# Builds the native system-audio capture helper.
set -e
cd "$(dirname "$0")"
mkdir -p bin
swiftc -O -swift-version 5 -o bin/audiotap audiotap/main.swift -framework ScreenCaptureKit -framework CoreMedia
echo "Built bin/audiotap"
