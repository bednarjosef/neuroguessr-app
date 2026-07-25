#!/usr/bin/env bash
# Install NeuroGuessr on a device and push the index + encoder it needs.
#
# Order matters: a FIRST install wipes the app's external files directory, so the 2.2 GB of
# assets have to go on AFTER the APK, never before. Re-installs over an existing app leave
# the assets alone, so this is safe to re-run.
#
# Usage: provision_device.sh [serial]      (serial optional when only one device is attached)
set -euo pipefail
export PATH=$PATH:$HOME/Android/Sdk/platform-tools
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG=com.neuroguessr.app
DEST=/sdcard/Android/data/$PKG/files
APK="$REPO/mobile/android/app/build/outputs/apk/debug/app-debug.apk"
ASSETS="$REPO/mobile/assets_build"

ADB=(adb)
[ $# -ge 1 ] && ADB=(adb -s "$1")

echo "== device =="
"${ADB[@]}" shell getprop ro.product.model | tr -d '\r'
echo -n "android "; "${ADB[@]}" shell getprop ro.build.version.release | tr -d '\r'
echo -n "abi "; "${ADB[@]}" shell getprop ro.product.cpu.abi | tr -d '\r'

FREE=$("${ADB[@]}" shell df /sdcard | tail -1 | awk '{print $4}')
echo "free space: $((FREE/1024/1024)) GB"
if [ "$FREE" -lt 3500000 ]; then
  echo "!! needs ~3 GB free; aborting" >&2
  exit 1
fi

echo "== installing apk =="
"${ADB[@]}" install -r "$APK"

echo "== pushing index (1.55 GB) =="
"${ADB[@]}" shell mkdir -p "$DEST/index" "$DEST/model" "$DEST/testdata"
"${ADB[@]}" push "$ASSETS/index/." "$DEST/index/"

echo "== pushing encoder (652 MB) =="
"${ADB[@]}" push "$ASSETS/model/encoder_fp16.onnx" "$DEST/model/"

if [ -d "$REPO/mobile/testdata" ]; then
  "${ADB[@]}" push "$REPO/mobile/testdata/." "$DEST/testdata/" >/dev/null
  echo "== pushed test images =="
fi

echo "== verifying =="
"${ADB[@]}" shell du -sh "$DEST"/* | tr -d '\r'
echo
echo "Done. Launch it from the app drawer, or:"
echo "  adb ${1:+-s $1 }shell am start -n $PKG/.MainActivity"
