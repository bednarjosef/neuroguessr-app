#!/usr/bin/env bash
# Sweep encoder configurations on the attached device and print one line per config.
# The encoder dominates end-to-end latency, so this is the measurement that decides
# what the app ships with.
set -u
export PATH=$PATH:$HOME/Android/Sdk/platform-tools
PKG=com.neuroguessr.app
ITERS=${ITERS:-3}

run() {
  local model=$1 threads=$2 ep=$3
  adb shell am force-stop $PKG
  adb logcat -c
  adb shell am start -n $PKG/.MainActivity \
      --ei bench $ITERS --es model "$model" --ei threads "$threads" --es ep "$ep" >/dev/null 2>&1
  local waited=0
  while [ $waited -lt 900 ]; do
    if adb logcat -d -s NGSELFTEST:* 2>/dev/null | grep -q "SELFTEST_DONE\|SELFTEST_FAIL"; then break; fi
    sleep 5; waited=$((waited+5))
  done
  local line
  line=$(adb logcat -d -s NGSELFTEST:* 2>/dev/null | grep -m1 "BENCH ")
  if [ -z "$line" ]; then
    line=$(adb logcat -d -s NGSELFTEST:* 2>/dev/null | grep -m1 "SELFTEST_FAIL")
  fi
  printf '%-22s t%-2s %-8s -> %s\n' "$model" "$threads" "$ep" "${line#*NGSELFTEST: }"
}

echo "=== encoder benchmark (${ITERS} iterations each) ==="
for cfg in "$@"; do
  IFS=: read -r m t e <<< "$cfg"
  run "$m" "$t" "$e"
done
