#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=:99

if xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
  echo "Xvfb already running"
  exit 0
fi

# Clean stale locks/sockets. Ignore permission/read-only filesystem failures.
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true
sudo rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true
mkdir -p /tmp/.X11-unix 2>/dev/null || true
sudo mkdir -p /tmp/.X11-unix 2>/dev/null || true
chmod 1777 /tmp/.X11-unix 2>/dev/null || true
sudo chmod 1777 /tmp/.X11-unix 2>/dev/null || true
chown root:root /tmp/.X11-unix 2>/dev/null || true
sudo chown root:root /tmp/.X11-unix 2>/dev/null || true

Xvfb :99 -screen 0 1280x720x24 -ac -nolisten tcp +extension GLX +render -noreset >/tmp/xvfb_99.log 2>&1 &
sleep 1

if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
  echo "Failed to start Xvfb on DISPLAY=${DISPLAY}" >&2
  exit 1
fi

# force stable software renderer (llvmpipe)
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export GALLIUM_DRIVER=llvmpipe

echo "Xvfb+llvmpipe ready on DISPLAY=$DISPLAY"
