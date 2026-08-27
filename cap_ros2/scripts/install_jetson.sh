#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"

sudo apt-get update
sudo apt-get install -y \
  python3-colcon-common-extensions \
  python3-opencv \
  python3-numpy \
  python3-yaml

pip3 install --user ultralytics

cd "${WS_DIR}"
colcon build --symlink-install
echo "Build complete. Run: source ${WS_DIR}/install/setup.bash"
