#!/usr/bin/env python3
"""
独立脚本: 每次启动 main.py 之前执行一次。
卸载 → 重载 v4l2loopback (exclusive_caps=1)，确保虚拟设备格式协商正常。

用法:
  python3 setup_v4l2.py
  python3 main.py

可以合在一行（只需输一次 sudo 密码）:
  python3 setup_v4l2.py && python3 main.py
"""

import subprocess
import sys


def main():
    cmds = [
        ["modprobe", "-r", "v4l2loopback"],
        [
            "modprobe",
            "v4l2loopback",
            "video_nr=2",
            'card_label="CAM_virtual"',
            "exclusive_caps=1",
        ],
    ]

    for cmd in cmds:
        print(f">>> [sudo] {' '.join(cmd)}")
        result = subprocess.run(
            ["sudo"] + cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # 如果 -r 时模块未加载会报错，可以忽略
            if cmd[0] == "modprobe" and cmd[1] == "-r":
                if "not currently loaded" in result.stderr or "not found" in result.stderr:
                    print("    (模块未加载，跳过卸载)")
                    continue
            print(f"[ERROR] {result.stderr.strip()}")
            sys.exit(result.returncode)
        if result.stderr.strip():
            print(f"    {result.stderr.strip()}")
    print(">>> 虚拟设备就绪，可以启动 main.py")


if __name__ == "__main__":
    main()
