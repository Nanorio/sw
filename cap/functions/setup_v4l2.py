#!/usr/bin/env python3
"""
卸载 → 重载 v4l2loopback (exclusive_caps=1)。
video_nr / card_label 从 config/capture.yaml 读取，避免硬编码。
** 注意modprobe需要 sudo 权限，如果需要sudo时不输入密码，请执行sudo visudo -f /etc/sudoers.d/v4l2loopback并添加nvidia ALL=(ALL) NOPASSWD: /sbin/modprobe
"""

import subprocess
import sys
import yaml
from pathlib import Path


def main():
    # ========== 从配置文件读取参数 ==========
    cfg_path = Path("./config/capture.yaml")
    virt_id = 2
    card_label = "CAM_virtual"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        v4l2_cfg = cfg.get("v4l2", {})
        virt_id = v4l2_cfg.get("virtualDeviceId", virt_id)
        card_label = v4l2_cfg.get("cardLabel", card_label)
    else:
        print(f"[setup_v4l2] 找不到 {cfg_path}，使用默认值")

    cmds = [
        ["modprobe", "-r", "v4l2loopback"],
        [
            "modprobe",
            "v4l2loopback",
            f"video_nr={virt_id}",
            f'card_label="{card_label}"',
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
