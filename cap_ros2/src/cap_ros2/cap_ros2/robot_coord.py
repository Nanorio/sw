"""Robot coordinate transform copied into the ROS2 workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import yaml


class RobotCoord:
    def __init__(self, config_path: Union[str, Path]):
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Camera setup not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        self.active_cam_id = int(cfg["activeCameraId"])
        self.cameras = cfg["cameras"]
        self._R = {}
        for cid_str, cdef in self.cameras.items():
            cid = int(cid_str)
            look = np.array(cdef["look_dir"], dtype=float)
            up = np.array(cdef["up_dir"], dtype=float)
            z_axis = look / np.linalg.norm(look)
            x_axis = np.cross(up, z_axis)
            x_norm = np.linalg.norm(x_axis)
            x_axis = x_axis / x_norm if x_norm > 1e-8 else np.array([1.0, 0.0, 0.0])
            y_axis = np.cross(z_axis, x_axis)
            self._R[cid] = np.column_stack([x_axis, y_axis, z_axis])

    def camera_to_abs(self, cam_id: int, point_cam_mm: np.ndarray) -> np.ndarray:
        r = self._R[cam_id]
        t = np.array(self.cameras[int(cam_id)]["position_mm"], dtype=float)
        p_adj = np.array(
            [point_cam_mm[0], -point_cam_mm[1], point_cam_mm[2]], dtype=float
        )
        return r @ p_adj + t
