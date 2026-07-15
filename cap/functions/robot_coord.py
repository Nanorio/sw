"""
机器人坐标系变换。
从 camera_setup.yaml 加载相机位姿，提供 相机系 → 绝对系 坐标变换。
"""

import yaml
import numpy
from pathlib import Path


class RobotCoord:
    """加载相机配置，提供坐标变换和视锥计算。"""

    def __init__(self, config_path: str = "./config/camera_setup.yaml"):
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"找不到相机配置：{config_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.active_cam_id: int = cfg["activeCameraId"]
        self.cameras: dict = cfg["cameras"]

        # 预计算所有相机的旋转矩阵 R（相机系 → 绝对系）
        self._R: dict[int, numpy.ndarray] = {}
        for cid_str, cdef in self.cameras.items():
            cid = int(cid_str)
            look = numpy.array(cdef["look_dir"], dtype=float)
            up = numpy.array(cdef["up_dir"], dtype=float)

            z_axis = look / numpy.linalg.norm(look)
            x_axis = numpy.cross(up, z_axis)
            x_norm = numpy.linalg.norm(x_axis)
            if x_norm < 1e-8:
                x_axis = numpy.array([1.0, 0.0, 0.0])
            else:
                x_axis = x_axis / x_norm
            y_axis = numpy.cross(z_axis, x_axis)

            self._R[cid] = numpy.column_stack([x_axis, y_axis, z_axis])

    def camera_to_abs(self, cam_id: int, point_cam_mm: numpy.ndarray) -> numpy.ndarray:
        """
        将目标点从相机坐标系变换到机器人绝对坐标系。

        参数:
            cam_id: 相机编号
            point_cam_mm: [Xc, Yc, Zc] 相机坐标系（Y 向下），单位 mm

        返回:
            [Xa, Ya, Za] 绝对坐标系（Y 向上），单位 mm
        """
        R = self._R[cam_id]
        T = numpy.array(self.cameras[int(cam_id)]["position_mm"], dtype=float)

        p_adj = numpy.array([
            point_cam_mm[0],
            -point_cam_mm[1],
            point_cam_mm[2],
        ], dtype=float)

        return R @ p_adj + T

    def get_frustum_verts(self, cam_id: int, fx: float, fy: float,
                          img_w: int = 640, img_h: int = 480,
                          max_dist_mm: float = 10000.0):
        """
        计算相机视锥的顶点（绝对坐标系），用于 3D 可视化。

        返回:
            apex_abs: [3] ndarray，视锥顶点（相机位置）
            base_abs: [4, 3] ndarray，视锥底部四个角
        """
        R = self._R[cam_id]
        T = numpy.array(self.cameras[int(cam_id)]["position_mm"], dtype=float)

        dx = max_dist_mm * (img_w / (2.0 * fx))
        dy = max_dist_mm * (img_h / (2.0 * fy))

        apex_cam = numpy.array([[0.0, 0.0, 0.0]])
        base_cam = numpy.array([
            [-dx, -dy, max_dist_mm],
            [+dx, -dy, max_dist_mm],
            [+dx, +dy, max_dist_mm],
            [-dx, +dy, max_dist_mm],
        ])

        def _to_abs(pts: numpy.ndarray) -> numpy.ndarray:
            result = []
            for p in pts:
                p_adj = numpy.array([p[0], -p[1], p[2]])
                result.append(R @ p_adj + T)
            return numpy.array(result)

        return _to_abs(apex_cam)[0], _to_abs(base_cam)
