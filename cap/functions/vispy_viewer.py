"""
3D 可视化模块：机器人 + 相机视锥 + 目标位置。
使用 matplotlib 3D（兼容 Jetson ARM64）。
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import yaml
from pathlib import Path
from .robot_coord import RobotCoord


class Robot3DViewer:
    """3D 场景，显示机器人、相机视锥和目标位置。鼠标可拖拽旋转。"""

    def __init__(self, robot_coord: RobotCoord, fx: float, fy: float,
                 img_w: int = 640, img_h: int = 480,
                 max_dist_mm: float = 10000.0):
        self._robot = robot_coord
        self._cam_id = robot_coord.active_cam_id
        self._max_dist_mm = max_dist_mm
        self._fx, self._fy = fx, fy
        self._img_w, self._img_h = img_w, img_h
        self._target_pos = None

        plt.ion()
        self.fig = plt.figure(figsize=(9, 7), facecolor="#1a1a2e")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
        # mpl 3D: X=右, Y=纵深, Z=上下
        self.ax.set_xlabel("X (mm)", color="white")
        self.ax.set_ylabel("Z (mm)", color="white")
        self.ax.set_zlabel("Y (mm)", color="white")
        self.ax.tick_params(colors="white")
        self.ax.set_title("Robot 3D View", color="white", fontsize=14)

        self.ax.view_init(elev=25, azim=-45)
        # 从 camera_setup.yaml 读取 view3d 范围
        _v3d_cfg = Path("./config/camera_setup.yaml")
        _v3d_xlim = [-8000, 8000]; _v3d_ylim = [-4000, 10000]; _v3d_zlim = [-4000, 8000]
        try:
            with open(_v3d_cfg, "r", encoding="utf-8") as _f:
                _v3d = yaml.safe_load(_f).get("view3d", {})
            if "xlim" in _v3d: _v3d_xlim = _v3d["xlim"]
            if "ylim" in _v3d: _v3d_ylim = _v3d["ylim"]
            if "zlim" in _v3d: _v3d_zlim = _v3d["zlim"]
        except Exception:
            pass
        self.ax.set_xlim(*_v3d_xlim)
        self.ax.set_ylim(*_v3d_ylim)
        self.ax.set_zlim(*_v3d_zlim)

        self._draw_static()

    def _draw_static(self):
        ax = self.ax; rc = self._robot

        # 坐标轴箭头
        _l = 3000
        # mpl 3D: X=右, Y=纵深, Z=上下
        # 我们的: X=右, Y=上, Z=前 → 交换 Y↔Z
        axis_def = [
            ([(0,0,0),(_l,0,0)], "red",   "X",  "X"),
            ([(0,0,0),(0,0,_l)], "green", "Y up", "Y"),
            ([(0,0,0),(0,_l,0)], "blue",  "Z fwd", "Z"),
        ]
        for pts, col, lbl, _ in axis_def:
            xs, ys, zs = zip(*pts)
            ax.plot(xs, ys, zs, color=col, linewidth=3)
            ax.text(pts[1][0], pts[1][1], pts[1][2], lbl.split()[0],
                    color=col, fontsize=14, fontweight="bold")

        # 机器人
        ax.scatter([0], [0], [0], c="gray", s=250, marker="s", alpha=0.8)

        # 相机位置 + 从原点连线 + 标签
        for cid_s, cdef in rc.cameras.items():
            p = cdef["position_mm"]
            is_act = (int(cid_s) == rc.active_cam_id)
            col = "lime" if is_act else "orange"
            # 连线到原点 (交换 Y↔Z)
            ax.plot([0, p[0]], [0, p[2]], [0, p[1]],
                    color=col, alpha=0.35, linewidth=1, linestyle="--")
            # 相机点 (交换 Y↔Z)
            ax.scatter([p[0]], [p[2]], [p[1]], c=col, s=100, marker="o",
                       alpha=0.95, edgecolors="white", linewidths=2,
                       label=f"Cam{cid_s}" if is_act else None)
            # 标签（相机上方 600mm, 这里"上方"=Z+在 mpl 中）
            ax.text(p[0], p[2], p[1]+600, f"Cam{cid_s}",
                    color=col, fontsize=11, ha="center", fontweight="bold")

        # 视锥
        try:
            apex, base = rc.get_frustum_verts(
                rc.active_cam_id, self._fx, self._fy, self._img_w, self._img_h, self._max_dist_mm)
            for i in range(4):
                ax.plot([apex[0],base[i,0]],[apex[2],base[i,2]],[apex[1],base[i,1]],
                        color="cyan", alpha=0.25, linewidth=1)
            for i in range(4):
                j = (i+1)%4
                ax.plot([base[i,0],base[j,0]],[base[i,2],base[j,2]],[base[i,1],base[j,1]],
                        color="cyan", alpha=0.25, linewidth=1)
        except Exception:
            pass

        hh, ll = ax.get_legend_handles_labels()
        seen, uniq = set(), []
        for h, l in zip(hh, ll):
            if l not in seen:
                seen.add(l); uniq.append((h, l))
        if uniq:
            ax.legend(*zip(*uniq), loc="upper left", facecolor="#333",
                      edgecolor="#999", labelcolor="white", fontsize=8)

        # 目标
        self._target = ax.scatter([],[],[], c="red", s=250, marker="o",
                                  alpha=0.9, edgecolors="yellow", linewidths=2)
        # 目标坐标文字
        self._target_label = ax.text(0, 0, 0, "", color="red", fontsize=10,
                                     fontweight="bold", ha="left")

        # 窗口启动时最小化，不抢前台
        try:
            self.fig.canvas.manager.window.iconify()
        except Exception:
            pass

    def update_target(self, pos_abs_mm: numpy.ndarray):
        self._target_pos = pos_abs_mm
        if pos_abs_mm is not None and len(pos_abs_mm) == 3:
            # mpl: X=右, Y=纵深, Z=上下
            # 我们的: X=右, Y=上, Z=前 → 传入 (our_X, our_Z, our_Y)
            self._target._offsets3d = (numpy.array([pos_abs_mm[0]]),
                                        numpy.array([pos_abs_mm[2]]),
                                        numpy.array([pos_abs_mm[1]]))
            self._target.set_alpha(0.9)
            # 坐标文字跟在点旁边（mpl 坐标）
            self._target_label.set_text(f"({pos_abs_mm[0]:.0f},{pos_abs_mm[1]:.0f},{pos_abs_mm[2]:.0f})")
            self._target_label.set_position((pos_abs_mm[0], pos_abs_mm[2]))
            self._target_label.set_3d_properties(pos_abs_mm[1], zdir="z")
            self._target_label.set_alpha(0.9)
        else:
            self._target._offsets3d = (numpy.array([]), numpy.array([]), numpy.array([]))
            self._target.set_alpha(0)
            self._target_label.set_text("")
            self._target_label.set_alpha(0)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def process_events():
        plt.pause(0.005)
