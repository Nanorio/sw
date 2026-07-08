#>>> 库导入
import os
import datetime
import numpy
import cv2
import math
import yaml
from dataclasses import dataclass
from pathlib import Path
#<<<

# ========== 深度修正系数（从 SGBM_params.yaml 读取） ==========
_DEPTH_COR_FACTOR = 0.71
try:
    with open("./config/SGBM_params.yaml", "r", encoding="utf-8") as _f:
        _DEPTH_COR_FACTOR = yaml.safe_load(_f).get("depthCorFactor", _DEPTH_COR_FACTOR)
except Exception:
    pass


@dataclass
class Target_Point:
    """检测到的目标点：像素坐标 + 3D 坐标（已乘深度修正系数）"""
    cx: int
    cy: int
    pt_3d: numpy.ndarray   # [x, y, z] 单位 mm

    @property
    def z(self) -> float:
        return float(self.pt_3d[2])


def get_robust_depth(xyz_matrix, x1, y1, x2, y2):
    """区域中位数滤波，提取最稳的 3D 坐标 (附带详细Debug输出)"""
    box_w, box_h = x2 - x1, y2 - y1
    margin_x = int(box_w * 0.2)
    margin_y = int(box_h * 0.2)
    
    roi_3d = xyz_matrix[y1+margin_y : y2-margin_y, x1+margin_x : x2-margin_x]
    
    if roi_3d.size == 0:
        print(f"[Depth Fail] 目标框太小 ({box_w}x{box_h})，收缩 20% 后没有像素了。")
        return None
        
    valid_mask = (roi_3d[:, :, 2] > 0) & (roi_3d[:, :, 2] < 10000)
    valid_pts = roi_3d[valid_mask]
    
    total_pixels = roi_3d.shape[0] * roi_3d.shape[1]
    
    if len(valid_pts) < 5:
        # 统计一下失效的具体原因
        invalid_z = roi_3d[:, :, 2]
        neg_count = numpy.sum(invalid_z <= 0)
        far_count = numpy.sum(invalid_z >= 10000)
        nan_count = numpy.sum(numpy.isnan(invalid_z))
        
        print(f"[Depth Fail] ROI内共 {total_pixels} 像素。有效点 {len(valid_pts)} 个 (不足5个)。")
        print(f" -> Z<=0像素: {neg_count} | Z>=10m像素: {far_count} | NaN/无效: {nan_count}")
        return None
        
    median_x = numpy.median(valid_pts[:, 0])
    median_y = numpy.median(valid_pts[:, 1])
    median_z = numpy.median(valid_pts[:, 2]) * _DEPTH_COR_FACTOR
    
    return numpy.array([median_x, median_y, median_z])
def outcome_action(results, xyz_matrix, rectify_bgr_left, fps=0.0):
    # 1. 初始化保存目录（利用函数静态属性，只执行一次）
    if not hasattr(outcome_action, "seq_count"):
        outcome_action.seq_count = 1
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        run_idx = 1
        while True:
            dir_name = f"./outcome/{date_str}_{run_idx}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                outcome_action.save_dir = dir_name
                os.makedirs(os.path.join(dir_name, "original"))
                break
            run_idx += 1

    boxes = results[0].boxes
    
    # 保存原始图（无标注），用于训练
    if len(boxes) >= 1:
        raw_path = f"{outcome_action.save_dir}/original/{outcome_action.seq_count:04d}.jpg"
        cv2.imwrite(raw_path, rectify_bgr_left)

    display_img = rectify_bgr_left.copy()
    
    # ===== 光轴十字线（主点位置） =====
    h_i, w_i = display_img.shape[:2]
    cx_i, cy_i = 331, 234  # 标定主点 (left_matrix 经转置后)
    cv2.line(display_img, (cx_i, 0), (cx_i, h_i - 1), (0, 255, 0), 1)
    cv2.line(display_img, (0, cy_i), (w_i - 1, cy_i), (0, 255, 0), 1)
    cv2.circle(display_img, (cx_i, cy_i), 5, (0, 255, 0), 2)
    
    if len(boxes) >= 1:
        blue_pts = []   # BlueLight (class 0) → 红色
        green_pts = []  # GreenLight (class 1) → 紫色
        blue_3d = []   # BlueLight 有有效深度的点
        green_3d = []  # GreenLight 有有效深度的点
        
        for target in boxes:
            cls = int(target.cls[0])
            x1, y1, x2, y2 = map(int, target.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            pt = (cx, cy)
            if cls == 0:         # BlueLight
                blue_pts.append(pt)
            else:                 # GreenLight
                green_pts.append(pt)
            
            # 计算该点 3D 坐标 → 解算相对相机光轴的角度
            pt_3d = get_robust_depth(xyz_matrix, x1, y1, x2, y2)
            if pt_3d is not None:
                _tp = Target_Point(cx=cx, cy=cy, pt_3d=pt_3d)
                if cls == 0:
                    blue_3d.append(_tp)
                else:
                    green_3d.append(_tp)
                h_ang = math.degrees(math.atan2(pt_3d[0], pt_3d[2]))
                v_ang = math.degrees(math.atan2(-pt_3d[1], pt_3d[2]))
                cv2.putText(display_img, f"Yaw:{h_ang:+.1f} Pitch:{v_ang:+.1f}",
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1)
                cv2.putText(display_img, f"Z:{pt_3d[2]:.0f}mm",
                            (cx + 10, cy + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1)
        
        # 画点
        for pt in blue_pts:
            cv2.circle(display_img, pt, 6, (0, 0, 255), -1)       # 红色
        for pt in green_pts:
            cv2.circle(display_img, pt, 6, (255, 0, 255), -1)     # 紫色
        
        # 画线：紫点 ↔ 红点 = 白色
        for g_pt in green_pts:
            for b_pt in blue_pts:
                cv2.line(display_img, g_pt, b_pt, (255, 255, 255), 2)
        
        # 画线：红点 ↔ 红点 = 黑色
        for i in range(len(blue_pts)):
            for j in range(i + 1, len(blue_pts)):
                cv2.line(display_img, blue_pts[i], blue_pts[j], (0, 0, 0), 2)
        
        # ===== 目标级角度解算（按检测点数量分情况） =====
        n_total = len(blue_3d) + len(green_3d)
        if n_total >= 1:
            all_tp = blue_3d + green_3d
            
            # 几何中心（1点=自身，2点=中点，3点=三角形重心）
            center_3d = numpy.mean([tp.pt_3d for tp in all_tp], axis=0)
            target_yaw = math.degrees(math.atan2(center_3d[0], center_3d[2]))
            target_pitch = math.degrees(math.atan2(-center_3d[1], center_3d[2]))
            _dist = numpy.linalg.norm(center_3d)
            
            # Roll：黑线垂线方向远离绿灯，与 Y 轴正半轴（朝上）的夹角
            target_roll = None
            if n_total >= 3 and len(blue_3d) >= 2 and len(green_3d) >= 1:
                _bp = sorted(blue_3d, key=lambda p: p.cx)
                _bL, _bR = _bp[0], _bp[-1]
                _dx = _bR.cx - _bL.cx
                _dy = _bR.cy - _bL.cy
                if abs(_dx) > 1 or abs(_dy) > 1:
                    # 黑线的两个垂直方向
                    _p1 = (_dy, -_dx)
                    _p2 = (-_dy, _dx)
                    # 黑线中点 → 绿灯
                    _mx = (_bL.cx + _bR.cx) / 2.0
                    _my = (_bL.cy + _bR.cy) / 2.0
                    _g = green_3d[0]
                    _tg = (_g.cx - _mx, _g.cy - _my)
                    # 选远离绿灯的垂线（dot < 0）
                    if _p1[0] * _tg[0] + _p1[1] * _tg[1] < 0:
                        _aux = _p1
                    else:
                        _aux = _p2
                    # 辅助线与 Y 正半轴(朝上=0,-1)的夹角
                    target_roll = -math.degrees(math.atan2(_aux[0], -_aux[1]))
            
            if target_roll is not None:
                print(f"🎯 目标 → Yaw:{target_yaw:+.1f} Pitch:{target_pitch:+.1f} Roll:{target_roll:+.1f}")
                cv2.putText(display_img, f"Yaw:{target_yaw:+.1f}  Pitch:{target_pitch:+.1f}  Roll:{target_roll:+.1f}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                print(f"🎯 目标 → Yaw:{target_yaw:+.1f} Pitch:{target_pitch:+.1f}")
                cv2.putText(display_img, f"Yaw:{target_yaw:+.1f}  Pitch:{target_pitch:+.1f}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.putText(display_img, f"Z:{center_3d[2]:.0f}mm  X:{center_3d[0]:.0f}  Y:{center_3d[1]:.0f}",
                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 三点：画青色三角 + 黄色顶点标记
            if n_total >= 3 and len(blue_3d) >= 2 and len(green_3d) >= 1:
                _sy = sorted(all_tp, key=lambda p: p.cy)
                _bot = _sy[-1]
                _top2 = sorted(_sy[:2], key=lambda p: p.cx)
                _lpt, _rpt = _top2[0], _top2[1]
                cv2.line(display_img, (_lpt.cx, _lpt.cy), (_rpt.cx, _rpt.cy), (255, 255, 0), 2)
                cv2.line(display_img, (_lpt.cx, _lpt.cy), (_bot.cx, _bot.cy), (255, 255, 0), 2)
                cv2.line(display_img, (_rpt.cx, _rpt.cy), (_bot.cx, _bot.cy), (255, 255, 0), 2)
                cv2.circle(display_img, (_lpt.cx, _lpt.cy), 4, (0, 255, 255), -1)
                cv2.circle(display_img, (_rpt.cx, _rpt.cy), 4, (0, 255, 255), -1)
                cv2.circle(display_img, (_bot.cx, _bot.cy), 4, (0, 255, 255), -1)
        
        # 保存标注图
        save_path = f"{outcome_action.save_dir}/{outcome_action.seq_count:04d}.jpg"
        cv2.imwrite(save_path, display_img)
        outcome_action.seq_count += 1
    
    # 帧率
    cv2.putText(display_img, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return display_img