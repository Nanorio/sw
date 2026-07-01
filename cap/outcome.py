#>>> 库导入
import os
import datetime
import numpy
import cv2
import math
#<<<

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
    median_z = numpy.median(valid_pts[:, 2])
    
    return numpy.array([median_x, median_y, median_z])
def outcome_action(results, xyz_matrix, rectify_bgr_left):
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
                break
            run_idx += 1

    boxes = results[0].boxes
    
    # === [核心修改] 防漏帧机制：只要识别数 >= 1，就准备保存图片 ===
    if len(boxes) >= 1:
        save_img = rectify_bgr_left.copy()
        
        # 将所有的框按照 YOLO 的置信度 (conf) 从高到低进行排序
        sorted_boxes = sorted(boxes, key=lambda b: b.conf[0].item(), reverse=True)
        
        # 最多依然只切取前 3 名“学霸”去算姿态，如果只有 1 个或 2 个，就取 1 个或 2 个
        best_boxes = sorted_boxes[:3]
        
        centers_2d_data = [] 
        
        # 遍历这几个靠谱的框
        for target in best_boxes:
            x1, y1, x2, y2 = map(int, target.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # 使用中位数滤波获取该灯的 3D 坐标
            pt_3d = get_robust_depth(xyz_matrix, x1, y1, x2, y2)
            
            if pt_3d is not None:
                centers_2d_data.append({'cx': cx, 'cy': cy, 'pt_3d': pt_3d})
                # [新增]: 只要算出了深度，无论凑没凑齐 3 个，都把当前灯的 Z 轴画出来方便观察
                cv2.putText(save_img, f"Z:{pt_3d[2]:.0f}", (cx-20, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
        # === 只有这 3 个最靠谱的灯的三维坐标都提取成功了，才进行姿态解算 ===
        if len(centers_2d_data) == 3:
            # 按 cy (画面Y坐标) 升序排列
            centers_2d_data.sort(key=lambda item: item['cy'])
            
            # 在倒三角里，cy最大的（排在最后的）是底部的那 1 个灯
            bottom_light_data = centers_2d_data[2] # C点数据
            
            # 使用红色标注底部灯的参考点，半径更大，实心
            cv2.circle(save_img, (bottom_light_data['cx'], bottom_light_data['cy']), 8, (0, 0, 255), -1) 
            
            # 前两个是顶部的 2 个灯，按 cx 排序分出左右
            top_lights_data = centers_2d_data[:2]
            top_lights_data.sort(key=lambda item: item['cx'])
            left_light_data = top_lights_data[0]   # 左上角的灯 (A点数据)
            right_light_data = top_lights_data[1]  # 右上角的灯 (B点数据)
            
            # 使用绿色标注顶部两个灯的参考点
            cv2.circle(save_img, (left_light_data['cx'], left_light_data['cy']), 5, (0, 255, 0), -1) 
            cv2.circle(save_img, (right_light_data['cx'], right_light_data['cy']), 5, (0, 255, 0), -1) 
            
            # 提取 3D 点
            A = left_light_data['pt_3d']
            B = right_light_data['pt_3d']
            C = bottom_light_data['pt_3d']
            
            # 目标的绝对距离
            target_distance = (A[2] + B[2] + C[2]) / 3.0
            
            # === 三维向量叉乘求法向量 → 构建正交基 → 提取姿态角 ===
            vec_AB = B - A   
            vec_AC = C - A   
            normal_vec = numpy.cross(vec_AB, vec_AC) 
            
            norm_length = numpy.linalg.norm(normal_vec)
            if norm_length > 0:
                normal_vec = normal_vec / norm_length
                nx, ny, nz = normal_vec
                
                yaw = math.degrees(math.atan2(nx, nz))
                pitch = math.degrees(math.asin(numpy.clip(-ny, -1.0, 1.0)))
                roll = math.degrees(math.atan2(vec_AB[1], vec_AB[0]))
                
                print(f"🎯 锁定目标！距离: {target_distance:.2f} mm")
                print(f"✈️ 姿态 -> 偏航(Yaw): {yaw:.1f}° | 俯仰(Pitch): {pitch:.1f}° | 横滚(Roll): {roll:.1f}°")
                
                # A -> B 顶边 (绿色)
                cv2.line(save_img, (left_light_data['cx'], left_light_data['cy']), (right_light_data['cx'], right_light_data['cy']), (0, 255, 0), 2)
                # A -> C 左侧斜边 (红色)
                cv2.line(save_img, (left_light_data['cx'], left_light_data['cy']), (bottom_light_data['cx'], bottom_light_data['cy']), (0, 0, 255), 2)
                # B -> C 右侧斜边 (蓝色)
                cv2.line(save_img, (right_light_data['cx'], right_light_data['cy']), (bottom_light_data['cx'], bottom_light_data['cy']), (255, 0, 0), 2)

                # 打水印
                cv2.putText(save_img, f"Dist: {target_distance:.0f}mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(save_img, f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            # 如果灯不够3个，或者有几个算不出深度，在终端提示一下，但不妨碍下方保存图片
            print(f"当前找到 {len(boxes)} 个灯，成功计算深度的有 {len(centers_2d_data)} 个。不足以解算姿态，跳过姿态解算。")

        save_path = f"{outcome_action.save_dir}/{outcome_action.seq_count:04d}.jpg"
        cv2.imwrite(save_path, save_img)
        outcome_action.seq_count += 1

        return save_img
            
    else:
        print("当前没有识别到任何灯，不保存图片。")
        # 即使无检测也返回一张带提示的图，用于实时显示
        display_img = rectify_bgr_left.copy()
        cv2.putText(display_img, "No detection", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return display_img