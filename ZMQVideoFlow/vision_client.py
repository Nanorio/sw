import cv2
import zmq
import numpy as np
import time

def vision_subscriber():
    print(">>> [Client] 正在连接视觉基站... <<<")
    
    # 1. 初始化 ZMQ 与 SUB（订阅）套接字
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    # 【核心优化】：极其关键的一句！丢弃堆积缓存，永远只拿最新的一帧，实现零延迟！
    socket.setsockopt(zmq.CONFLATE, 1)
    
    # 连接到 Server (如果在同一台机器就是 localhost，跨网线就填 Nvidia 的 IP)
    socket.connect("tcp://localhost:5555")
    # 订阅所有话题内容（空字符串代表全部接收）
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    last_time = time.time()

    try:
        while True:
            # 2. 接收传过来的 JPEG 字节流
            image_bytes = socket.recv()
            
            # 3. 将字节流解码回 OpenCV 认识的 BGR numpy 矩阵
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # 计算接收端的真实 FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_time)
                last_time = current_time
                
                cv2.putText(frame, f"Client FPS: {fps:.1f}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 尽情在这里加入你的 YOLO 算法或者 SGBM 双目处理逻辑！
                cv2.imshow("ZMQ Video Stream Receiver", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        context.term()

if __name__ == '__main__':
    vision_subscriber()