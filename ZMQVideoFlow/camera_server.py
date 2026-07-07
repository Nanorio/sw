import cv2
import zmq
import time

def camera_publisher():
    # 1. 初始化 ZMQ 上下文与 PUB（发布）套接字
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # 绑定到本地的 5555 端口 (如果是跨电脑，写成 tcp://0.0.0.0:5555)
    socket.bind("tcp://*:5555")
    
    print(">>> [Server] 视觉基站已启动，正在广播画面... (按 Ctrl+C 终止) <<<")
    
    # 2. 独占相机硬件
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 【核心优化】：将庞大的 numpy 数组压缩为 JPEG 字节流
            # 第一个参数是格式，第二个是原图，第三个控制压缩质量（0-100，90是极佳的平衡点）
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            success, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            
            if success:
                # 将压缩后的字节流扔进 ZMQ 广播通道
                socket.send(encoded_img.tobytes())
                
    except KeyboardInterrupt:
        print("\n>>> [Server] 收到终止指令，释放相机硬件。")
    finally:
        cap.release()
        context.term()

if __name__ == '__main__':
    camera_publisher()