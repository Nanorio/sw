import cv2
import time

def standard_capture(camera_id=0):
    print(">>> 开始标准高频抓图测试 (按 'q' 退出) <<<")
    
    # 1. 在循环外部，仅初始化一次硬件
    cap = cv2.VideoCapture(camera_id)
    
    # 可选：如果你想降低延迟，可以把硬件缓冲池设为 1
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    if not cap.isOpened():
        print("无法打开相机硬件！")
        return

    while True:
        start_time = time.time()
        
        # 2. 循环内仅进行极速的内存搬运
        ret, frame = cap.read()
        
        if ret:
            # 此时的 FPS 才是真实的硬件传输+算法处理帧率 (通常 30 或 60 FPS)
            fps = 1.0 / (time.time() - start_time)
            
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Standard Capture", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 3. 彻底结束任务时，再释放硬件
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    standard_capture(0)