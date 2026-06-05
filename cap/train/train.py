from ultralytics import YOLO

def main():
    # 1. 加载模型
    # 建议加载预训练权重 'yolov8n.pt'，这叫“迁移学习”，比从零训练快且准
    model = YOLO("/home/nvidia/CAP/weights/best.pt") 

    dotpt_out = "./runs/detect/train/weights"    # 固定目录

    # 2. 开始训练
    model.train(
        data='yolov8.yaml',  # yaml配置文件路径
        epochs=100,          # 训练轮数
        imgsz=640,           # 图片尺寸
        batch=4,             # 每批次处理的图像数量,显存过小则调小4/8/16
        device=0,            # 使用索引值为0的GPU，无GPU则device='cpu'
    )

    # 3.权重文件生成engine
    #model.export(format="onnx", imgsz=640, half=True) # 可选属性imgsz,half,simplify,int8
    model.export(format="engine", imgsz=640, half=True)  # 生成Jetson专用的模型文件TensorRT,使用GPU获取5倍加速推理,谁推理谁用该模型,模型不通用

if __name__ == '__main__':
    main()
