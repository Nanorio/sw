#>>> 库导入
import yaml
import numpy
from pathlib import Path
import cv2
from ultralytics import YOLO
#<<<

class YOLOV8():

    def __init__(self):
        self._predict_init()
    
    def _predict_init(self):

        print(">>>\n正在读取yolo配置文件...\n...")

        yolo_configuration = "./config/yolo_params.yaml"
        if not Path(yolo_configuration).exists():
            raise FileNotFoundError(f"找不到yolo配置文件: {yolo_configuration}")
      
        with open(yolo_configuration, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
            yolo_weight = Path("./weights") / params['weightName']
            if not Path(yolo_weight).exists():
                raise FileNotFoundError(f"找不到yolo权重文件: {yolo_weight}")
            self.save = params["save"]
            self.conf = params["conf"]
            self.show = params["show"]
            self.stream = params["stream"]
            self.verbose = params["verbose"]
            self.device = params["device"]
            
        self.model = YOLO(yolo_weight, task="detect")

        print("yolo配置初始化完毕!\n<<<\n————————————————————")
    
    def predict(self, image):
        self.results = self.model.predict(source = image, 
							    save = self.save, 
							    conf = self.conf, 
							    show = self.show, 
							    device = self.device, 
							    stream = self.stream,
                                verbose = self.verbose)