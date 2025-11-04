from PySide6.QtCore import QThread, Signal

class PredictThread(QThread):
    finished_signal = Signal(object)  # 发送预测结果

    def __init__(self, img_path, model_path):
        super().__init__()
        self.img_path = img_path
        self.model_path = model_path

    def run(self):
        import sys
        sys.path.append(r'c:\somefiles\Cell-Segmentation')
        from yolov11.predict import yolo_predict
        import numpy as np

        mask = yolo_predict(self.img_path, self.model_path)
        self.finished_signal.emit(mask)
