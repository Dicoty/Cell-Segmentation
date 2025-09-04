from PySide6.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QSizePolicy)
from PySide6.QtCore import Qt, Signal

class saveWindow(QWidget):

    saveSignal = Signal(str)
    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.setWindowTitle('保存图片')
        self.resize_and_center()
        
        self.btn1 = QPushButton('保存原图')
        self.btn2 = QPushButton('保存重叠mask后的图')
        self.btn3 = QPushButton('保存裁切图')
        self.btn4 = QPushButton('取消')

        self.btn1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.btn2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.btn3.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.btn4.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.btn1)
        self.layout.addWidget(self.btn2)
        self.layout.addWidget(self.btn3)
        self.layout.addWidget(self.btn4)
        self.setLayout(self.layout)
        self.bind()

    def bind(self):
        self.btn1.clicked.connect(lambda: self.saveSignal.emit('origin'))
        self.btn2.clicked.connect(lambda: self.saveSignal.emit('mask'))
        self.btn3.clicked.connect(lambda: self.saveSignal.emit('crop'))
        self.btn4.clicked.connect(self.close)

    def resize_and_center(self):
        # 获取主窗口尺寸
        parent_size = self.parent.size()
        parent_pos = self.parent.pos()
        
        # 计算小窗口尺寸为主窗口的40%
        width = int(parent_size.width() * 0.4)
        height = int(parent_size.height() * 0.4)
        
        # 设置尺寸
        self.resize(width, height)
        
        # 居中显示在主窗口上
        x = parent_pos.x() + (parent_size.width() - width) // 2
        y = parent_pos.y() + (parent_size.height() - height) // 2
        self.move(x, y)



        

