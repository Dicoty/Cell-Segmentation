from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow, QMenu, QFileDialog, QLabel, 
                               QVBoxLayout, QToolBar, QSizePolicy, QPlainTextEdit)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PIL import Image, ImageQt
from utils import np2mask, sub_img

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 600)

        # 创建菜单
        self.create_menu()

        # 创建右侧工具栏
        self.create_toolbar()

        # 创建中心部件
        centralWidget = self.creat_central_widget()
        self.setCentralWidget(centralWidget)
        

        self.bind()

    def bind(self):
        self.openFile.triggered.connect(self.Open_File)
        self.saveFile.triggered.connect(self.Save_File)
        self.importMask.triggered.connect(self.Import_Mask)
        self.subImg.triggered.connect(self.Sub_Img)
        self.select.triggered.connect(self.Select_Model)
        self.predict.triggered.connect(self.Predict)

    def create_menu(self):
        """
        创建菜单栏
        """     
        self.menu = self.menuBar()

        # 文件菜单
        self.openFile = QAction('导入')
        self.saveFile = QAction('保存')
        self.importMask = QAction('导入Mask')
        self.fileMenu = QMenu('文件')
        self.fileMenu.addAction(self.openFile)
        self.fileMenu.addAction(self.saveFile)
        self.fileMenu.addAction(self.importMask)
        self.menu.addMenu(self.fileMenu)

        # 模型菜单
        self.modelMenu = QMenu('模型')
        self.predict = QAction('预测')
        self.select = QAction('选择模型')
        self.modelMenu.addAction(self.predict)
        self.modelMenu.addAction(self.select)
        self.menu.addMenu(self.modelMenu)

        # 设置菜单
        self.setting = QMenu('设置')
        self.menu.addMenu(self.setting)

    def create_toolbar(self):
        """
        创建工具栏
        """
        # 停靠在右侧的工具栏
        self.toolbar = QToolBar('工具栏')
        self.toolbar.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.toolbar)

        # 打开文件按钮
        self.toolbar.addAction(self.openFile)

        # 预测按钮
        self.toolbar.addAction(self.predict)

        # 裁剪按钮
        self.subImg = QAction('裁剪')
        self.toolbar.addAction(self.subImg)

        # 保存按钮
        self.toolbar.addAction(self.saveFile)

    def creat_central_widget(self):
        """
        创建中心部件
        """
        central_widget = QWidget()
        self.mainlayout = QVBoxLayout()

        # 展示图片
        self.lbShowImg = QLabel()
        self.lbShowImg.setAlignment(Qt.AlignCenter)
        self.lbShowImg.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # 信息显示文本框
        self.infoTextEdit = QPlainTextEdit()
        self.infoTextEdit.setMaximumHeight(100)
        self.infoTextEdit.setReadOnly(True)

        # 添加部件到布局
        self.mainlayout.addWidget(self.lbShowImg)
        self.mainlayout.addWidget(self.infoTextEdit)

        central_widget.setLayout(self.mainlayout)

        return central_widget
        
    def Open_File(self):
        print('打开文件')
        self.img_path = QFileDialog.getOpenFileName(self, '选择图片', './', '图片文件(*.png *.jpg *.jpeg *.bmp)')[0]
        if self.img_path:  # 确保用户选择了文件
            self.img = Image.open(self.img_path)
            self.infoTextEdit.appendPlainText(f'文件名: {self.img_path}')
        else:
            self.infoTextEdit.appendPlainText('未选择文件')
                    
        # 展示图片
        self.show_img(self.img)

    def Save_File(self):
        """
        将当前展示的图片保存,先这样写,之后再改改
        """
        self.current_img = self.lbShowImg.pixmap()
        save_path = QFileDialog.getSaveFileName(self, '保存图片', './', '图片文件(*.png *.jpg)')[0]
        self.current_img.save(save_path)
        self.infoTextEdit.appendPlainText(f'已保存当前文件于：{save_path}')

    def Import_Mask(self):
        print('导入Mask')
        mask_path = QFileDialog.getOpenFileName(self, '选择Mask', './', '图片文件(*.tif)')[0]
        if mask_path:  # 确保用户选择了文件
            import tifffile
            np_mask = tifffile.imread(mask_path)
            self.imgmask = np2mask(np_mask)
            self.infoTextEdit.appendPlainText(f'导入Mask: {mask_path}')
        else:
            print('未选择文件')
            self.infoTextEdit.appendPlainText('未选择文件')

        alpha = self.imgmask.split()[-1]          # 取出 α 通道
        alpha = alpha.point(lambda p: int(p * 0.4))  # 乘以 0.4
        self.imgmask.putalpha(alpha)              # 把改过的 α 写回去

        self.img_with_mask = self.img.copy().convert('RGBA')
        self.img_with_mask.paste(self.imgmask, (0, 0), self.imgmask)

        self.show_img(self.img_with_mask)

    def Sub_Img(self):
        print('裁剪图片')
        self.infoTextEdit.appendPlainText('裁剪图片')
        self.sub_img = sub_img(self.img, self.imgmask)
        self.show_img(self.sub_img)

    def Predict(self):
        self.infoTextEdit.appendPlainText('预测中……')
        import sys
        sys.path.append(r'c:\somefiles\Cell-Segmentation')  # 添加项目根目录到路径
        from yolov11.predict import yolo_predict
        import numpy as np
        

        mask = yolo_predict(self.img_path, self.model_path) # mask是一个二维np数组
        self.infoTextEdit.appendPlainText('预测完成')
        self.imgmask = np2mask(mask)

        alpha = self.imgmask.split()[-1]          # 取出 α 通道
        alpha = alpha.point(lambda p: int(p * 0.6))  # 乘以 0.4
        self.imgmask.putalpha(alpha)              # 把改过的 α 写回去

        self.img_with_mask = self.img.copy().convert('RGBA')
        self.img_with_mask.paste(self.imgmask, (0, 0), self.imgmask)

        self.show_img(self.img_with_mask)
        
        
    def Select_Model(self):
        self.model_path = QFileDialog.getOpenFileName(self, '选择模型文件', './', '权重文件(*)')[0]
        self.infoTextEdit.appendPlainText('模型路径：' + self.model_path)
    def show_img(self, img):

        # 获取 QLabel 的尺寸
        label_size = self.lbShowImg.size()
        
        # 按比例缩放图片
        img_scaled = img.copy()
        img_scaled.thumbnail((label_size.width(), label_size.height()), Image.Resampling.LANCZOS)

        self.lbShowImg.setPixmap(ImageQt.toqpixmap(img_scaled))




if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

