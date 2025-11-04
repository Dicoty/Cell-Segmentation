from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow, QMenu, QFileDialog, QLabel, 
                               QVBoxLayout, QToolBar, QSizePolicy, QPlainTextEdit, QSlider, QHBoxLayout,
                               QFrame, QStackedWidget, QPushButton, QDialog)
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
import numpy as np
from PIL import Image, ImageQt
from utils import np2mask, sub_img, calculate
from littlewindow import saveWindow
from predict_thread import PredictThread
from batch_thread import BatchProcessThread
from batch_config_dialog import BatchConfigDialog
import qdarkstyle
from qt_material import apply_stylesheet

# 控件的绑定方法命名格式为下划线分隔首字母大写，功能性方法命名为下划线分隔纯小写

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 600)
        self.setWindowTitle('细胞检测')

        # 创建堆叠布局
        self.stacked_widget = QStackedWidget()

        # 创建菜单
        self.create_menu()

        # 创建右侧工具栏
        self.create_toolbar()

        # 创建中心部件
        seg_centralWidget = self.creat_seg_central_widget()
        self.stacked_widget.addWidget(seg_centralWidget) # 索引为0为细胞检测

        # 视频处理界面
        self.creat_video_widget()
        self.stacked_widget.addWidget(self.video_widget) # 索引为1为视频处理

        self.setCentralWidget(self.stacked_widget)

        # 重要变量
        # 图片处理相关
        self.img = None
        self.img_with_mask = None
        self.img_sub = None
        self.current_img = None
        self.np_mask = None
        self.opaque = 0.6
        self.model_path = None

        # 视频处理相关
        self.media_player = None
        self.audio_output = None
        self.is_video_mode = False
        

        self.bind()
        

    def bind(self):
        self.openFile.triggered.connect(self.Open_File)
        self.saveFile.triggered.connect(self.Save_File)
        self.importMask.triggered.connect(self.Import_Mask)
        self.subImg.triggered.connect(self.Sub_Img)
        self.select.triggered.connect(self.Select_Model)
        self.predict.triggered.connect(self.Predict)
        self.calculate.triggered.connect(self.Calculate)
        self.openVideo.triggered.connect(self.Open_Video)
        self.batchProcess.triggered.connect(self.Batch_Process)

    def create_menu(self):
        """
        创建菜单栏
        """     
        self.menu = self.menuBar()

        # 文件菜单
        self.openFile = QAction('导入图片')
        self.openVideo = QAction('导入视频')
        self.saveFile = QAction('保存')
        self.importMask = QAction('导入Mask')
        self.batchProcess = QAction('批量SPR处理')
        self.fileMenu = QMenu('文件')
        self.fileMenu.addAction(self.openFile)
        self.fileMenu.addAction(self.openVideo)
        self.fileMenu.addAction(self.saveFile)
        self.fileMenu.addAction(self.importMask)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.batchProcess)
        self.menu.addMenu(self.fileMenu)

        # 模型菜单
        self.modelMenu = QMenu('模型')
        self.predict = QAction('预测')
        self.select = QAction('选择模型')
        self.calculate = QAction('计算')
        self.modelMenu.addAction(self.predict)
        self.modelMenu.addAction(self.select)
        self.modelMenu.addAction(self.calculate)
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

        # 计算按钮
        self.toolbar.addAction(self.calculate)

        # 分隔符
        self.toolbar.addSeparator()

        # 批量处理按钮
        self.toolbar.addAction(self.batchProcess)

    def creat_seg_central_widget(self):
        """
        创建图像分割的界面
        """
        central_widget = QWidget()
        self.mainlayout = QVBoxLayout()

        # 展示图片
        self.lbShowImg = QLabel()
        self.lbShowImg.setAlignment(Qt.AlignCenter)
        self.lbShowImg.setMinimumSize(400, 300)
        self.lbShowImg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbShowImg.setStyleSheet("border: 1px solid gray;")

        # 调整模糊度的滑条
        opaquesliderWidget = QWidget()
        opaquesliderLayout = QHBoxLayout()
        self.opaqueLabel = QLabel('透明度: 60%')
        opaquesliderLayout.addWidget(self.opaqueLabel)
        self.opaqueSlider = QSlider(Qt.Orientation.Horizontal)
        self.opaqueSlider.setRange(0, 100)
        self.opaqueSlider.setValue(50)
        self.opaqueSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.opaqueSlider.setTickInterval(10)
        self.opaqueSlider.valueChanged.connect(self.update_alpha)
        opaquesliderLayout.addWidget(self.opaqueSlider)
        opaquesliderWidget.setLayout(opaquesliderLayout)

        # 调整缩放比例的滑条
        sizeWidget = QWidget()
        sizeslider = QHBoxLayout()
        self.sizeLabel = QLabel('缩放: 50%')
        sizeslider.addWidget(self.sizeLabel)
        self.sizeSlider = QSlider(Qt.Orientation.Horizontal)
        self.sizeSlider.setRange(0,200)
        self.sizeSlider.setValue(50)
        self.sizeSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sizeSlider.setTickInterval(20)
        self.sizeSlider.valueChanged.connect(self.update_size)
        sizeslider.addWidget(self.sizeSlider)
        sizeWidget.setLayout(sizeslider)

        # 信息显示文本框
        self.infoTextEdit = QPlainTextEdit()
        self.infoTextEdit.setMaximumHeight(200)
        self.infoTextEdit.setReadOnly(True)

        # 添加部件到布局
        self.mainlayout.addWidget(self.lbShowImg)
        self.mainlayout.addWidget(opaquesliderWidget)
        self.mainlayout.addWidget(sizeWidget)
        self.mainlayout.addWidget(self.infoTextEdit)        

        central_widget.setLayout(self.mainlayout)

        return central_widget
    
    def creat_video_widget(self):
        """
        创建视频处理界面
        """
        self.video_widget = QWidget()
        video_layout = QVBoxLayout()

        # 视频显示区域
        self.video_display = QVideoWidget()
        self.video_display.setMinimumSize(400, 300)
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_display.setStyleSheet("border: 1px solid gray;")

        # 视频控制按钮
        video_control_layout = QHBoxLayout()
        self.play_btn = QPushButton('播放')
        self.pause_btn = QPushButton('暂停')
        self.stop_btn = QPushButton('停止')

        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)
        self.stop_btn.clicked.connect(self.stop_video)

        # 添加按钮
        video_control_layout.addWidget(self.play_btn)
        video_control_layout.addWidget(self.pause_btn)
        video_control_layout.addWidget(self.stop_btn)

        # 信息显示文本框
        self.video_infoTextEdit = QPlainTextEdit()
        self.video_infoTextEdit.setMaximumHeight(200)
        self.video_infoTextEdit.setReadOnly(True)

        # 添加部件到布局
        video_layout.addWidget(self.video_display)
        video_layout.addLayout(video_control_layout)
        video_layout.addWidget(self.video_infoTextEdit)

        self.video_widget.setLayout(video_layout)

    def init_media_player(self):
        """
        初始化媒体播放器
        """
        if not self.media_player:
            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_display)
    
    def update_alpha(self, value):
        """
        更新alpha透明度并刷新显示
        """
        self.opaque = value / 100.0
        if self.imgmask is not None:
            self.imgmaskcopy = self.imgmask.copy()
            alpha = self.imgmaskcopy.split()[-1]          # 取出 α 通道
            alpha = alpha.point(lambda p: int(p * self.opaque))  # 乘以 self.opaque
            self.imgmaskcopy.putalpha(alpha)              # 把改过的 α 写回去

            # 重新合成图像
            self.img_with_mask = self.img.copy().convert('RGBA')
            self.img_with_mask.paste(self.imgmaskcopy, (0, 0), self.imgmaskcopy)
            # 刷新显示
            self.current_img = self.img_with_mask.copy()
            self.show_img(self.current_img)

            self.opaqueLabel.setText(f'透明度: {value}%')
                
    def update_size(self, value):
        """
        更新图片大小并刷新显示
        """
        if self.current_img is not None:
            self.show_img(self.current_img)
        self.sizeLabel.setText(f'缩放: {value}%')
    
    def Open_File(self):
        """
        打开图片文件
        """
        # 切换图片处理界面
        self.stacked_widget.setCurrentIndex(0)
        self.is_video_mode = False
        # 停止正在播放的视频
        if self.media_player and self.media_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:  # 如果正在播放视频
            self.media_player.stop()

        # 打开图片文件
        self.img_path = QFileDialog.getOpenFileName(self, '选择图片', './', '图片文件(*.png *.jpg *.jpeg *.bmp *.tif)')[0]
        if self.img_path:  # 确保用户选择了文件
            self.img = Image.open(self.img_path)
            self.infoTextEdit.appendPlainText(f'文件名: {self.img_path}')
        else:
            self.infoTextEdit.appendPlainText('未选择文件')
                    
        # 展示图片
        self.current_img = self.img.copy()
        self.show_img(self.current_img)

    def Open_Video(self):
        """
        导入并播放视频
        """
        video_path, _ = QFileDialog.getOpenFileName(
        self, 
        '选择视频文件', 
        './', 
        '视频文件(*.mp4 *.avi *.mov *.mkv *.flv *.wmv)'
    )
    
        if video_path:
            # 切换到视频界面
            self.stacked_widget.setCurrentIndex(1)
            self.is_video_mode = True
            
            # 初始化媒体播放器
            self.init_media_player()
            
            # 设置视频源并播放
            from PySide6.QtCore import QUrl
            self.media_player.setSource(QUrl.fromLocalFile(video_path))
            self.video_infoTextEdit.appendPlainText(f'导入视频: {video_path}')
            self.media_player.play()

    def Save_File(self, save_type):
        """
        保存图片，点击按钮后弹出窗口选择保存方式
        """
        # 创建保存子窗口
        self.save_window = saveWindow(self)

        # 连接子窗口的信号到处理函数
        self.save_window.saveSignal.connect(self.handle_save_signal)
        self.save_window.show()
   
    def handle_save_signal(self, save_type):
        """
        处理保存信号
        """
        if save_type == 'npy':
            if self.np_mask is None:
                self.infoTextEdit.appendPlainText('请预测Mask')
                return
            else:
                save_path = QFileDialog.getSaveFileName(self, '保存文件', './', 'npy文件(*.npy)')[0]
                np.save(save_path, self.np_mask)
                self.infoTextEdit.appendPlainText(f"保存Mask: {save_path}")               
        elif save_type == 'mask':
            if self.img_with_mask is None:  # 如果没有导入mask，则提示导入mask
                self.infoTextEdit.appendPlainText('请导入Mask')
                return
            else:
                save_path = QFileDialog.getSaveFileName(self, '保存图片', './', '图片文件(*.png *.jpg *.jpeg *.bmp)')[0]
                self.img_with_mask.save(save_path)
                self.infoTextEdit.appendPlainText(f"保存重叠mask后的图: {save_path}")
        elif save_type == 'crop':
            if self.img_sub is None:  # 如果没有裁切，则提示裁切
                self.infoTextEdit.appendPlainText('请裁切图片')
                return
            else:
                save_path = QFileDialog.getSaveFileName(self, '保存图片', './', '图片文件(*.png *.jpg *.jpeg *.bmp)')[0]
                self.img_sub.save(save_path)
                self.infoTextEdit.appendPlainText(f"保存裁切图： {save_path}")

    def Import_Mask(self):
        """
        导入Mask
        """
        mask_path = QFileDialog.getOpenFileName(self, '选择Mask', './', '图片文件(*.tif)')[0]
        if mask_path:  # 确保用户选择了文件
            import tifffile
            self.np_mask = tifffile.imread(mask_path)
            self.imgmask = np2mask(self.np_mask)
            self.infoTextEdit.appendPlainText(f'导入Mask: {mask_path}')
        else:
            print('未选择文件')
            self.infoTextEdit.appendPlainText('未选择文件')

        self.update_alpha(self.opaque * 100)

    def Sub_Img(self):
        """
        只保留遮罩内的区域
        """
        if self.img and self.imgmask is not None:
            print('裁剪图片')
            self.infoTextEdit.appendPlainText('裁剪图片')
            self.img_sub = sub_img(self.img, self.imgmask)
            self.current_img = self.img_sub.copy()
            self.show_img(self.current_img)

    def Calculate(self):
        """
        计算细胞在视野中所占的比例,细胞数量,细胞区域G-R总值和G-R平均值
        """
        if self.img and self.np_mask is not None:
            ratio, cell_nums, GR_sub, GR_mean, signal_heatmap= calculate(self.img, self.np_mask)
            self.infoTextEdit.appendPlainText(f'细胞在视野中所占面积比例:{ratio * 100:.2f}%')
            self.infoTextEdit.appendPlainText(f'细胞数量: {cell_nums}')
            self.infoTextEdit.appendPlainText(f'细胞区域G-R总值:{GR_sub:.2f},G-R平均值:{GR_mean:.2f}')
            self.current_img = signal_heatmap
            self.show_img(self.current_img)

    def Predict(self):
        """
        预测，并自动将结果显示在界面中
        """
        if self.model_path:
            self.infoTextEdit.appendPlainText('预测中……')
        else:
            self.infoTextEdit.appendPlainText('请选择模型！')
        
        self.predict_thread = PredictThread(self.img_path, self.model_path)
        self.predict_thread.finished_signal.connect(self.on_predict_finished)
        self.predict_thread.start()

    def on_predict_finished(self, mask):
        self.np_mask = np.array(mask)
        self.infoTextEdit.appendPlainText('预测完成')
        self.imgmask = np2mask(self.np_mask)
        self.update_alpha(self.opaque * 100)
        
    def Select_Model(self):
        """
        选择yolov11模型
        """
        self.model_path = QFileDialog.getOpenFileName(self, '选择模型文件', './', '权重文件(*)')[0]
        self.infoTextEdit.appendPlainText('模型路径：' + self.model_path)

    def show_img(self, img):
        """
        传入Image对象,显示在QLabel中
        """
        if img:
            # 转换为QPixmap
            pixmap = ImageQt.toqpixmap(img)

            # 获取标签尺寸
            label_width, label_height = self.lbShowImg.size().width(), self.lbShowImg.size().height()

            # 计算缩放比例
            scale_factor = self.sizeSlider.value() / 100.0
            original_width, original_height = pixmap.width(), pixmap.height()

            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # 缩放图片
            pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 显示图片
            self.lbShowImg.setPixmap(pixmap)

    def resizeEvent(self, event):
        """
        窗口大小改变时，重新显示图片
        """
        super().resizeEvent(event)
        if self.current_img:
            self.show_img(self.current_img)

    def play_video(self):
        """
        播放视频
        """
        if self.media_player:
            self.media_player.play()
            
    def pause_video(self):
        """
        暂停视频
        """
        if self.media_player:
            self.media_player.pause()
            
    def stop_video(self):
        """
        停止视频
        """
        if self.media_player:
            self.media_player.stop()

    def Batch_Process(self):
        """
        批量SPR对齐+信号提取处理
        """
        # 显示配置对话框
        dialog = BatchConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            if config:
                self.start_batch_processing(config)
    
    def start_batch_processing(self, config):
        """
        启动批量处理线程
        """
        # 清空信息显示
        self.infoTextEdit.clear()
        self.infoTextEdit.appendPlainText("=" * 50)
        self.infoTextEdit.appendPlainText("批量SPR对齐+信号提取处理")
        self.infoTextEdit.appendPlainText("=" * 50)
        
        # 创建并启动批量处理线程
        self.batch_thread = BatchProcessThread(config)
        
        # 连接信号
        self.batch_thread.progress_signal.connect(self.on_batch_progress)
        self.batch_thread.finished_signal.connect(self.on_batch_finished)
        self.batch_thread.error_signal.connect(self.on_batch_error)
        
        # 禁用批量处理按钮，防止重复点击
        self.batchProcess.setEnabled(False)
        
        # 启动线程
        self.batch_thread.start()
    
    def on_batch_progress(self, message):
        """
        处理批量处理进度信息
        """
        self.infoTextEdit.appendPlainText(message)
        # 自动滚动到底部
        self.infoTextEdit.verticalScrollBar().setValue(
            self.infoTextEdit.verticalScrollBar().maximum()
        )
    
    def on_batch_finished(self, result_info):
        """
        批量处理完成
        """
        self.infoTextEdit.appendPlainText("\n" + "=" * 50)
        self.infoTextEdit.appendPlainText("处理完成！")
        self.infoTextEdit.appendPlainText("=" * 50)
        
        # 重新启用批量处理按钮
        self.batchProcess.setEnabled(True)
        
        # 显示结果信息对话框
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("批量处理完成")
        msg.setText(f"批量处理已成功完成！\n\n"
                   f"Excel文件: {result_info['excel_path']}\n"
                   f"总帧数: {result_info['total_frames']}\n"
                   f"掩膜数量: {result_info['total_masks']}")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def on_batch_error(self, error_msg):
        """
        批量处理出错
        """
        self.infoTextEdit.appendPlainText("\n" + "=" * 50)
        self.infoTextEdit.appendPlainText("处理出错！")
        self.infoTextEdit.appendPlainText("=" * 50)
        
        # 重新启用批量处理按钮
        self.batchProcess.setEnabled(True)
        
        # 显示错误对话框
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "批量处理错误", error_msg)


if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')
    window = MainWindow()
    window.show()
    app.exec()
