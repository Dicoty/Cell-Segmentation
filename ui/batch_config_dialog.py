# -*- coding: utf-8 -*-
"""
批量处理配置对话框
用于设置batch_imgs_process4的参数
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QLineEdit, QFileDialog, QCheckBox,
                               QSpinBox, QGroupBox, QFormLayout)
from PySide6.QtCore import Qt
import multiprocessing as mp

class BatchConfigDialog(QDialog):
    """批量处理配置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('批量SPR对齐+信号提取配置')
        self.setMinimumWidth(600)
        self.config = None
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 路径配置组
        path_group = QGroupBox("路径配置")
        path_layout = QFormLayout()
        
        # 输入目录
        input_layout = QHBoxLayout()
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("选择包含图像序列的文件夹")
        input_btn = QPushButton("浏览...")
        input_btn.clicked.connect(self.select_input_dir)
        input_layout.addWidget(self.input_dir_edit)
        input_layout.addWidget(input_btn)
        path_layout.addRow("输入目录:", input_layout)
        
        # 掩膜文件
        mask_layout = QHBoxLayout()
        self.mask_file_edit = QLineEdit()
        self.mask_file_edit.setPlaceholderText("选择掩膜文件 (.npy)")
        mask_btn = QPushButton("浏览...")
        mask_btn.clicked.connect(self.select_mask_file)
        mask_layout.addWidget(self.mask_file_edit)
        mask_layout.addWidget(mask_btn)
        path_layout.addRow("掩膜文件:", mask_layout)
        
        # 保存目录
        save_layout = QHBoxLayout()
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("选择结果保存目录")
        save_btn = QPushButton("浏览...")
        save_btn.clicked.connect(self.select_save_dir)
        save_layout.addWidget(self.save_dir_edit)
        save_layout.addWidget(save_btn)
        path_layout.addRow("保存目录:", save_layout)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # 处理选项组
        options_group = QGroupBox("处理选项")
        options_layout = QFormLayout()
        
        # 强制重新对齐
        self.force_realign_check = QCheckBox()
        self.force_realign_check.setToolTip("勾选后将重新计算图像对齐，忽略已有的对齐结果")
        options_layout.addRow("强制重新对齐:", self.force_realign_check)
        
        # 输出对齐图像
        self.write_aligned_check = QCheckBox()
        self.write_aligned_check.setToolTip("勾选后将保存对齐+裁剪后的图像用于核验")
        options_layout.addRow("输出对齐图像:", self.write_aligned_check)
        
        # 强制重新计算信号
        self.force_recompute_check = QCheckBox()
        self.force_recompute_check.setToolTip("勾选后将忽略进度CSV，强制重新计算全部帧")
        options_layout.addRow("强制重新计算:", self.force_recompute_check)
        
        # 启用背景ROI
        self.enable_background_check = QCheckBox()
        self.enable_background_check.setChecked(True)
        self.enable_background_check.setToolTip("勾选后可以选择背景ROI区域")
        options_layout.addRow("启用背景ROI:", self.enable_background_check)
        
        # 最大工作线程数
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setMinimum(1)
        self.max_workers_spin.setMaximum(32)
        self.max_workers_spin.setValue(max(2, mp.cpu_count()//2))
        self.max_workers_spin.setToolTip("并行处理的线程数量")
        options_layout.addRow("工作线程数:", self.max_workers_spin)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.ok_btn = QPushButton("开始处理")
        self.ok_btn.clicked.connect(self.accept_config)
        self.ok_btn.setDefault(True)
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def select_input_dir(self):
        """选择输入目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "选择输入图像目录",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if dir_path:
            self.input_dir_edit.setText(dir_path)
            
    def select_mask_file(self):
        """选择掩膜文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择掩膜文件",
            "",
            "Numpy文件 (*.npy);;所有文件 (*)"
        )
        if file_path:
            self.mask_file_edit.setText(file_path)
            
    def select_save_dir(self):
        """选择保存目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择保存目录",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if dir_path:
            self.save_dir_edit.setText(dir_path)
            
    def accept_config(self):
        """验证并接受配置"""
        # 验证必填项
        if not self.input_dir_edit.text():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请选择输入目录！")
            return
            
        if not self.mask_file_edit.text():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请选择掩膜文件！")
            return
            
        if not self.save_dir_edit.text():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请选择保存目录！")
            return
            
        # 收集配置
        self.config = {
            'input_dir': self.input_dir_edit.text(),
            'mask_file_path': self.mask_file_edit.text(),
            'save_dir': self.save_dir_edit.text(),
            'force_realign': self.force_realign_check.isChecked(),
            'write_aligned': self.write_aligned_check.isChecked(),
            'force_recompute_signal': self.force_recompute_check.isChecked(),
            'enable_background': self.enable_background_check.isChecked(),
            'max_workers': self.max_workers_spin.value(),
            'output_suffix': '_aligned_opt_v4'
        }
        
        self.accept()
        
    def get_config(self):
        """获取配置"""
        return self.config
