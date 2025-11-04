# -*- coding: utf-8 -*-
"""
批量图像处理线程
用于在后台执行batch_imgs_process4的SPR对齐+信号提取功能
"""

from PySide6.QtCore import QThread, Signal
from pathlib import Path
import sys
import traceback

class BatchProcessThread(QThread):
    """批量处理线程"""
    progress_signal = Signal(str)  # 发送进度信息
    finished_signal = Signal(dict)  # 发送完成信号，包含结果信息
    error_signal = Signal(str)  # 发送错误信息
    
    def __init__(self, config):
        """
        初始化批量处理线程
        
        参数:
            config: dict, 包含以下键值
                - input_dir: 输入图像目录
                - mask_file_path: 掩膜文件路径
                - save_dir: 保存目录
                - force_realign: 是否强制重新对齐
                - force_recompute_signal: 是否强制重新计算信号
                - enable_background: 是否启用背景ROI
                - write_aligned: 是否输出对齐后的图像
                - max_workers: 最大工作线程数
        """
        super().__init__()
        self.config = config
        
    def run(self):
        """执行批量处理"""
        try:
            # 添加项目路径到sys.path
            sys.path.insert(0, r'c:\somefiles\Cell-Segmentation')
            
            # 动态导入batch_imgs_process4模块
            import batch_imgs_process4 as batch
            
            self.progress_signal.emit("开始批量处理...")
            
            # 转换路径为Path对象
            input_dir = Path(self.config['input_dir'])
            mask_file_path = Path(self.config['mask_file_path'])
            save_dir = Path(self.config['save_dir'])
            
            # 创建保存目录
            save_dir.mkdir(parents=True, exist_ok=True)
            
            self.progress_signal.emit(f"输入目录: {input_dir}")
            self.progress_signal.emit(f"掩膜文件: {mask_file_path}")
            self.progress_signal.emit(f"输出目录: {save_dir}")
            
            # Stage A: 图像对齐
            self.progress_signal.emit("\n=== Stage A: 图像对齐 ===")
            align_meta = batch.compute_or_load_alignment(
                input_dir=input_dir,
                force_realign=self.config.get('force_realign', False),
                write_aligned=self.config.get('write_aligned', False),
                output_suffix=self.config.get('output_suffix', '_aligned_opt_v4')
            )
            
            self.progress_signal.emit("图像对齐完成")
            crop = align_meta['crop']
            self.progress_signal.emit(f"公共裁剪区域: x0={crop['x0']}, y0={crop['y0']}, w={crop['w']}, h={crop['h']}")
            
            # Stage B: 信号提取
            self.progress_signal.emit("\n=== Stage B: 信号提取 ===")
            result = batch.spr_signal_extraction_v4(
                input_dir=input_dir,
                mask_file_path=mask_file_path,
                save_dir=save_dir,
                align_meta=align_meta,
                max_workers=self.config.get('max_workers', 4),
                force_recompute=self.config.get('force_recompute_signal', False),
                enable_background=self.config.get('enable_background', True)
            )
            
            self.progress_signal.emit("信号提取完成")
            
            # 保存结果到Excel
            self.progress_signal.emit("\n=== 保存结果 ===")
            batch.save_results_to_excel_v4(
                result=result,
                save_dir=save_dir,
                mask_file_path=mask_file_path,
                align_meta=align_meta
            )
            
            # 准备返回的结果信息
            out_dir = Path(align_meta["output_dir"])
            excel_name = out_dir.name + f"_{mask_file_path.stem}.xlsx"
            excel_path = save_dir / excel_name
            
            result_info = {
                'success': True,
                'excel_path': str(excel_path),
                'cache_dir': str(result['cache_dir']),
                'progress_csv': str(result['progress_csv']),
                'mean_shape': result['mean'].shape,
                'total_frames': len(result['mean']),
                'total_masks': len(result['mean'].columns)
            }
            
            self.progress_signal.emit(f"\n处理完成！")
            self.progress_signal.emit(f"Excel文件: {excel_path}")
            self.progress_signal.emit(f"总帧数: {result_info['total_frames']}")
            self.progress_signal.emit(f"掩膜数量: {result_info['total_masks']}")
            
            self.finished_signal.emit(result_info)
            
        except Exception as e:
            error_msg = f"批量处理出错:\n{str(e)}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)
            self.progress_signal.emit(f"\n错误: {error_msg}")
