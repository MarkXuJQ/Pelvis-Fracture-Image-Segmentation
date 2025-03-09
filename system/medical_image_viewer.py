import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                             QComboBox, QSlider, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import matplotlib.font_manager as fm
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']  # 添加中文字体支持

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们的医学图像处理类
from system.medical_image_utils import MedicalImageProcessor, list_available_models

# 在文件开头添加
os.environ['TORCH_HOME'] = './weights'  # 设置自定义缓存目录
os.environ['PYTORCH_NO_DOWNLOAD'] = '1'  # 尝试禁用自动下载

# 查看所有可用字体
font_list = [f.name for f in fm.fontManager.ttflist]
print(font_list)

class MedicalImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = MedicalImageProcessor()
        self.available_models = list_available_models()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('医学图像处理器')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 创建顶部控制区
        controls_layout = QHBoxLayout()
        
        # 添加打开图像按钮
        self.open_btn = QPushButton('打开图像')
        self.open_btn.clicked.connect(self.open_image)
        controls_layout.addWidget(self.open_btn)
        
        # 添加模型选择下拉框
        self.model_selector = QComboBox()
        for model_name, info in self.available_models.items():
            self.model_selector.addItem(f"{model_name}: {info['description']} ({info['status']})")
        controls_layout.addWidget(self.model_selector)
        
        # 添加分割按钮
        self.segment_btn = QPushButton('分割图像')
        self.segment_btn.clicked.connect(self.segment_image)
        self.segment_btn.setEnabled(False)  # 初始禁用
        controls_layout.addWidget(self.segment_btn)
        
        # 添加保存结果按钮
        self.save_btn = QPushButton('保存结果')
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)  # 初始禁用
        controls_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(controls_layout)
        
        # 创建图像显示区
        display_layout = QHBoxLayout()
        
        # 原始图像区
        self.original_fig = Figure(figsize=(5, 4), dpi=100)
        self.original_canvas = FigureCanvas(self.original_fig)
        self.original_ax = self.original_fig.add_subplot(111)
        self.original_ax.set_title('原始图像')
        
        original_group = QGroupBox('原始图像')
        original_layout = QVBoxLayout()
        original_layout.addWidget(self.original_canvas)
        
        # 添加3D图像切片滑块
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.update_slice)
        original_layout.addWidget(self.slice_slider)
        
        original_group.setLayout(original_layout)
        display_layout.addWidget(original_group)
        
        # 分割结果区
        self.result_fig = Figure(figsize=(5, 4), dpi=100)
        self.result_canvas = FigureCanvas(self.result_fig)
        self.result_ax = self.result_fig.add_subplot(111)
        self.result_ax.set_title('分割结果')
        
        result_group = QGroupBox('分割结果')
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_canvas)
        result_group.setLayout(result_layout)
        display_layout.addWidget(result_group)
        
        main_layout.addLayout(display_layout)
        
        self.setCentralWidget(main_widget)
        
        # 初始化成员变量
        self.current_slice = 0
        self.mask = None
        
    def open_image(self):
        """打开图像文件对话框并加载图像"""
        options = QFileDialog.Options()
        file_types = "医学图像 (*.mha *.nii *.nii.gz *.tif *.jpg *.png *.bmp *.jpeg);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择医学图像", "", file_types, options=options)
        
        if file_path:
            # 加载图像
            if self.processor.load_image(file_path):
                # 显示图像
                self.display_image()
                
                # 如果是3D图像，启用切片滑块
                if self.processor.is_3d:
                    self.slice_slider.setEnabled(True)
                    self.slice_slider.setMinimum(0)
                    self.slice_slider.setMaximum(self.processor.image_data.shape[0] - 1)
                    self.slice_slider.setValue(self.processor.image_data.shape[0] // 2)
                    self.current_slice = self.processor.image_data.shape[0] // 2
                else:
                    self.slice_slider.setEnabled(False)
                
                # 启用分割按钮
                self.segment_btn.setEnabled(True)
                
                # 显示图像信息
                self.statusBar().showMessage(f"已加载图像: {file_path}")
            else:
                QMessageBox.critical(self, "错误", "无法加载图像文件")
    
    def display_image(self):
        """在界面上显示当前图像"""
        self.original_ax.clear()
        
        if self.processor.is_3d:
            self.original_ax.imshow(self.processor.image_data[self.current_slice], cmap='gray')
            self.original_ax.set_title(f'原始图像 (切片 {self.current_slice}/{self.processor.image_data.shape[0]-1})')
        else:
            if len(self.processor.image_data.shape) == 3 and self.processor.image_data.shape[2] == 3:
                self.original_ax.imshow(self.processor.image_data)
            else:
                self.original_ax.imshow(self.processor.image_data, cmap='gray')
            self.original_ax.set_title('原始图像')
        
        self.original_canvas.draw()
    
    def update_slice(self, value):
        """更新3D图像的当前切片"""
        self.current_slice = value
        self.display_image()
        
        # 如果有分割结果，也更新结果显示
        if self.mask is not None:
            self.display_result()
    
    def segment_image(self):
        """对当前图像进行分割"""
        # 获取所选模型
        selected_text = self.model_selector.currentText()
        model_name = selected_text.split(':')[0]
        
        try:
            # 设置分割模型
            if model_name == 'medsam':
                # 对于MedSAM模型，我们需要点击点来指导分割
                # 简单起见，这里使用图像中心作为标记点
                if self.processor.is_3d:
                    height, width = self.processor.image_data[self.current_slice].shape
                else:
                    if len(self.processor.image_data.shape) == 3 and self.processor.image_data.shape[2] == 3:
                        height, width, _ = self.processor.image_data.shape
                    else:
                        height, width = self.processor.image_data.shape
                
                center_point = [[width // 2, height // 2]]
                
                # 设置模型
                self.processor.set_segmentation_model(
                    model_name='medsam',
                    model_type='vit_b',
                    checkpoint_path=self.available_models['medsam']['weights_path']
                )
                
                # 分割图像
                self.mask = self.processor.segment_image(points=center_point, point_labels=[1])
                
            elif model_name.startswith('deeplabv3'):
                # 设置模型
                self.processor.set_segmentation_model(
                    model_name=model_name,  # 直接使用完整模型名称（包含骨干网络信息）
                    num_classes=21,  # 默认使用VOC数据集的类别数
                    checkpoint_path=self.available_models[model_name]['weights_path']
                )
                
                # 分割图像
                self.mask = self.processor.segment_image(target_class=1)
            
            # 显示分割结果
            if self.mask is not None:
                self.display_result()
                self.save_btn.setEnabled(True)
            
        except Exception as e:
            import traceback
            traceback.print_exc()  # 打印完整错误堆栈
            QMessageBox.critical(self, "错误", f"分割过程中出错: {str(e)}")
    
    def display_result(self):
        """显示分割结果"""
        self.result_ax.clear()
        
        if self.processor.is_3d:
            # 显示原图
            self.result_ax.imshow(self.processor.image_data[self.current_slice], cmap='gray')
            # 叠加分割掩码
            self.result_ax.imshow(self.mask[self.current_slice], alpha=0.5, cmap='viridis')
            self.result_ax.set_title(f'分割结果 (切片 {self.current_slice})')
        else:
            # 显示原图
            if len(self.processor.image_data.shape) == 3 and self.processor.image_data.shape[2] == 3:
                self.result_ax.imshow(self.processor.image_data)
            else:
                self.result_ax.imshow(self.processor.image_data, cmap='gray')
            # 叠加分割掩码
            self.result_ax.imshow(self.mask, alpha=0.5, cmap='viridis')
            self.result_ax.set_title('分割结果')
        
        self.result_canvas.draw()
    
    def save_result(self):
        """保存分割结果"""
        if self.mask is None:
            return
        
        options = QFileDialog.Options()
        file_types = "医学图像 (*.mha *.nii *.nii.gz *.tif *.jpg *.png);;所有文件 (*)"
        file_path, _ = QFileDialog.getSaveFileName(self, "保存分割结果", "", file_types, options=options)
        
        if file_path:
            if self.processor.save_segmentation_result(self.mask, file_path):
                QMessageBox.information(self, "成功", f"分割结果已保存到: {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalImageApp()
    window.show()
    sys.exit(app.exec_()) 