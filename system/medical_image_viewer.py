import sys
import os
import numpy as np
import matplotlib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                             QComboBox, QSlider, QMessageBox, QGroupBox,
                             QRadioButton, QButtonGroup, QToolBar, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPen, QBrush, QColor, QPainter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# 设置中文字体
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在文件开头添加
os.environ['TORCH_HOME'] = './weights'  # 设置自定义缓存目录
os.environ['PYTORCH_NO_DOWNLOAD'] = '1'  # 尝试禁用自动下载

# 导入我们的医学图像处理类
from system.medical_image_utils import MedicalImageProcessor, list_available_models

class InteractiveCanvas(FigureCanvas):
    """支持点击和框选的交互式画布"""
    pointAdded = pyqtSignal(float, float, int)  # x, y, label (1=前景, 0=背景)
    boxDrawn = pyqtSignal(list)  # [x1, y1, x2, y2]
    
    def __init__(self, figure=None):
        if figure is None:
            figure = Figure(figsize=(5, 5), dpi=100)
        super().__init__(figure)
        self.axes = figure.add_subplot(111)
        self.axes.axis('off')
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMouseTracking(True)
        
        # 绑定事件
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # 交互状态
        self.box_mode = False
        self.drawing_box = False
        self.start_x = None
        self.start_y = None
        self.foreground_point = True  # True=前景点, False=背景点
        
        # 可视化元素
        self.points_plotted = []  # 存储已绘制的点
        self.box_rect = None
        self.start_marker = None
        self.end_marker = None
        self.point_size = 10
        
        # 数据
        self.points = []  # 存储 (x, y, label) 三元组
        self.current_image = None
        
    def display_image(self, img):
        """显示图像"""
        self.axes.clear()
        self.points = []
        self.points_plotted = []
        self.current_image = img
        
        if len(img.shape) == 3:  # 彩色图像
            self.axes.imshow(img)
        else:  # 灰度图像
            self.axes.imshow(img, cmap='gray')
            
        self.axes.axis('off')
        self.draw_idle()
        
    def display_mask(self, mask, alpha=0.3):
        """显示分割掩码"""
        if mask is None or self.current_image is None:
            return
            
        # 显示掩码叠加
        self.axes.imshow(mask, alpha=alpha, cmap='viridis')
        self.draw_idle()
        
    def set_box_mode(self, enabled):
        """设置是否处于框选模式"""
        self.box_mode = enabled
        if not enabled:
            self.clear_box()
            
    def set_foreground_point(self, is_foreground):
        """设置是否为前景点"""
        self.foreground_point = is_foreground
        
    def clear_points(self):
        """清除所有点"""
        for point in self.points_plotted:
            if point in self.axes.lines:
                point.remove()
        self.points_plotted = []
        self.points = []
        self.draw_idle()
        
    def clear_box(self):
        """清除框"""
        if self.box_rect and self.box_rect in self.axes.patches:
            self.box_rect.remove()
            self.box_rect = None
            
        if self.start_marker and self.start_marker in self.axes.patches:
            self.start_marker.remove()
            self.start_marker = None
            
        if self.end_marker and self.end_marker in self.axes.patches:
            self.end_marker.remove()
            self.end_marker = None
            
        self.drawing_box = False
        self.start_x = None
        self.start_y = None
        self.draw_idle()
        
    def on_mouse_press(self, event):
        """鼠标按下事件"""
        if event.inaxes != self.axes or self.current_image is None:
            return
            
        if self.box_mode:
            # 框选模式
            self.drawing_box = True
            self.start_x, self.start_y = event.xdata, event.ydata
            
            # 绘制起始点
            if self.start_marker and self.start_marker in self.axes.patches:
                self.start_marker.remove()
                
            self.start_marker = plt.Circle(
                (self.start_x, self.start_y), 
                radius=self.point_size/2, 
                color='red',
                fill=True,
                alpha=0.7
            )
            self.axes.add_patch(self.start_marker)
            self.draw_idle()
        else:
            # 点击模式
            x, y = event.xdata, event.ydata
            
            # 根据鼠标按键和设置确定标签
            if event.button == 1:  # 左键
                label = 1 if self.foreground_point else 0
            elif event.button == 3:  # 右键
                label = 0 if self.foreground_point else 1
            else:
                return
                
            # 绘制点
            color = 'green' if label == 1 else 'red'
            point = self.axes.plot(
                x, y, 'o', 
                markersize=self.point_size,
                markeredgecolor='black',
                markerfacecolor=color,
                alpha=0.7
            )[0]
            
            self.points_plotted.append(point)
            self.points.append((x, y, label))
            self.pointAdded.emit(x, y, label)
            self.draw_idle()
        
    def on_mouse_move(self, event):
        """鼠标移动事件"""
        if not self.drawing_box or not self.box_mode or event.inaxes != self.axes:
            return
            
        # 更新结束点
        if self.end_marker and self.end_marker in self.axes.patches:
            self.end_marker.remove()
            
        self.end_marker = plt.Circle(
            (event.xdata, event.ydata), 
            radius=self.point_size/2, 
            color='red',
            fill=True,
            alpha=0.7
        )
        self.axes.add_patch(self.end_marker)
        
        # 更新矩形
        if self.box_rect and self.box_rect in self.axes.patches:
            self.box_rect.remove()
            
        x = min(self.start_x, event.xdata)
        y = min(self.start_y, event.ydata)
        width = abs(self.start_x - event.xdata)
        height = abs(self.start_y - event.ydata)
        
        self.box_rect = mpatches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            alpha=0.7
        )
        self.axes.add_patch(self.box_rect)
        self.draw_idle()
        
    def on_mouse_release(self, event):
        """鼠标释放事件"""
        if not self.drawing_box or not self.box_mode or event.inaxes != self.axes:
            return
            
        self.drawing_box = False
        
        # 计算框坐标
        x1 = min(self.start_x, event.xdata)
        y1 = min(self.start_y, event.ydata)
        x2 = max(self.start_x, event.xdata)
        y2 = max(self.start_y, event.ydata)
        
        # 发送信号
        self.boxDrawn.emit([x1, y1, x2, y2])


class MedicalImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口属性
        self.setWindowTitle("医学图像分割应用")
        self.resize(1200, 800)
        
        # 状态变量
        self.processor = MedicalImageProcessor()
        self.available_models = list_available_models()
        self.mask = None
        self.points = []  # 点提示列表 [(x1, y1, label1), (x2, y2, label2), ...]
        self.point_labels = []  # 点标签列表 [label1, label2, ...]
        self.box = None  # 框提示 [x1, y1, x2, y2]
        self.current_slice = 0  # 当前3D图像切片
        
        # 初始化UI
        self.initUI()
        
    def initUI(self):
        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 创建工具栏
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)
        
        # 添加文件操作按钮
        open_btn = QPushButton("打开图像")
        open_btn.clicked.connect(self.open_image)
        toolbar.addWidget(open_btn)
        
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        toolbar.addWidget(self.save_btn)
        
        toolbar.addSeparator()
        
        # 添加模型选择器
        model_label = QLabel("选择模型:")
        toolbar.addWidget(model_label)
        
        self.model_selector = QComboBox()
        for model_name, model_info in self.available_models.items():
            self.model_selector.addItem(f"{model_name}: {model_info['description']}")
            
        toolbar.addWidget(self.model_selector)
        self.model_selector.currentIndexChanged.connect(self.on_model_changed)
        
        toolbar.addSeparator()
        
        # 添加分割按钮
        segment_btn = QPushButton("分割")
        segment_btn.clicked.connect(self.segment_image)
        toolbar.addWidget(segment_btn)
        
        # 创建图像显示区域
        display_layout = QHBoxLayout()
        
        # 原始图像区域
        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout()
        
        # 创建交互式画布显示原始图像
        original_fig = Figure(figsize=(5, 5), dpi=100)
        self.original_view = InteractiveCanvas(original_fig)
        self.original_view.pointAdded.connect(self.on_point_added)
        self.original_view.boxDrawn.connect(self.on_box_drawn)
        
        original_layout.addWidget(self.original_view)
        
        # 创建点提示控制区
        prompt_controls_layout = QHBoxLayout()
        
        # 点提示类型
        self.point_type_group = QButtonGroup(self)
        self.fg_radio = QRadioButton("前景点 (左键)")
        self.fg_radio.setChecked(True)
        self.bg_radio = QRadioButton("背景点 (右键)")
        self.point_type_group.addButton(self.fg_radio, 1)
        self.point_type_group.addButton(self.bg_radio, 0)
        self.point_type_group.buttonClicked.connect(self.on_point_type_changed)
        
        prompt_controls_layout.addWidget(self.fg_radio)
        prompt_controls_layout.addWidget(self.bg_radio)
        
        # 添加框选模式按钮
        self.box_mode_btn = QPushButton("框选模式 📦")
        self.box_mode_btn.setCheckable(True)
        self.box_mode_btn.setChecked(False)
        self.box_mode_btn.clicked.connect(self.on_box_mode_clicked)
        self.box_mode_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                height: 30px;
                padding: 0 10px;
            }
            QPushButton:checked {
                background-color: #AED6F1;
                border: 2px solid #3498DB;
            }
        """)
        prompt_controls_layout.addWidget(self.box_mode_btn)
        
        # 清除点提示按钮
        self.clear_points_btn = QPushButton("清除点")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)
        prompt_controls_layout.addWidget(self.clear_points_btn)
        
        # 清除框按钮
        self.clear_box_btn = QPushButton("清除框")
        self.clear_box_btn.clicked.connect(self.clear_box)
        self.clear_box_btn.setEnabled(False)
        prompt_controls_layout.addWidget(self.clear_box_btn)
        
        original_layout.addLayout(prompt_controls_layout)
        
        # 3D图像切片滑块
        self.slice_slider_container = QWidget()
        slice_layout = QHBoxLayout(self.slice_slider_container)
        slice_label = QLabel("切片:")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.valueChanged.connect(self.update_slice)
        slice_layout.addWidget(slice_label)
        slice_layout.addWidget(self.slice_slider)
        self.slice_slider_container.setVisible(False)
        
        original_layout.addWidget(self.slice_slider_container)
        original_group.setLayout(original_layout)
        
        # 结果显示区域
        result_group = QGroupBox("分割结果")
        result_layout = QVBoxLayout()
        
        # 创建结果图像显示
        result_fig = Figure(figsize=(5, 5), dpi=100)
        self.result_canvas = FigureCanvas(result_fig)
        self.result_ax = result_fig.add_subplot(111)
        self.result_ax.axis('off')
        
        result_layout.addWidget(self.result_canvas)
        result_group.setLayout(result_layout)
        
        # 添加两个显示区域
        display_layout.addWidget(original_group)
        display_layout.addWidget(result_group)
        
        # 添加图像显示区域到主布局
        main_layout.addLayout(display_layout)
        
        # 设置中央部件
        self.setCentralWidget(main_widget)
        
        # 默认隐藏点提示控件，直到选择MedSAM模型
        self.prompt_controls_widget = QWidget()
        self.prompt_controls_widget.setLayout(prompt_controls_layout)
        self.prompt_controls_widget.setVisible(False)
        original_layout.addWidget(self.prompt_controls_widget)
        
        # 初始检查模型选择
        self.on_model_changed(self.model_selector.currentIndex())
        
    def open_image(self):
        """打开并显示图像"""
        options = QFileDialog.Options()
        file_types = "医学图像 (*.mha *.nii *.nii.gz *.tif *.jpg *.png);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "打开医学图像", "", file_types, options=options)
        
        if not file_path:
            return
            
        try:
            # 加载图像
            self.processor.load_image(file_path)
            
            # 清除之前的分割结果
            self.mask = None
            self.points = []
            self.point_labels = []
            self.box = None
            self.clear_points_btn.setEnabled(False)
            self.clear_box_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            
            # 设置3D图像切片控制
            if self.processor.is_3d:
                self.slice_slider_container.setVisible(True)
                self.slice_slider.setMinimum(0)
                self.slice_slider.setMaximum(self.processor.image_data.shape[0] - 1)
                self.slice_slider.setValue(0)
                self.current_slice = 0
                
                # 显示初始切片
                self.original_view.display_image(self.processor.image_data[0])
            else:
                self.slice_slider_container.setVisible(False)
                self.original_view.display_image(self.processor.image_data)
                
            # 清除结果显示
            self.result_ax.clear()
            self.result_ax.axis('off')
            self.result_canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像时出错: {str(e)}")
            
    def on_point_type_changed(self, button):
        """当点提示类型改变时调用"""
        is_foreground = button.text().startswith("前景")
        self.original_view.set_foreground_point(is_foreground)
            
    def on_box_mode_clicked(self):
        """当框选模式按钮点击时调用"""
        is_checked = self.box_mode_btn.isChecked()
        self.original_view.set_box_mode(is_checked)
        
        # 更新UI状态
        if is_checked:
            self.box_mode_btn.setText("点击模式 👆")
            self.fg_radio.setEnabled(False)
            self.bg_radio.setEnabled(False)
        else:
            self.box_mode_btn.setText("框选模式 📦")
            self.fg_radio.setEnabled(True)
            self.bg_radio.setEnabled(True)
    
    def on_model_changed(self, index):
        """当模型选择改变时调用"""
        selected_text = self.model_selector.currentText()
        model_name = selected_text.split(':')[0]
        
        # 如果选择的是MedSAM模型，显示点提示控件
        self.prompt_controls_widget.setVisible(model_name == 'medsam')
        
        # 清除当前的点提示和框
        self.clear_points()
        self.clear_box()
    
    def on_point_added(self, x, y, label):
        """当添加一个点时调用"""
        # 存储点和标签
        self.points.append((x, y))
        self.point_labels.append(label)
        
        # 更新UI状态
        self.clear_points_btn.setEnabled(True)
        
    def on_box_drawn(self, box):
        """当绘制一个框时调用"""
        # 存储框坐标
        self.box = box
        
        # 更新UI状态
        self.clear_box_btn.setEnabled(True)
        
    def update_slice(self, slice_index):
        """更新3D图像的当前切片"""
        if not self.processor.is_3d:
            return
            
        self.current_slice = slice_index
        
        # 显示当前切片
        self.original_view.display_image(self.processor.image_data[slice_index])
        
        # 如果有分割结果，也更新结果显示
        if self.mask is not None:
            self.display_result()
            
    def clear_points(self):
        """清除所有点提示"""
        self.points = []
        self.point_labels = []
        self.original_view.clear_points()
        self.clear_points_btn.setEnabled(False)
        
    def clear_box(self):
        """清除框提示"""
        self.box = None
        self.original_view.clear_box()
        self.clear_box_btn.setEnabled(False)
        
    def segment_image(self):
        """对当前图像进行分割"""
        # 获取所选模型
        selected_text = self.model_selector.currentText()
        model_name = selected_text.split(':')[0]
        
        try:
            # 设置分割模型
            if model_name == 'medsam':
                # 检查是否有点提示或框提示
                if not self.points and self.box is None:
                    QMessageBox.warning(self, "提示", "请在图像上添加提示点或绘制框")
                    return
                
                # 设置模型
                self.processor.set_segmentation_model(
                    model_name='medsam',
                    model_type='vit_b',
                    checkpoint_path=self.available_models['medsam']['weights_path']
                )
                
                # 准备模型输入
                points_array = np.array(self.points) if self.points else None
                labels_array = np.array(self.point_labels) if self.point_labels else None
                box_array = np.array(self.box) if self.box else None
                
                print(f"使用以下提示进行分割: 点={self.points}, 标签={self.point_labels}, 框={self.box}")
                
                # 分割图像
                if self.processor.is_3d:
                    # 对当前切片进行分割
                    slice_img = self.processor.image_data[self.current_slice]
                    self.mask = np.zeros_like(self.processor.image_data, dtype=bool)
                    slice_mask = self.processor.segmenter.segment(
                        slice_img, 
                        points=points_array, 
                        point_labels=labels_array,
                        box=box_array
                    )
                    self.mask[self.current_slice] = slice_mask
                else:
                    # 对2D图像进行分割
                    self.mask = self.processor.segmenter.segment(
                        self.processor.image_data,
                        points=points_array,
                        point_labels=labels_array,
                        box=box_array
                    )
                
            elif model_name.startswith('deeplabv3'):
                # 设置模型
                self.processor.set_segmentation_model(
                    model_name=model_name,
                    num_classes=21,
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
            if len(self.processor.image_data[self.current_slice].shape) == 3:
                self.result_ax.imshow(self.processor.image_data[self.current_slice])
            else:
                self.result_ax.imshow(self.processor.image_data[self.current_slice], cmap='gray')
                
            # 叠加分割掩码
            mask_slice = self.mask[self.current_slice] if self.mask is not None else None
            if mask_slice is not None:
                # 创建带轮廓的遮罩显示
                from skimage import measure
                contours = measure.find_contours(mask_slice, 0.5)
                for contour in contours:
                    self.result_ax.plot(contour[:, 1], contour[:, 0], 'y-', linewidth=2)
                
                self.result_ax.imshow(mask_slice, alpha=0.3, cmap='viridis')
            self.result_ax.set_title(f'分割结果 (切片 {self.current_slice})')
        else:
            # 显示原图
            if len(self.processor.image_data.shape) == 3 and self.processor.image_data.shape[2] == 3:
                self.result_ax.imshow(self.processor.image_data)
            else:
                self.result_ax.imshow(self.processor.image_data, cmap='gray')
                
            # 叠加分割掩码
            if self.mask is not None:
                # 创建带轮廓的遮罩显示
                from skimage import measure
                contours = measure.find_contours(self.mask, 0.5)
                for contour in contours:
                    self.result_ax.plot(contour[:, 1], contour[:, 0], 'y-', linewidth=2)
                
                self.result_ax.imshow(self.mask, alpha=0.3, cmap='viridis')
            self.result_ax.set_title('分割结果')
        
        self.result_ax.axis('off')
        self.result_canvas.draw()
        
    def save_result(self):
        """保存分割结果"""
        if self.mask is None:
            return
        
        options = QFileDialog.Options()
        file_types = "医学图像 (*.mha *.nii *.nii.gz *.tif *.jpg *.png);;所有文件 (*)"
        file_path, _ = QFileDialog.getSaveFileName(self, "保存分割结果", "", file_types, options=options)
        
        if file_path:
            try:
                result = self.processor.save_segmentation_result(self.mask, file_path)
                if result:
                    QMessageBox.information(self, "成功", f"分割结果已保存到: {file_path}")
                else:
                    QMessageBox.warning(self, "警告", "保存分割结果失败")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存分割结果时出错: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalImageApp()
    window.show()
    sys.exit(app.exec_()) 