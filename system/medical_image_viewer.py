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
import matplotlib.colors as mcolors

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
        
        # 添加存储已绘制框的列表
        self.box_rects = []  # 存储已绘制的所有框
        self.box_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        self.current_color_idx = 0  # 当前使用的颜色索引
        
    def display_image(self, img):
        """显示图像"""
        self.axes.clear()
        self.points = []
        self.points_plotted = []
        self.box_rects = []  # 清除已绘制的框
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
            
        # 清除所有已绘制的框
        for rect in self.box_rects:
            if rect in self.axes.patches:
                rect.remove()
        self.box_rects = []
            
        self.drawing_box = False
        self.start_x = None
        self.start_y = None
        self.draw_idle()
        
    def set_current_color_index(self, idx):
        """设置当前使用的颜色索引"""
        self.current_color_idx = idx % len(self.box_colors)
        
    def get_current_color(self):
        """获取当前使用的颜色"""
        return self.box_colors[self.current_color_idx]
        
    def on_mouse_press(self, event):
        """鼠标按下事件"""
        if event.inaxes != self.axes or self.current_image is None:
            return
            
        if self.box_mode:
            # 框选模式
            self.drawing_box = True
            self.start_x, self.start_y = event.xdata, event.ydata
            
            # 获取当前颜色
            current_color = self.get_current_color()
            
            # 绘制起始点
            if self.start_marker and self.start_marker in self.axes.patches:
                self.start_marker.remove()
                
            self.start_marker = plt.Circle(
                (self.start_x, self.start_y), 
                radius=self.point_size/2, 
                color=current_color,
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
            
        # 获取当前颜色
        current_color = self.get_current_color()
        
        # 更新结束点
        if self.end_marker and self.end_marker in self.axes.patches:
            self.end_marker.remove()
            
        self.end_marker = plt.Circle(
            (event.xdata, event.ydata), 
            radius=self.point_size/2, 
            color=current_color,
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
            edgecolor=current_color,
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
        
    def draw_saved_box(self, box, color_idx=0):
        """绘制保存的框"""
        x1, y1, x2, y2 = box
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 选择颜色
        color = self.box_colors[color_idx % len(self.box_colors)]
        
        # 绘制框
        rect = mpatches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.7
        )
        self.axes.add_patch(rect)
        self.box_rects.append(rect)
        self.draw_idle()
        
        # 返回使用的颜色
        return color
        
    def draw_all_boxes(self, boxes):
        """绘制所有保存的框"""
        # 先清除之前的框
        for rect in self.box_rects:
            if rect in self.axes.patches:
                rect.remove()
        self.box_rects = []
        
        # 绘制所有框
        for i, box in enumerate(boxes):
            self.draw_saved_box(box, i)


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
        self.boxes = []  # 框提示列表 [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        self.box_colors = []  # 框颜色
        self.box_masks = []  # 每个框对应的掩码
        self.current_slice = 0  # 当前3D图像切片
        
        # 添加三视图相关变量
        self.current_view = 'axial'  # 当前活动视图: 'axial', 'coronal', 'sagittal'
        self.axial_slice = 0
        self.coronal_slice = 0
        self.sagittal_slice = 0
        
        # 初始化UI
        self.initUI()
        
    def initUI(self):
        # 创建中央部件和主布局
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # 创建左侧控制面板
        control_panel = QWidget()
        control_panel.setFixedWidth(250)  # 设置控制面板宽度
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)
        
        # 1. 文件操作按钮
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        open_btn = QPushButton("打开图像")
        open_btn.clicked.connect(self.open_image)
        file_layout.addWidget(open_btn)
        
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)
        
        control_layout.addWidget(file_group)
        
        # 2. 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        self.model_selector = QComboBox()
        for model_name, model_info in self.available_models.items():
            self.model_selector.addItem(f"{model_name}: {model_info['description']}")
        self.model_selector.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_selector)
        
        control_layout.addWidget(model_group)
        
        # 3. 3D切面选择 (初始隐藏)
        self.view_group = QGroupBox("切面选择")
        view_layout = QVBoxLayout(self.view_group)
        
        # 视图类型选择
        view_type_layout = QHBoxLayout()
        view_type_label = QLabel("视图类型:")
        self.view_type_combo = QComboBox()
        self.view_type_combo.addItems(["轴状视图", "冠状视图", "矢状视图"])
        self.view_type_combo.currentIndexChanged.connect(self.on_view_type_changed)
        view_type_layout.addWidget(view_type_label)
        view_type_layout.addWidget(self.view_type_combo)
        view_layout.addLayout(view_type_layout)
        
        # 切片滑块
        slice_layout = QVBoxLayout()
        self.slice_label = QLabel("切片: 0/0")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.valueChanged.connect(self.update_slice)
        slice_layout.addWidget(self.slice_label)
        slice_layout.addWidget(self.slice_slider)
        view_layout.addLayout(slice_layout)
        
        control_layout.addWidget(self.view_group)
        self.view_group.setVisible(False)  # 初始隐藏
        
        # 4. 点和框提示工具 (初始隐藏)
        self.prompt_group = QGroupBox("交互提示工具")
        prompt_layout = QVBoxLayout(self.prompt_group)
        
        # 点提示类型
        point_type_layout = QVBoxLayout()
        point_type_label = QLabel("点提示类型:")
        self.point_type_group = QButtonGroup(self)
        self.fg_radio = QRadioButton("前景点 (左键)")
        self.fg_radio.setChecked(True)
        self.bg_radio = QRadioButton("背景点 (右键)")
        self.point_type_group.addButton(self.fg_radio, 1)
        self.point_type_group.addButton(self.bg_radio, 0)
        self.point_type_group.buttonClicked.connect(self.on_point_type_changed)
        
        point_type_layout.addWidget(point_type_label)
        point_type_layout.addWidget(self.fg_radio)
        point_type_layout.addWidget(self.bg_radio)
        prompt_layout.addLayout(point_type_layout)
        
        # 框模式开关
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
        prompt_layout.addWidget(self.box_mode_btn)
        
        # 清除按钮
        clear_btns_layout = QVBoxLayout()
        self.clear_points_btn = QPushButton("清除所有点")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)
        
        self.clear_box_btn = QPushButton("清除所有框")
        self.clear_box_btn.clicked.connect(self.clear_box)
        self.clear_box_btn.setEnabled(False)
        
        self.clear_last_box_btn = QPushButton("清除最后一个框")
        self.clear_last_box_btn.clicked.connect(self.clear_last_box)
        self.clear_last_box_btn.setEnabled(False)
        
        clear_btns_layout.addWidget(self.clear_points_btn)
        clear_btns_layout.addWidget(self.clear_box_btn)
        clear_btns_layout.addWidget(self.clear_last_box_btn)
        prompt_layout.addLayout(clear_btns_layout)
        
        control_layout.addWidget(self.prompt_group)
        self.prompt_group.setVisible(False)  # 初始隐藏
        
        # 5. 分割按钮
        segment_group = QGroupBox("分割操作")
        segment_layout = QVBoxLayout(segment_group)
        
        self.segment_btn = QPushButton("执行分割")
        self.segment_btn.clicked.connect(self.segment_image)
        self.segment_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
                font-weight: bold;
                height: 40px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
            QPushButton:pressed {
                background-color: #1E8449;
            }
        """)
        segment_layout.addWidget(self.segment_btn)
        
        control_layout.addWidget(segment_group)
        
        # 添加弹性占位符
        control_layout.addStretch(1)
        
        # 添加控制面板到主布局
        main_layout.addWidget(control_panel)
        
        # 创建右侧图像显示区域（改为横向排列）
        image_display = QWidget()
        image_layout = QHBoxLayout(image_display)  # 改为横向布局
        
        # 原始图像显示
        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout(original_group)
        
        original_fig = Figure(figsize=(5, 5), dpi=100)
        self.original_view = InteractiveCanvas(original_fig)
        self.original_view.pointAdded.connect(self.on_point_added)
        self.original_view.boxDrawn.connect(self.on_box_drawn)
        
        original_layout.addWidget(self.original_view)
        
        # 3D切片滑块（仅在3D模式显示）
        slice_control = QWidget()
        slice_layout = QHBoxLayout(slice_control)
        
        self.slice_label = QLabel("切片: 0/0")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.valueChanged.connect(self.update_slice)
        
        slice_layout.addWidget(self.slice_label)
        slice_layout.addWidget(self.slice_slider)
        
        original_layout.addWidget(slice_control)
        slice_control.setVisible(False)  # 初始隐藏
        self.slice_control_widget = slice_control
        
        # 设置原始图像组的布局
        original_group.setLayout(original_layout)
        image_layout.addWidget(original_group)
        
        # 分割结果显示
        result_group = QGroupBox("分割结果")
        result_layout = QVBoxLayout(result_group)
        
        result_fig = Figure(figsize=(5, 5), dpi=100)
        self.result_canvas = FigureCanvas(result_fig)
        self.result_ax = result_fig.add_subplot(111)
        self.result_ax.axis('off')
        
        result_layout.addWidget(self.result_canvas)
        
        # 设置分割结果组的布局
        result_group.setLayout(result_layout)
        image_layout.addWidget(result_group)
        
        # 添加图像显示区到主布局
        main_layout.addWidget(image_display, 1)  # 1是拉伸因子，使其占用更多空间
        
        # 初始检查模型选择
        self.on_model_changed(self.model_selector.currentIndex())
        
    def clear_points(self):
        """清除所有点提示"""
        self.points = []
        self.point_labels = []
        self.original_view.clear_points()
        self.clear_points_btn.setEnabled(False)
        
        # 更新显示
        self.update_display()
        
    def on_model_changed(self, index):
        """当模型选择改变时调用"""
        selected_text = self.model_selector.currentText()
        model_name = selected_text.split(':')[0]
        
        # 如果选择的是MedSAM模型，显示点提示控件
        self.prompt_group.setVisible(model_name == 'medsam')
        
        # 清除当前的点提示和框
        self.clear_points()
        self.clear_box()
        
    def on_point_type_changed(self, button):
        """当点提示类型改变时调用"""
        is_foreground = button.text().startswith("前景")
        self.original_view.set_foreground_point(is_foreground)
            
    def on_box_mode_clicked(self):
        """当框选模式按钮点击时调用"""
        is_checked = self.box_mode_btn.isChecked()
        self.original_view.set_box_mode(is_checked)
        
        # 设置下一个框的颜色索引
        next_color_idx = len(self.boxes) % len(self.original_view.box_colors)
        self.original_view.set_current_color_index(next_color_idx)
        
        # 更新UI状态
        if is_checked:
            self.box_mode_btn.setText("点击模式 👆")
            self.fg_radio.setEnabled(False)
            self.bg_radio.setEnabled(False)
        else:
            self.box_mode_btn.setText("框选模式 📦")
            self.fg_radio.setEnabled(True)
            self.bg_radio.setEnabled(True)
            
    def on_view_type_changed(self, index):
        """当视图类型改变时调用"""
        if not hasattr(self, 'processor') or self.processor.image_data is None or not self.processor.is_3d:
            return
            
        if index == 0:
            self.current_view = 'axial'
            self.current_slice = self.axial_slice
            # 更新滑块范围
            depth = self.processor.image_data.shape[0]
            self.slice_slider.setMaximum(depth - 1)
            self.slice_slider.setValue(self.axial_slice)
            self.slice_label.setText(f"切片: {self.axial_slice}/{depth - 1}")
        elif index == 1:
            self.current_view = 'coronal'
            self.current_slice = self.coronal_slice
            # 更新滑块范围
            height = self.processor.image_data.shape[1]
            self.slice_slider.setMaximum(height - 1)
            self.slice_slider.setValue(self.coronal_slice)
            self.slice_label.setText(f"切片: {self.coronal_slice}/{height - 1}")
        elif index == 2:
            self.current_view = 'sagittal'
            self.current_slice = self.sagittal_slice
            # 更新滑块范围
            width = self.processor.image_data.shape[2]
            self.slice_slider.setMaximum(width - 1)
            self.slice_slider.setValue(self.sagittal_slice)
            self.slice_label.setText(f"切片: {self.sagittal_slice}/{width - 1}")
            
        # 清除当前点和框
        self.clear_points()
        self.clear_box()
        
        # 更新显示
        self.update_display()
    
    def update_slice(self, value):
        """更新当前切片"""
        if self.processor.is_3d and self.processor.image_data is not None:
            # 保存当前视图对应的切片索引
            if self.current_view == 'axial':
                self.axial_slice = value
            elif self.current_view == 'coronal':
                self.coronal_slice = value
            elif self.current_view == 'sagittal':
                self.sagittal_slice = value
                
            # 更新切片标签
            total_slices = self.slice_slider.maximum()
            self.slice_label.setText(f"切片: {value}/{total_slices}")
            
            # 清除当前显示的点和框
            self.original_view.clear_points()
            self.points = []
            self.point_labels = []
            self.original_view.clear_box()
            self.boxes = []
            
            # 更新清除按钮状态
            self.clear_points_btn.setEnabled(False)
            self.clear_box_btn.setEnabled(False)
            self.clear_last_box_btn.setEnabled(False)
            
            # 刷新显示
            self.update_display()
            
    def update_display(self):
        """更新显示内容"""
        if not hasattr(self, 'processor') or self.processor.image_data is None:
            return
            
        # 根据当前视图类型获取对应的切片
        if self.processor.is_3d:
            if self.current_view == 'axial':
                img = self.processor.image_data[self.axial_slice]
            elif self.current_view == 'coronal':
                # 取冠状面(前额面)切片，需要重新组织数据
                img = self.processor.image_data[:, self.coronal_slice, :]
            elif self.current_view == 'sagittal':
                # 取矢状面切片，需要重新组织数据
                img = self.processor.image_data[:, :, self.sagittal_slice]
            else:
                return
        else:
            img = self.processor.image_data
        
        # 显示原始图像
        self.original_view.display_image(img)
        
        # 显示框和点提示
        for box in self.boxes:
            self.original_view.draw_saved_box(box)
        
        # 如果有分割结果，显示结果
        if self.mask is not None:
            self.display_result(img)
    
    def display_result(self, img):
        """显示分割结果"""
        if self.mask is None:
            return
            
        # 获取当前视图的掩码
        if self.processor.is_3d:
            if self.current_view == 'axial':
                mask_slice = self.mask[self.axial_slice]
            elif self.current_view == 'coronal':
                mask_slice = self.mask[:, self.coronal_slice, :]
            elif self.current_view == 'sagittal':
                mask_slice = self.mask[:, :, self.sagittal_slice]
            else:
                return
        else:
            mask_slice = self.mask
        
        # 清除之前的绘图
        self.result_ax.clear()
        
        # 显示原始图像
        if len(img.shape) == 3:  # 彩色图像
            self.result_ax.imshow(img)
        else:  # 灰度图像
            self.result_ax.imshow(img, cmap='gray')
        
        # 使用不同颜色显示每个框的分割结果
        if self.box_masks and self.box_colors:
            for i, (mask, color) in enumerate(zip(self.box_masks, self.box_colors)):
                # 将matplotlib颜色转换为RGB值
                if isinstance(color, str):
                    # 如果颜色是字符串(如'red')，转换为RGB
                    rgb = mcolors.to_rgb(color)
                    r, g, b = [int(c * 255) for c in rgb]
                else:
                    # 如果已经是RGB元组
                    r, g, b = [int(c * 255) for c in color]
                
                # 创建彩色掩码
                colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)
                colored_mask[mask > 0] = [r/255, g/255, b/255, 0.6]  # 更高的透明度
                
                # 显示掩码
                self.result_ax.imshow(colored_mask)
        else:
            # 如果没有框特定的掩码，使用默认红色显示整体掩码
            colored_mask = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
            colored_mask[mask_slice > 0] = [1, 0, 0, 0.6]  # 红色，透明度0.6
            self.result_ax.imshow(colored_mask)
        
        self.result_ax.axis('off')
        self.result_canvas.draw()
    
    def open_image(self):
        """打开医学图像文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开医学图像", "", 
            "医学图像 (*.nii *.nii.gz *.dcm *.png *.jpg *.tif);;所有文件 (*)", 
            options=options
        )
        
        if not file_path:
            return
            
        try:
            # 加载图像
            self.processor.load_image(file_path)
            
            # 清除之前的分割结果
            self.mask = None
            self.points = []
            self.boxes = []
            
            # 重置视图变量
            self.axial_slice = 0
            self.coronal_slice = 0
            self.sagittal_slice = 0
            
            # 更新显示
            if self.processor.is_3d:
                # 如果是3D图像，设置切片控制器
                depth, height, width = self.processor.image_data.shape
                
                # 显示3D切面选择面板
                self.view_group.setVisible(True)
                
                # 设置默认为轴状视图，更新滑块
                self.view_type_combo.setCurrentIndex(0)
                self.current_view = 'axial'
                self.slice_slider.setMaximum(depth - 1)
                self.slice_slider.setValue(0)
                self.slice_label.setText(f"切片: 0/{depth - 1}")
                
                # 显示切片控制器
                self.slice_control_widget.setVisible(True)
                
                # 显示初始切片
                self.update_display()
            else:
                # 如果是2D图像，隐藏切片控制器
                self.view_group.setVisible(False)
                self.slice_control_widget.setVisible(False)
                
                # 显示图像
                self.original_view.display_image(self.processor.image_data)
                
            # 启用保存按钮
            self.save_btn.setEnabled(True)
            
            # 更新界面状态
            selected_model = self.model_selector.currentText().split(':')[0]
            self.prompt_group.setVisible(selected_model == 'medsam')
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")
            traceback.print_exc()
        
    def on_point_added(self, x, y, label):
        """当添加一个点时调用"""
        # 存储点和标签
        self.points.append((x, y))
        self.point_labels.append(label)
        
        # 更新UI状态
        self.clear_points_btn.setEnabled(True)
        
    def on_box_drawn(self, box):
        """处理框选事件"""
        self.boxes.append(box)
        
        # 使用当前颜色索引绘制框
        current_color_idx = self.original_view.current_color_idx
        color = self.original_view.draw_saved_box(box, current_color_idx)
        self.box_colors.append(color)
        
        # 准备下一个框的颜色索引
        next_color_idx = (current_color_idx + 1) % len(self.original_view.box_colors)
        self.original_view.set_current_color_index(next_color_idx)
        
        # 启用清除按钮
        self.clear_box_btn.setEnabled(True)
        self.clear_last_box_btn.setEnabled(True)
    
    def clear_box(self):
        """清除所有框"""
        self.boxes = []
        self.box_colors = []
        self.box_masks = []
        self.original_view.clear_box()
        
        # 更新按钮状态
        self.clear_box_btn.setEnabled(False)
        self.clear_last_box_btn.setEnabled(False)
        
        # 更新显示
        self.update_display()
    
    def clear_last_box(self):
        """清除最后一个框"""
        if not self.boxes:
            return
            
        # 移除最后一个框
        self.boxes.pop()
        if self.box_colors:
            self.box_colors.pop()
        if self.box_masks:
            self.box_masks.pop()
        
        # 清除所有框，然后重新绘制剩余的框
        self.original_view.clear_box()
        for i, box in enumerate(self.boxes):
            self.original_view.draw_saved_box(box, i)
            
        # 更新按钮状态
        self.clear_box_btn.setEnabled(bool(self.boxes))
        self.clear_last_box_btn.setEnabled(bool(self.boxes))
        
        # 更新显示
        self.update_display()
    
    def segment_image(self):
        """使用MedSAM模型分割图像"""
        # 获取当前选择的模型
        selected_model = self.model_selector.currentText().split(':')[0]
        
        if selected_model != 'medsam':
            QMessageBox.warning(self, "提示", "请选择MedSAM模型进行点提示分割")
            return
            
        # 检查是否有图像
        if not hasattr(self, 'processor') or self.processor.image_data is None:
            QMessageBox.warning(self, "提示", "请先加载图像")
            return
            
        # 设置模型
        self.processor.set_segmentation_model(
            model_name='medsam',
            checkpoint_path=self.available_models['medsam']['weights_path']
        )
        
        # 准备点提示和框提示
        points_array = np.array(self.points) if self.points else None
        labels_array = np.array(self.point_labels) if self.point_labels else None
        boxes_array = np.array(self.boxes) if self.boxes else None
        
        # 创建掩码存储
        if self.processor.is_3d:
            if self.mask is None:
                self.mask = np.zeros_like(self.processor.image_data, dtype=bool)
            
            # 根据当前视图进行分割
            if self.current_view == 'axial':
                slice_img = self.processor.image_data[self.axial_slice]
                
                # 清空当前切片的框掩码
                self.box_masks = []
                
                # 为每个框单独生成掩码
                if boxes_array is not None and len(boxes_array) > 0:
                    combined_mask = np.zeros_like(slice_img, dtype=bool)
                    for i, box in enumerate(boxes_array):
                        slice_mask = self.processor.segmenter.segment(
                            slice_img, 
                            points=points_array, 
                            point_labels=labels_array,
                            box=box
                        )
                        self.box_masks.append(slice_mask)
                        combined_mask = np.logical_or(combined_mask, slice_mask)
                    
                    # 将合并掩码放回3D掩码中
                    self.mask[self.axial_slice] = combined_mask
                else:
                    # 只使用点提示
                    slice_mask = self.processor.segmenter.segment(
                        slice_img, 
                        points=points_array, 
                        point_labels=labels_array,
                        box=None
                    )
                    self.mask[self.axial_slice] = slice_mask
                    self.box_masks = [slice_mask]  # 单个掩码
            
            # ... (类似处理其他视图)
                    
        else:
            # 清空框掩码
            self.box_masks = []
            
            # 对2D图像进行分割
            if boxes_array is not None and len(boxes_array) > 0:
                combined_mask = np.zeros_like(self.processor.image_data, dtype=bool)
                for i, box in enumerate(boxes_array):
                    mask = self.processor.segmenter.segment(
                        self.processor.image_data,
                        points=points_array,
                        point_labels=labels_array,
                        box=box
                    )
                    self.box_masks.append(mask)
                    combined_mask = np.logical_or(combined_mask, mask)
                self.mask = combined_mask
            else:
                # 只使用点提示
                self.mask = self.processor.segmenter.segment(
                    self.processor.image_data,
                    points=points_array,
                    point_labels=labels_array,
                    box=None
                )
                self.box_masks = [self.mask]  # 单个掩码
        
        # 显示结果
        self.update_display()
    
    def save_result(self):
        """保存分割结果"""
        if self.mask is None:
            QMessageBox.warning(self, "提示", "没有分割结果可保存")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存分割结果", "", "NIFTI文件 (*.nii.gz);;PNG图像 (*.png);;所有文件 (*)", options=options
        )
        
        if not file_path:
            return
            
        try:
            # 保存掩码
            self.processor.save_mask(self.mask, file_path)
            QMessageBox.information(self, "成功", f"分割结果已保存到 {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存结果失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalImageApp()
    window.show()
    sys.exit(app.exec_()) 