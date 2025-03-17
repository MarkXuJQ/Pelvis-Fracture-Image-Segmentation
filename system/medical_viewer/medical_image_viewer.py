import sys
import os
import numpy as np
import matplotlib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget, 
                             QComboBox, QSlider, QMessageBox, QGroupBox,
                             QRadioButton, QButtonGroup, QToolBar, QCheckBox, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPen, QBrush, QColor, QPainter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import traceback  # 添加traceback模块导入
import logging
from medical_viewer.image_manager import ImageSelectionDialog
from utils.download_thread import DownloadThread
from utils.progress_dialog import UploadProgressDialog
import tempfile

# 减少各种库的日志输出
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# 设置中文字体
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

# 添加项目根目录到 Python 路径 - 使用绝对路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 设置权重目录 - 使用绝对路径
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
os.environ['TORCH_HOME'] = WEIGHTS_DIR
os.environ['PYTORCH_NO_DOWNLOAD'] = '1'

# 智能导入 - 处理直接运行和作为包导入的情况
try:
    # 先尝试相对导入 (当作为包的一部分导入时可以工作)
    from .medical_image_utils import MedicalImageProcessor, list_available_models
    from .vtk_3d_viewer import VTK3DViewer
except ImportError:
    # 相对导入失败时，回退到绝对导入 (直接运行脚本时使用)
    from system.medical_viewer.medical_image_utils import MedicalImageProcessor, list_available_models
    from system.medical_viewer.vtk_3d_viewer import VTK3DViewer

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


# 添加辅助函数
def normalize_box(box, shape):
    """将框坐标归一化到[0,1]范围"""
    h, w = shape
    return np.array([
        box[0] / w,
        box[1] / h,
        box[2] / w,
        box[3] / h
    ], dtype=np.float64)


def apply_ct_window(img, window_center=50, window_width=400):
    """应用CT窗宽窗位"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min)
    return img

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
                radius=self.point_size / 2,
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
            radius=self.point_size / 2,
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

    def set_circles(self, points, labels):
        """设置点提示"""
        # 清除之前的圆点
        for circle in self.points_plotted:
            if circle in self.axes.lines:
                circle.remove()

        self.points_plotted = []
        self.points = []

        # 如果有新的点，则添加它们
        if points and labels and len(points) > 0 and len(labels) > 0:
            for i, (x, y) in enumerate(points):
                if i < len(labels):  # 确保索引有效
                    label = labels[i]
                    self.add_circle(x, y, label)

        self.draw_idle()  # 重绘画布

    def add_circle(self, x, y, label):
        """添加一个圆点到画布上"""
        # 选择颜色
        color = 'red' if label == 1 else 'blue'

        # 绘制圆点
        circle = plt.Circle((x, y), 5, color=color, fill=True, alpha=0.6)
        self.axes.add_patch(circle)
        self.points_plotted.append(circle)
        self.points.append((x, y, label))

    def name_to_rgb(self, color_name):
        """将颜色名称转换为RGB值"""
        color_map = {
            'red': [255, 0, 0],
            'green': [0, 255, 0],
            'blue': [0, 0, 255],
            'yellow': [255, 255, 0],
            'cyan': [0, 255, 255],
            'magenta': [255, 0, 255],
            'white': [255, 255, 255]
        }
        return color_map.get(color_name, [255, 0, 0])  # 默认返回红色


class MedicalImageApp(QMainWindow):
    def __init__(self, patient_id=None):
        super().__init__()
        self.patient_id = patient_id

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

        # **添加 "打开图像" 按钮**
        open_btn = QPushButton("打开图像")
        open_btn.clicked.connect(self.open_image_from_db)  # ✅ 绑定数据库中的图像选择
        file_layout.addWidget(open_btn)

        # **添加 "打开文件夹" 按钮**
        select_file_btn = QPushButton("打开文件夹")
        select_file_btn.clicked.connect(self.open_image)  # ✅ 绑定 open_image 方法
        file_layout.addWidget(select_file_btn)

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
        
        # 添加Gamma调整控件组（初始隐藏）
        self.gamma_group = QGroupBox("X光图像增强")
        gamma_layout = QVBoxLayout(self.gamma_group)
        
        # Gamma值滑块
        gamma_slider_layout = QVBoxLayout()
        self.gamma_value_label = QLabel("Gamma值: 2.0")
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(40)  # 2.0
        self.gamma_slider.setMaximum(200)  # 7.0
        self.gamma_slider.setValue(20)    # 默认2.0
        self.gamma_slider.setTickPosition(QSlider.TicksBelow)
        self.gamma_slider.setTickInterval(10)
        self.gamma_slider.valueChanged.connect(self.on_gamma_changed)
        
        gamma_slider_layout.addWidget(self.gamma_value_label)
        gamma_slider_layout.addWidget(self.gamma_slider)
        gamma_layout.addLayout(gamma_slider_layout)
        
        # 重置按钮
        self.reset_gamma_btn = QPushButton("重置为推荐值")
        self.reset_gamma_btn.clicked.connect(self.reset_gamma)
        gamma_layout.addWidget(self.reset_gamma_btn)
        
        control_layout.addWidget(self.gamma_group)
        self.gamma_group.setVisible(False)  # 初始隐藏
        
        # 3. 3D视图控制 (初始隐藏)
        self.view_control_group = QGroupBox("3D视图控制")
        view_control_layout = QVBoxLayout(self.view_control_group)
        
        # 视图类型选择
        view_type_layout = QHBoxLayout()
        view_type_label = QLabel("视图类型:")
        self.view_type_combo = QComboBox()
        self.view_type_combo.addItems(["轴状视图", "冠状视图", "矢状视图"])
        self.view_type_combo.currentIndexChanged.connect(self.on_view_type_changed)
        view_type_layout.addWidget(view_type_label)
        view_type_layout.addWidget(self.view_type_combo)
        view_control_layout.addLayout(view_type_layout)
        
        # 切片滑块
        slice_layout = QVBoxLayout()
        self.slice_label = QLabel("切片: 0/0")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.valueChanged.connect(self.update_slice)
        slice_layout.addWidget(self.slice_label)
        slice_layout.addWidget(self.slice_slider)
        view_control_layout.addLayout(slice_layout)
        
        control_layout.addWidget(self.view_control_group)
        self.view_control_group.setVisible(False)  # 初始隐藏
        
        # 4. 点和框提示工具
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
        
        # 创建右侧图像显示区域（横向排列）
        image_display = QWidget()
        image_layout = QHBoxLayout(image_display)  # 横向布局
        
        # 原始图像显示
        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout(original_group)
        
        original_fig = Figure(figsize=(5, 5), dpi=100)
        self.original_view = InteractiveCanvas(original_fig)
        self.original_view.pointAdded.connect(self.on_point_added)
        self.original_view.boxDrawn.connect(self.on_box_drawn)
        
        original_layout.addWidget(self.original_view)
        
        # 设置原始图像组的布局
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
        image_layout.addWidget(result_group)
        
        # 添加图像显示区到主布局
        main_layout.addWidget(image_display, 1)  # 1是拉伸因子，使其占用更多空间
        
        # 初始检查模型选择
        self.on_model_changed(self.model_selector.currentIndex())
        
        # 添加3D查看按钮
        self.view_3d_btn = QPushButton("3D查看")
        self.view_3d_btn.setEnabled(False)
        self.view_3d_btn.clicked.connect(self.view_in_3d)
        control_layout.addWidget(self.view_3d_btn)
        
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
        selected_model = selected_text.split(':')[0]
        
        # 根据模型类型显示相应控件
        is_medsam = selected_model == 'medsam'
        self.prompt_group.setVisible(is_medsam)
        
        # 清除当前的点提示和框
        self.clear_points()
        self.clear_box()
        
        # 根据模型类型启用/禁用3D视图按钮
        is_3d_capable = self.is_3d_capable_model(selected_model)
        if hasattr(self, 'view_3d_btn'):
            self.view_3d_btn.setEnabled(is_3d_capable)
        
        # 如果模型不支持3D，并且当前在显示3D视图，则切换回2D视图
        if not is_3d_capable and hasattr(self, 'vtk_viewer') and self.vtk_viewer.isVisible():
            self.vtk_viewer.hide()
            if hasattr(self, 'main_widget'):
                self.main_widget.show()
        
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
        """处理视图类型改变事件"""
        view_types = ['axial', 'coronal', 'sagittal']
        self.current_view = view_types[index]
        
        # 清除分割结果和交互提示，避免维度不匹配问题
        self.mask = None
        self.box_masks = []
        self.clear_points()
        self.clear_box()
        
        # 更新切片滑块最大值
        if self.processor.is_3d:
            if self.current_view == 'axial':
                max_slice = self.processor.image_data.shape[0] - 1
                self.slice_slider.setValue(self.axial_slice)
            elif self.current_view == 'coronal':
                max_slice = self.processor.image_data.shape[1] - 1
                self.slice_slider.setValue(self.coronal_slice)
            elif self.current_view == 'sagittal':
                max_slice = self.processor.image_data.shape[2] - 1
                self.slice_slider.setValue(self.sagittal_slice)
                
            self.slice_slider.setMaximum(max_slice)
            self.slice_label.setText(f"切片: {self.slice_slider.value()}/{max_slice}")
        
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
        """更新显示"""
        if not hasattr(self, 'processor') or self.processor.image_data is None:
            return
            
        # 获取当前切片
        if self.processor.is_3d:
            if self.current_view == 'axial':
                img = self.processor.image_data[self.axial_slice]
            elif self.current_view == 'coronal':
                img = self.processor.image_data[:, self.coronal_slice, :]
                img = np.rot90(img, k=2)  # 旋转180度
            elif self.current_view == 'sagittal':
                img = self.processor.image_data[:, :, self.sagittal_slice]
                img = np.rot90(img, k=2)  # 旋转180度
        else:
            img = self.processor.image_data
        
        # 显示图像
        self.original_view.display_image(img)
        
        # 更新点和框
        self.original_view.set_circles(self.points, self.point_labels)
        for i, box in enumerate(self.boxes):
            color_idx = i % len(self.original_view.box_colors)
            self.original_view.draw_saved_box(box, color_idx)
        
        # 显示分割结果
        if self.mask is not None:
            self.display_result(img)
    
    def display_result(self, current_slice):
        """显示分割结果"""
        if self.mask is None:
            return
            
        # 确定掩码是2D还是3D
        mask_is_3d = len(self.mask.shape) == 3
        
        # 获取当前视图对应的掩码切片
        if mask_is_3d:
            if self.current_view == 'axial':
                mask_slice = self.mask[self.axial_slice]
            elif self.current_view == 'coronal':
                mask_slice = self.mask[:, self.coronal_slice, :]
                mask_slice = np.rot90(mask_slice, k=2)
            elif self.current_view == 'sagittal':
                mask_slice = self.mask[:, :, self.sagittal_slice]
                mask_slice = np.rot90(mask_slice, k=2)
        else:
            # 如果是2D掩码，直接使用
            mask_slice = self.mask
        
        # 清除之前的结果
        self.result_ax.clear()
        
        # 获取当前选择的模型
        selected_text = self.model_selector.currentText()
        selected_model = selected_text.split(':')[0]
        
        # 显示结果
        if selected_model == 'deeplabv3':
            # DeepLabV3 特殊处理 - 使用彩色分割方案
            # 先显示原始图像
            self.result_ax.imshow(current_slice, cmap='gray')
            
            # 创建彩色掩码显示
            cmap = plt.cm.get_cmap('tab20', 24)  # 24类，使用tab20彩色映射
            
            # 生成彩色掩码
            colored_mask = np.zeros((*mask_slice.shape, 4))
            unique_values = np.unique(mask_slice)
            print(f"掩码中的唯一值: {unique_values}")
            
            for i, val in enumerate(unique_values):
                if val > 0:  # 跳过背景
                    # 获取该类别的颜色
                    color = cmap(int(val % 24))
                    # 将该值对应的像素设置为对应颜色
                    mask_color = np.zeros((*mask_slice.shape, 4))
                    mask_color[mask_slice == val] = color
                    mask_color[mask_slice == val, 3] = 0.7  # 设置Alpha通道
                    
                    # 显示该类别的掩码
                    self.result_ax.imshow(mask_color)
                    
                    # 绘制该类别的轮廓
                    self.result_ax.contour(mask_slice == val, colors=[color[:3]], linewidths=1.0)
        
        elif len(self.box_masks) > 1:
            # 多个框，使用彩色掩码
            print(f"使用彩色掩码: {len(self.box_masks)}个")
            
            # 先显示原始图像
            self.result_ax.imshow(current_slice, cmap='gray')
            
            # 然后叠加每个掩码
            for i, mask in enumerate(self.box_masks):
                if i >= len(self.original_view.box_colors):
                    break
                
                # 确定当前掩码的切片
                if mask_is_3d:
                    if self.current_view == 'axial':
                        mask_i_slice = mask[self.axial_slice]
                    elif self.current_view == 'coronal':
                        mask_i_slice = mask[:, self.coronal_slice, :]
                        mask_i_slice = np.rot90(mask_i_slice, k=2)
                    elif self.current_view == 'sagittal':
                        mask_i_slice = mask[:, :, self.sagittal_slice]
                        mask_i_slice = np.rot90(mask_i_slice, k=2)
                else:
                    mask_i_slice = mask
                
                # 创建彩色掩码
                color_name = self.original_view.box_colors[i]
                color_rgb = self.original_view.name_to_rgb(color_name)
                
                # 创建彩色掩码数组
                h, w = mask_i_slice.shape
                colored_mask = np.zeros((h, w, 4))
                colored_mask[mask_i_slice > 0, 0] = color_rgb[0] / 255.0
                colored_mask[mask_i_slice > 0, 1] = color_rgb[1] / 255.0
                colored_mask[mask_i_slice > 0, 2] = color_rgb[2] / 255.0
                colored_mask[mask_i_slice > 0, 3] = 0.5  # 半透明
                
                # 显示彩色掩码
                self.result_ax.imshow(colored_mask)
        else:
            # 单个掩码，使用改进的显示方式
            # 先显示原始图像
            self.result_ax.imshow(current_slice, cmap='gray')
            
            # 创建带透明度的红色掩码
            if np.any(mask_slice):
                h, w = mask_slice.shape
                red_mask = np.zeros((h, w, 4))
                red_mask[mask_slice > 0, 0] = 1.0  # 红色通道
                red_mask[mask_slice > 0, 3] = 0.5  # Alpha通道，半透明
                
                # 显示填充的红色掩码
                self.result_ax.imshow(red_mask)
                
                # 同时也显示轮廓，增强可视性
                self.result_ax.contour(mask_slice > 0, colors=['red'], linewidths=1.5)
        
        self.result_ax.axis('off')
        self.result_canvas.draw()

    def open_image_from_db(self):
        """ 打开病人的医学图像选择窗口 """
        if not self.patient_id:
            QMessageBox.warning(self, "错误", "无法获取病人ID")
            return

        # 创建并显示选择对话框
        dialog = ImageSelectionDialog(self.patient_id, self)
        if dialog.exec_() == QDialog.Accepted:
            print(1)
            selected_image_path = dialog.selected_image_path
            if selected_image_path:
                self.load_selected_image(selected_image_path)

    def load_selected_image(self, image_path):
        """从数据库下载图像"""
        try:
            # **创建临时文件存储下载的图像**
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mha') as temp_file:
                temp_path = temp_file.name

            # **显示下载进度**
            self.progress_dialog = UploadProgressDialog(self)
            self.progress_dialog.setWindowTitle("下载进度")
            self.progress_dialog.status_label.setText("正在下载图像...")
            self.progress_dialog.show()

            # **创建下载线程**
            self.download_thread = DownloadThread(image_path, temp_path)
            self.download_thread.progress.connect(self.progress_dialog.update_progress)
            self.download_thread.finished.connect(
                lambda success, message: self.on_download_finished(temp_path, success, message)
            )
            self.download_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败：{str(e)}")

    def on_download_finished(self, temp_path, success, message):
        """下载完成后，自动检测图像类型并更新 UI"""
        try:
            if hasattr(self, 'progress_dialog'):
                self.progress_dialog.close()
                self.progress_dialog.deleteLater()

            if success:
                # **使用 MedicalImageProcessor 读取图像**
                self.processor.load_image(temp_path)
                try:

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
                        print("✅ 3D 图像加载成功")

                        # 如果是3D图像，设置切片控制器
                        depth, height, width = self.processor.image_data.shape

                        # 显示3D视图控制面板
                        self.view_control_group.setVisible(True)

                        # 隐藏Gamma控制面板
                        self.gamma_group.setVisible(False)

                        # 设置默认为轴状视图，更新滑块
                        self.view_type_combo.setCurrentIndex(0)
                        self.current_view = 'axial'
                        self.slice_slider.setMaximum(depth - 1)
                        self.slice_slider.setValue(0)
                        self.slice_label.setText(f"切片: 0/{depth - 1}")

                        # 显示初始切片
                        self.update_display()
                    else:
                        print("✅ 2D 图像加载成功")

                        # 如果是2D图像，隐藏切片控制器，显示Gamma控制器
                        self.view_control_group.setVisible(False)

                        # 检查是否可能是X光图像
                        is_xray = self._check_if_xray(self.processor.image_data)

                        # 显示Gamma控制面板
                        self.gamma_group.setVisible(True)

                        # 对于X光图像，设置推荐的初始gamma值
                        if is_xray:
                            self.gamma_slider.setValue(30)  # 3.0
                        else:
                            self.gamma_slider.setValue(20)  # 2.0
                        self.on_gamma_changed(self.gamma_slider.value())

                        # 显示图像
                        self.original_view.display_image(self.processor.image_data)

                    # 启用保存按钮
                    self.save_btn.setEnabled(True)

                    # 更新界面状态
                    selected_model = self.model_selector.currentText().split(':')[0]
                    self.prompt_group.setVisible(selected_model == 'medsam')

                    # 更新3D查看按钮状态
                    self.view_3d_btn.setEnabled(self.processor.is_3d and self.mask is not None)

                except Exception as e:
                    QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")

            #     # **自动检测 3D / 2D**
            #     if self.processor.is_3d:
            #         print("✅ 3D 图像加载成功")
            #     else:
            #         print("✅ 2D 图像加载成功")
            #
            #     QMessageBox.information(self, "成功", "图像加载完成！")
            #
            # else:
            #     QMessageBox.critical(self, "错误", f"下载失败：{message}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示图像失败：{str(e)}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def open_image(self):
        """打开医学图像文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开医学图像", "",
            "医学图像 (*.mha *.nii *.nii.gz *.dcm *.png *.jpg *.tif);;所有文件 (*)",
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

                # 显示3D视图控制面板
                self.view_control_group.setVisible(True)

                # 隐藏Gamma控制面板
                self.gamma_group.setVisible(False)

                # 设置默认为轴状视图，更新滑块
                self.view_type_combo.setCurrentIndex(0)
                self.current_view = 'axial'
                self.slice_slider.setMaximum(depth - 1)
                self.slice_slider.setValue(0)
                self.slice_label.setText(f"切片: 0/{depth - 1}")

                # 显示初始切片
                self.update_display()
            else:
                # 如果是2D图像，隐藏切片控制器，显示Gamma控制器
                self.view_control_group.setVisible(False)

                # 检查是否可能是X光图像
                is_xray = self._check_if_xray(self.processor.image_data)

                # 显示Gamma控制面板
                self.gamma_group.setVisible(True)

                # 对于X光图像，设置推荐的初始gamma值
                if is_xray:
                    self.gamma_slider.setValue(30)  # 3.0
                else:
                    self.gamma_slider.setValue(20)  # 2.0
                self.on_gamma_changed(self.gamma_slider.value())

                # 显示图像
                self.original_view.display_image(self.processor.image_data)

            # 启用保存按钮
            self.save_btn.setEnabled(True)

            # 更新界面状态
            selected_model = self.model_selector.currentText().split(':')[0]
            self.prompt_group.setVisible(selected_model == 'medsam')

            # 更新3D查看按钮状态
            self.view_3d_btn.setEnabled(self.processor.is_3d and self.mask is not None)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")
            traceback.print_exc()

    def _check_if_xray(self, image):
        """检查图像是否可能是X光图像"""
        # 简单启发式方法：检查图像特征
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # 灰度图
            mean_val = np.mean(image)
            std_val = np.std(image)
            # X光通常有中等平均亮度和较高对比度
            return 30 < mean_val < 200 and std_val > 40
        elif len(image.shape) == 3:
            # 检查RGB图像是否近似灰度（如果是X光的RGB表示）
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            if np.abs(np.mean(r - g)) < 5 and np.abs(np.mean(r - b)) < 5 and np.abs(np.mean(g - b)) < 5:
                return True
        return False

    def on_gamma_changed(self, value):
        """处理gamma值滑块变化"""
        gamma = value / 10.0  # 滑块值转换为gamma值
        self.gamma_value_label.setText(f"Gamma值: {gamma:.1f}")

        if hasattr(self, 'processor') and self.processor.image_data is not None and not self.processor.is_3d:
            # 保存原始图像
            if not hasattr(self, 'original_image_backup'):
                self.original_image_backup = self.processor.image_data.copy()
            else:
                # 恢复原始图像，然后应用新的gamma
                self.processor.image_data = self.original_image_backup.copy()

            # 应用gamma校正
            self.processor.apply_gamma_correction(gamma)

            # 更新显示
            self.original_view.display_image(self.processor.image_data)

            # 如果有分割结果，也更新分割结果显示
            if self.mask is not None:
                self.display_result(self.processor.image_data)

    def reset_gamma(self):
        """重置为推荐的gamma值"""
        # 检查图像是否可能是X光
        if hasattr(self, 'processor') and self.processor.image_data is not None:
            is_xray = self._check_if_xray(
                self.original_image_backup if hasattr(self, 'original_image_backup') else self.processor.image_data)

            # 设置推荐值
            if is_xray:
                recommended_value = 30  # 3.0
            else:
                recommended_value = 20  # 2.0

            self.gamma_slider.setValue(recommended_value)
            # on_gamma_changed会自动被调用

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
        """使用选定的模型分割图像"""
        # 获取当前选择的模型
        selected_text = self.model_selector.currentText()
        selected_model = selected_text.split(':')[0]

        # 检查是否有图像
        if not hasattr(self, 'processor') or self.processor.image_data is None:
            QMessageBox.warning(self, "提示", "请先加载图像")
            return

        # 设置模型
        try:
            print(f"开始使用 {selected_model} 进行分割...")
            model_info = self.available_models.get(selected_model)
            if not model_info:
                QMessageBox.warning(self, "提示", f"未找到模型: {selected_model}")
                return

            self.processor.set_segmentation_model(
                model_name=selected_model,
                checkpoint_path=model_info['weights_path']
            )

            # 获取当前图像切片
            if self.processor.is_3d:
                if self.current_view == 'axial':
                    image_slice = self.processor.image_data[self.axial_slice]
                elif self.current_view == 'coronal':
                    image_slice = self.processor.image_data[:, self.coronal_slice, :]
                    image_slice = np.rot90(image_slice, k=2)
                elif self.current_view == 'sagittal':
                    image_slice = self.processor.image_data[:, :, self.sagittal_slice]
                    image_slice = np.rot90(image_slice, k=2)
            else:
                image_slice = self.processor.image_data

            # 清空框掩码
            self.box_masks = []

            # ===== MedSAM 分割路径 =====
            if selected_model == 'medsam':
                # 准备MedSAM的点提示和框提示
                points_array = np.array(self.points) if self.points else None
                labels_array = np.array(self.point_labels) if self.point_labels else None
                boxes_array = np.array(self.boxes) if self.boxes else None

                if boxes_array is not None and len(boxes_array) > 0:
                    # 如果有多个框，处理每个框
                    combined_mask = np.zeros_like(image_slice, dtype=bool)

                    for i, box in enumerate(boxes_array):
                        print(f"处理MedSAM框 {i+1}/{len(boxes_array)}: {box}")

                        # MedSAM使用原始接口
                        mask = self.processor.segmenter.segment(
                            image_slice,
                            points=points_array,
                            point_labels=labels_array,
                            box=box
                        )

                        # 保存每个框的掩码
                        self.box_masks.append(mask)

                        # 更新组合掩码
                        combined_mask = np.logical_or(combined_mask, mask > 0)

                    # 将布尔掩码转换为uint8
                    self.mask = (combined_mask * 255).astype(np.uint8)
                else:
                    # 只使用点提示或不使用任何提示
                    print(f"使用MedSAM模型进行分割，无框提示")

                    mask = self.processor.segmenter.segment(
                        image_slice,
                        points=points_array,
                        point_labels=labels_array
                    )

                    # 保存掩码
                    self.mask = mask
                    self.box_masks = [mask]

            # ===== DeepLabV3 分割路径 =====
            elif selected_model == 'deeplabv3':
                print("使用DeepLabV3进行分割")

                # DeepLabV3 不使用点标记或框，直接进行分割
                # 设置raw_output=True获取原始多类别预测，便于后处理
                use_raw_output = True

                # 执行分割
                multi_class_mask = self.processor.segmenter.segment(
                    image_slice,
                    raw_output=use_raw_output
                )

                if multi_class_mask is not None:
                    print(f"DeepLabV3分割完成，类别范围: {np.min(multi_class_mask)} - {np.max(multi_class_mask)}")
                    # 创建二值掩码 (非背景为前景)
                    binary_mask = (multi_class_mask > 0).astype(np.uint8) * 255

                    # 保存掩码
                    self.mask = binary_mask
                    self.box_masks = [binary_mask]
                else:
                    print("DeepLabV3分割失败，返回了空掩码")
                    QMessageBox.warning(self, "警告", "分割失败，返回了空掩码")

            # 更新显示
            self.update_display()

            # 启用3D查看按钮
            self.view_3d_btn.setEnabled(self.processor.is_3d and self.mask is not None)

            print("分割完成！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"分割时出错: {str(e)}")
            traceback.print_exc()

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

    def view_in_3d(self):
        """在3D中查看图像"""
        if not hasattr(self, 'processor') or not self.processor.image_data is not None:
            QMessageBox.warning(self, "错误", "请先加载图像")
            return

        if not self.processor.is_3d:
            QMessageBox.warning(self, "错误", "只有3D图像才能以3D方式查看")
            return

        # 获取当前选择的模型
        current_model = self.model_selector.currentText() if hasattr(self, 'model_selector') else None

        # 检查是否是支持3D的模型
        if current_model and not self.is_3d_capable_model(current_model):
            QMessageBox.information(self, "提示",
                                  f"当前选择的 {current_model} 模型不支持3D显示。\n"
                                  "请使用 3D U-Net 或其他3D分割模型。")
            return

        # 创建3D查看器
        if not hasattr(self, 'vtk_viewer'):
            self.vtk_viewer = VTK3DViewer()

        # 获取当前图像数据
        volume = self.processor.image_data

        # 如果已经有掩码，也加载它
        mask = self.current_mask if hasattr(self, 'current_mask') and self.current_mask is not None else None

        # 设置数据
        self.vtk_viewer.set_volume_data(volume, mask)

        # 显示3D查看器
        self.vtk_viewer.setWindowTitle("3D医学图像查看器")
        self.vtk_viewer.resize(800, 600)
        self.vtk_viewer.show()

    def is_3d_capable_model(self, model_name):
        """判断所选模型是否支持3D显示"""
        if model_name == 'deeplabv3':
            return False  # DeepLabV3 只支持 2D 切片分割
        elif model_name in ['3dunet', 'vnet', 'medsam']:
            return True  # 这些模型支持 3D 分割
        return False  # 默认不支持


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalImageApp()
    window.show()
    sys.exit(app.exec_())