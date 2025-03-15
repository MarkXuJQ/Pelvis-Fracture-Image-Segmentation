"""
图像查看器窗口
提供独立的图像查看功能
"""

import sys
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QHeaderView, QTableWidgetItem
from PyQt5 import uic
import SimpleITK as sitk
from sqlalchemy.dialects.mysql import pymysql
from system.medical_viewer.ct_viewer import CTViewer
from system.database.db_manager import get_connection
from system.medical_viewer.xray_viewer import XRayViewer
from utils.file_upload import FileUploader
import tempfile
from utils.download_thread import DownloadThread
from utils.progress_dialog import UploadProgressDialog
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtWidgets import QPushButton, QGroupBox, QComboBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from system.medical_viewer.medical_image_utils import MedicalImageProcessor


class ImageViewerWindow(QMainWindow):
    """独立的图像查看器窗口"""
    
    def __init__(self, parent=None):
        """初始化图像查看器窗口"""
        super().__init__(parent)
        self.setWindowTitle("医学图像查看器")
        self.setGeometry(100, 100, 1000, 800)
        
        # 初始化图像处理器
        self.processor = MedicalImageProcessor()
        
        # 初始化UI
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部工具栏
        toolbar = QHBoxLayout()
        open_btn = QPushButton("打开图像")
        open_btn.clicked.connect(self.open_image)
        toolbar.addWidget(open_btn)
        
        save_btn = QPushButton("保存图像")
        save_btn.clicked.connect(self.save_image)
        toolbar.addWidget(save_btn)
        
        toolbar.addStretch()
        main_layout.addLayout(toolbar)
        
        # 图像显示区域
        image_layout = QHBoxLayout()
        
        # 原始图像
        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout(original_group)
        
        # 创建matplotlib图形
        self.original_figure = Figure(figsize=(5, 4), dpi=100)
        self.original_canvas = FigureCanvas(self.original_figure)
        original_layout.addWidget(self.original_canvas)
        
        self.original_axes = self.original_figure.add_subplot(111)
        self.original_axes.set_xticks([])
        self.original_axes.set_yticks([])
        
        image_layout.addWidget(original_group)
        
        # 处理后图像
        processed_group = QGroupBox("处理后图像")
        processed_layout = QVBoxLayout(processed_group)
        
        self.processed_figure = Figure(figsize=(5, 4), dpi=100)
        self.processed_canvas = FigureCanvas(self.processed_figure)
        processed_layout.addWidget(self.processed_canvas)
        
        self.processed_axes = self.processed_figure.add_subplot(111)
        self.processed_axes.set_xticks([])
        self.processed_axes.set_yticks([])
        
        image_layout.addWidget(processed_group)
        
        main_layout.addLayout(image_layout)
        
        # 控制面板
        controls_layout = QHBoxLayout()
        
        # 基本控制
        basic_controls = QGroupBox("基本控制")
        basic_layout = QVBoxLayout(basic_controls)
        
        # 视图类型选择（仅对3D图像有效）
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("视图类型:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["轴向", "冠状", "矢状"])
        self.view_combo.setEnabled(False)  # 默认禁用
        self.view_combo.currentIndexChanged.connect(self.change_view)
        view_layout.addWidget(self.view_combo)
        basic_layout.addLayout(view_layout)
        
        controls_layout.addWidget(basic_controls)
        
        main_layout.addLayout(controls_layout)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def open_image(self):
        """打开图像文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开医学图像", "", 
            "医学图像文件 (*.nii *.nii.gz *.dcm *.mha *.mhd *.nrrd *.tif *.tiff *.png *.jpg);;所有文件 (*)",
            options=options
        )
        
        if file_path:
            try:
                self.processor.load_image(file_path)
                self.display_image()
                
                # 启用3D视图控制（如果适用）
                self.view_combo.setEnabled(self.processor.is_3d)
                
                self.statusBar().showMessage(f"已加载图像: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")
                
    def save_image(self):
        """保存当前图像"""
        if self.processor.image_data is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", 
            "PNG文件 (*.png);;JPEG文件 (*.jpg);;TIFF文件 (*.tif);;所有文件 (*)",
            options=options
        )
        
        if file_path:
            try:
                # 获取当前显示的切片
                if self.processor.is_3d:
                    # 根据当前视图获取正确的切片
                    view_idx = self.view_combo.currentIndex()
                    view_type = ["axial", "coronal", "sagittal"][view_idx]
                    
                    if view_type == "axial":
                        slice_idx = 0  # 这里应该使用实际的切片索引
                        image = self.processor.image_data[slice_idx, :, :]
                    elif view_type == "coronal":
                        slice_idx = 0  # 这里应该使用实际的切片索引
                        image = self.processor.image_data[:, slice_idx, :]
                    else:  # sagittal
                        slice_idx = 0  # 这里应该使用实际的切片索引
                        image = self.processor.image_data[:, :, slice_idx]
                else:
                    image = self.processor.image_data
                
                # 归一化到0-255范围
                image = (image - image.min()) / (image.max() - image.min()) * 255
                image = image.astype(np.uint8)
                
                # 使用matplotlib保存图像
                plt.imsave(file_path, image, cmap='gray')
                self.statusBar().showMessage(f"图像已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图像时出错: {str(e)}")
    
    def display_image(self):
        """显示当前加载的图像"""
        if self.processor.image_data is None:
            return
            
        # 清除当前显示
        self.original_axes.clear()
        self.processed_axes.clear()
        
        # 设置轴标签
        self.original_axes.set_xticks([])
        self.original_axes.set_yticks([])
        self.processed_axes.set_xticks([])
        self.processed_axes.set_yticks([])
        
        if self.processor.is_3d:
            # 3D图像，显示当前选择的视图和切片
            view_idx = self.view_combo.currentIndex()
            view_type = ["axial", "coronal", "sagittal"][view_idx]
            
            if view_type == "axial":
                slice_idx = 0  # 这里应该使用实际的切片索引
                image = self.processor.image_data[slice_idx, :, :]
            elif view_type == "coronal":
                slice_idx = 0  # 这里应该使用实际的切片索引
                image = self.processor.image_data[:, slice_idx, :]
            else:  # sagittal
                slice_idx = 0  # 这里应该使用实际的切片索引
                image = self.processor.image_data[:, :, slice_idx]
                
            self.original_axes.imshow(image, cmap='gray')
            self.processed_axes.imshow(image, cmap='gray')
            
            # 设置标题
            self.original_axes.set_title(f"{view_type.capitalize()} 视图 - 切片 {slice_idx}")
            self.processed_axes.set_title("处理后图像")
        else:
            # 2D图像，直接显示
            self.original_axes.imshow(self.processor.image_data, cmap='gray')
            self.processed_axes.imshow(self.processor.image_data, cmap='gray')
            
            self.original_axes.set_title("原始图像")
            self.processed_axes.set_title("处理后图像")
            
        # 更新画布
        self.original_canvas.draw()
        self.processed_canvas.draw()
    
    def change_view(self, index):
        """改变3D图像的视图"""
        if self.processor.is_3d:
            self.display_image()


class MedicalImageViewer(QMainWindow):
    def __init__(self, patient_id=None):
        super().__init__()
        self.patient_id = patient_id
        
        # 尝试加载UI文件
        try:
            # 尝试多个可能的UI文件位置
            ui_paths = [
                # 1. 相对于当前脚本目录
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "image_viewer_window.ui"),
                
                # 2. 用户指定的位置
                os.path.abspath(os.path.join("system", "ui", "image_viewer_window.ui")),
                
                # 3. 基于当前工作目录
                os.path.join(os.getcwd(), "system", "ui", "image_viewer_window.ui"),
                
                # 4. 绝对路径 (如果在Windows下运行)
                "d:\\pelvis\\system\\ui\\image_viewer_window.ui" if os.name == 'nt' else None
            ]
            
            # 移除None值
            ui_paths = [p for p in ui_paths if p]
            
            # 打印所有尝试的路径，帮助调试
            print("尝试加载MedicalImageViewer UI文件，路径列表:")
            for path in ui_paths:
                print(f" - {path} (存在: {os.path.exists(path)})")
            
            # 尝试加载第一个存在的UI文件
            ui_loaded = False
            for ui_path in ui_paths:
                if os.path.exists(ui_path):
                    print(f"发现UI文件: {ui_path}")
                    uic.loadUi(ui_path, self)
                    print(f"成功加载UI文件: {ui_path}")
                    ui_loaded = True
                    break
            
            if not ui_loaded:
                print("警告: 未找到UI文件，程序将继续运行但可能界面不完整")
                # 手动创建基本UI
                self.create_basic_ui()
                
        except Exception as e:
            print(f"加载UI文件时出错: {str(e)}")
            # 创建基本UI
            self.create_basic_ui()
        
        # 设置窗口属性
        self.setWindowTitle("医学图像查看器")
        self.setup_connections()
        
        # 其余初始化代码...

    def create_basic_ui(self):
        """创建基本UI（当UI文件加载失败时）"""
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        layout = QVBoxLayout(self.centralwidget)
        
        # 添加基本控件
        self.label = QLabel("医学图像查看器", self.centralwidget)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        # 添加基本按钮
        self.load_ct_button = QPushButton("加载CT图像", self.centralwidget)
        self.load_ct_button.setObjectName("load_ct_button")
        layout.addWidget(self.load_ct_button)
        
        self.back_button = QPushButton("返回", self.centralwidget)
        self.back_button.setObjectName("back_button")
        layout.addWidget(self.back_button)
        
        # 设置尺寸
        self.resize(800, 600)

    def setup_connections(self):
        """设置信号和槽连接"""
        # 查找并连接按钮
        load_ct_btn = self.findChild(QPushButton, 'load_ct_button')
        if load_ct_btn:
            load_ct_btn.clicked.connect(self.load_ct_image)
            
        back_btn = self.findChild(QPushButton, 'back_button')
        if back_btn:
            back_btn.clicked.connect(self.go_back)
        
        # 其他连接...

    def go_back(self):
        """点击返回按钮时，关闭窗口"""
        self.close()

    def perform_segmentation(self):
        print("执行分割")

    def visualize_results(self):
        """点击 '可视化' 按钮时，加载选中的图像"""
        if not self.selected_image_path:
            QMessageBox.warning(self, "未选中图像", "请先在表格中选择一张图像。")
            return

        try:
            self.load_image(self.selected_image_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")

    def load_patient_images(self):
        """加载病人的所有图像记录"""
        try:
            connection = get_connection()
            cursor = connection.cursor()
            
            # 查询病人的所有图像
            cursor.execute("""
                SELECT image_name, modality, image_path, upload_date 
                FROM patient_images 
                WHERE patient_id = %s
                ORDER BY upload_date DESC
            """, (self.patient_id,))
            
            images = cursor.fetchall()
            
            # 清空表格
            self.imageTable.setRowCount(0)
            
            # 填充表格
            for row, (image_name, modality, image_path, upload_date) in enumerate(images):
                self.imageTable.insertRow(row)
                self.imageTable.setItem(row, 0, QTableWidgetItem(image_name))
                self.imageTable.setItem(row, 1, QTableWidgetItem(modality))
                self.imageTable.setItem(row, 2, QTableWidgetItem(str(upload_date)))
                # 存储图像路径（隐藏）
                path_item = QTableWidgetItem(image_path)
                self.imageTable.setItem(row, 3, path_item)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像记录失败：{str(e)}")
        finally:
            cursor.close()
            connection.close()

    def load_medical_image(self, image_path, modality):
        """加载医学图像"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mha') as temp_file:
                temp_path = temp_file.name

            # 创建进度对话框
            self.progress_dialog = UploadProgressDialog(self)
            self.progress_dialog.setWindowTitle("下载进度")
            self.progress_dialog.status_label.setText("正在下载图像...")
            self.progress_dialog.show()

            # 创建下载线程
            self.download_thread = DownloadThread(image_path, temp_path)
            self.download_thread.progress.connect(self.progress_dialog.update_progress)
            self.download_thread.finished.connect(
                lambda image, success, message: self.on_download_finished(
                    image, success, message, temp_path, modality
                )
            )
            self.download_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败：{str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def on_download_finished(self, image, success, message, temp_path, modality):
        """下载完成的回调"""
        try:
            # 关闭进度对话框
            if hasattr(self, 'progress_dialog'):
                self.progress_dialog.close()
                self.progress_dialog.deleteLater()

            if success and image is not None:
                # 根据模态类型显示图像
                if modality.upper() == 'CT':
                    self.show_ct_viewer(image)
                elif modality.upper() == 'XRAY':
                    self.show_xray_viewer(image)
            else:
                QMessageBox.critical(self, "错误", f"加载图像失败：{message}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示图像失败：{str(e)}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def image_selected(self, row, column):
        """当用户选择一个图像时"""
        try:
            # 获取选中行的图像信息
            image_path = self.imageTable.item(row, 3).text()  # 获取存储的路径
            modality = self.imageTable.item(row, 1).text()    # 获取模态类型
            
            # 加载并显示图像
            self.load_medical_image(image_path, modality)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开图像失败：{str(e)}")

    def show_ct_viewer(self, image):
        """显示CT图像查看器"""
        try:
            ct_viewer = CTViewer(image, self, patient_id=self.patient_id)
            self.setCentralWidget(ct_viewer)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示CT图像失败：{str(e)}")

    def show_xray_viewer(self, image):
        """显示X光图像查看器"""
        try:
            xray_viewer = XRayViewer(image, self, patient_id=self.patient_id)
            self.setCentralWidget(xray_viewer)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示X光图像失败：{str(e)}")

    def get_absolute_path(self, image_path):
        """将数据库存储的相对路径转换为当前项目的绝对路径"""
        base_dir = os.path.abspath(os.path.dirname(__file__))  # 获取当前脚本所在目录

        # 确保路径是相对的，不是 /data/medical_images/
        if image_path.startswith("\\data\\"):
            image_path = image_path.lstrip("\\")  # 去掉开头的 "/"

        return os.path.join(base_dir, image_path)  # 生成正确的路径


    def load_image(self, file_path):
        """加载并显示医学图像"""
        try:
            # 先转换路径（防止路径不正确）
            abs_path = self.get_absolute_path(file_path)

            print(abs_path)

            # 确保文件存在
            if not os.path.exists(abs_path):
                QMessageBox.critical(self, "错误", f"文件未找到: {abs_path}")
                return

            # 加载图像
            self.image = sitk.ReadImage(abs_path)
            dimension = self.image.GetDimension()

            if dimension == 2:
                image_array = sitk.GetArrayFromImage(self.image)
                self.viewer = XRayViewer(image_array)
            elif dimension == 3:
                self.viewer = CTViewer(self.image)
                self.viewer = CTViewer(self.image, parent=self)  # 传递 self 作为 parent

            else:
                QMessageBox.warning(self, "不支持的图像", "选中的图像格式不支持。")
                return
            self.viewer.show()
            self.hide()  # 隐藏 MedicalImageViewer 窗口
            # self.setCentralWidget(self.viewer)
            self.statusBar().showMessage(f'Loaded image: {abs_path}')
            print(f"成功加载图像: {abs_path}")  # 调试信息

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败:\n{str(e)}")
    def save_image(self):
        if not hasattr(self, 'image'):
            QMessageBox.warning(self, "No Image", "No image is loaded to save.")
            return

        # Define supported formats
        formats = [
            ("NIfTI (*.nii)", "*.nii"),
            ("NIfTI Compressed (*.nii.gz)", "*.nii.gz"),
            ("NRRD (*.nrrd)", "*.nrrd"),
            ("MetaImage (*.mha *.mhd)", "*.mha *.mhd"),
            ("DICOM (*.dcm)", "*.dcm"),
            ("PNG Image (*.png)", "*.png"),
            ("JPEG Image (*.jpg *.jpeg)", "*.jpg *.jpeg"),
        ]

        # Create file dialog for saving
        options = QFileDialog.Options()
        file_filter = ";;".join([desc for desc, ext in formats])
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Image As", "", file_filter, options=options
        )

        if save_path:
            # Determine the selected format
            for desc, ext in formats:
                if desc == selected_filter:
                    output_extension = ext.replace("*", "").strip().split()[0]
                    break
            else:
                output_extension = os.path.splitext(save_path)[1]

            # Ensure the save path has the correct extension
            if not save_path.lower().endswith(output_extension.lower()):
                save_path += output_extension

            try:
                # Handle DICOM separately if needed
                if output_extension.lower() == ".dcm":
                    self.save_as_dicom(self.image, save_path)
                else:
                    sitk.WriteImage(self.image, save_path)
                QMessageBox.information(self, "Save Successful", f"Image saved to {save_path}")
                self.statusBar().showMessage(f'Image saved to {save_path}')
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")

    def save_as_dicom(self, image, save_path):
        # Check if the image is 3D or 2D
        dimension = image.GetDimension()
        if dimension == 3:
            # For 3D images, save each slice as a separate DICOM file
            size = image.GetSize()
            dir_name = os.path.splitext(save_path)[0]  # Remove extension for directory
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for i in range(size[2]):
                slice_i = image[:, :, i]
                slice_filename = os.path.join(dir_name, f"slice_{i}.dcm")
                sitk.WriteImage(slice_i, slice_filename)
        else:
            # For 2D images
            sitk.WriteImage(image, save_path)

    def view_image(self, image_id, image_path, modality):
        """查看图像"""
        try:
            # 构建和检查可能的完整路径列表
            possible_paths = [
                # 1. 直接使用数据库中存储的路径
                image_path,
                
                # 2. 相对于当前工作目录的uploads文件夹
                os.path.join(os.getcwd(), "uploads", image_path),
                
                # 3. 相对于系统根目录的uploads文件夹
                os.path.join(os.path.dirname(os.getcwd()), "uploads", image_path),
                
                # 4. 相对于当前目录
                os.path.join("uploads", image_path),
                
                # 5. 绝对路径(适用于Windows)
                os.path.join("d:\\pelvis\\uploads", image_path) if os.name == 'nt' else None,
            ]
            
            # 过滤掉None值
            possible_paths = [p for p in possible_paths if p]
            
            # 打印所有尝试的路径，帮助调试
            print(f"尝试打开图像，图像ID: {image_id}, 模态: {modality}")
            print("尝试的路径列表:")
            for path in possible_paths:
                path_exists = os.path.exists(path)
                print(f" - {path} (存在: {path_exists})")
                
                # 如果找到文件，使用此路径
                if path_exists:
                    full_path = path
                    print(f"找到图像文件: {full_path}")
                    break
            else:
                # 如果循环正常结束但没找到文件
                print(f"警告: 在所有可能的位置都找不到图像文件")
                QMessageBox.warning(self, "文件不存在", f"找不到图像文件。\n数据库路径: {image_path}")
                return
            
            # 根据模态类型打开不同的查看器
            if modality == "CT":
                sitk_image = sitk.ReadImage(full_path)
                self.ct_viewer = CTViewer(sitk_image, parent=self, patient_id=self.patient_id)
                self.ct_viewer.show()
                # 隐藏当前窗口
                self.hide()
            else:
                QMessageBox.information(self, "未支持的模态", f"当前不支持查看 {modality} 类型的图像")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开图像时出错: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    patient_id = 'P00001'
    viewer = MedicalImageViewer(patient_id)
    viewer.show()
    sys.exit(app.exec_())
