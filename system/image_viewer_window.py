import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QHeaderView, QTableWidgetItem
from PyQt5.uic import loadUi

import SimpleITK as sitk
import os

from sqlalchemy.dialects.mysql import pymysql

from ct_viewer import CTViewer
from system.db_manager import get_connection
from xray_viewer import XRayViewer


class MedicalImageViewer(QMainWindow):
    def __init__(self, patient_id=None):
        super().__init__()
        loadUi("ui/image_viewer_window.ui", self)  # 加载 .ui 文件
        self.patient_id = patient_id  # 存储病人 ID
        self.selected_image_path = None  # 初始化存储选中图像路径
        self.initUI()
        self.load_patient_images()  # 载入病人图像数据

    def initUI(self):
        # 让三个区域大小相等
        # 获取 QSplitter 控件
        self.mainSplitter.setSizes([400, 400, 400])
        self.mainSplitter.setStretchFactor(0, 1)  # 左侧
        self.mainSplitter.setStretchFactor(1, 1)  # 中间
        self.mainSplitter.setStretchFactor(2, 1)  # 右侧

        self.backButton.clicked.connect(self.go_back)
        self.splitButton.clicked.connect(self.perform_segmentation)
        self.visualizeButton.clicked.connect(self.visualize_results)

        # 初始化表格
        self.imageTable.setColumnCount(3)  # 三列（图像名、类型、时间）
        self.imageTable.setHorizontalHeaderLabels(["名称", "类型", "时间"])
        # 让三列宽度均分表格的宽度
        header = self.imageTable.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        # 监听表格点击事件，获取选中的图像路径
        self.imageTable.cellClicked.connect(self.image_selected)

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
        """从数据库加载该病人的医学图像列表"""
        try:
            connection = get_connection()  # 获取数据库连接
            if not connection:
                QMessageBox.critical(self, "数据库错误", "无法连接到数据库！")
                return

            cursor = connection.cursor()

            # 查询病人的所有医学图像
            query = """
            SELECT image_name, modality, upload_date 
            FROM patient_images 
            WHERE patient_id = %s
            ORDER BY upload_date DESC
            """
            cursor.execute(query, (self.patient_id,))
            results = cursor.fetchall()

            # 清空表格并插入新数据
            self.imageTable.setRowCount(0)  # 清空表格

            for row_idx, row_data in enumerate(results):
                self.imageTable.insertRow(row_idx)  # 添加新行

                for col_idx, value in enumerate(row_data):
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(Qt.AlignCenter)  # 文字居中
                    self.imageTable.setItem(row_idx, col_idx, item)

            cursor.close()
            connection.close()

        except pymysql.MySQLError as e:
            QMessageBox.critical(self, "数据库错误", f"加载图像列表失败: {str(e)}")

    def image_selected(self, row, column):
        """当用户点击表格中的一行时，获取图像路径"""
        try:
            image_name = self.imageTable.item(row, 0).text()  # 第一列是图像名称
            modality = self.imageTable.item(row, 1).text()  # 第二列是图像类型

            # 获取数据库中的图像路径
            connection = get_connection()
            if not connection:
                QMessageBox.critical(self, "数据库错误", "无法连接到数据库！")
                return

            cursor = connection.cursor()
            query = "SELECT image_path FROM patient_images WHERE patient_id = %s AND image_name = %s"
            cursor.execute(query, (self.patient_id, image_name))
            result = cursor.fetchone()
            cursor.close()
            connection.close()

            if result:
                self.selected_image_path = result[0]  # 存储图像路径
                print(f"选中的图像路径: {self.selected_image_path}")
            else:
                QMessageBox.warning(self, "图像未找到", f"无法找到 {image_name} 的存储路径。")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"选取图像失败: {str(e)}")

    # def open_image(self):
    #     # Open file dialog to select image
    #     options = QFileDialog.Options()
    #     file_types = "All Files (*);;DICOM Files (*.dcm);;NIfTI Files (*.nii *.nii.gz);;NRRD Files (*.nrrd);;MetaImage Files (*.mha *.mhd)"
    #     file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", file_types, options=options)
    #     if file_path:
    #         self.load_image(file_path)

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
                self.viewer = XRayViewer(image_array, parent_window=self)
            elif dimension == 3:
                self.viewer = CTViewer(self.image, parent=self)  # 传递 self 作为 parent
                self.hide()
            else:
                QMessageBox.warning(self, "不支持的图像", "选中的图像格式不支持。")
                return
            self.viewer.show()
            # self.hide()  # 隐藏 MedicalImageViewer 窗口
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    patient_id = 'P00001'
    viewer = MedicalImageViewer(patient_id)
    viewer.show()
    sys.exit(app.exec_())
