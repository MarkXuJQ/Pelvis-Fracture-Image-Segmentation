import sys
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QHeaderView, QTableWidgetItem
from PyQt5.uic import loadUi
import SimpleITK as sitk
from sqlalchemy.dialects.mysql import pymysql
from ct_viewer import CTViewer
from system.db_manager import get_connection
from xray_viewer import XRayViewer
from utils.file_upload import FileUploader
import tempfile
from utils.download_thread import DownloadThread
from utils.progress_dialog import UploadProgressDialog


class MedicalImageViewer(QMainWindow):
    def __init__(self, patient_id=None):
        super().__init__()
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建UI文件的完整路径
        ui_file = os.path.join(current_dir, "ui", "image_viewer_window.ui")
        
        try:
            loadUi(ui_file, self)  # 加载 .ui 文件
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", f"找不到UI文件: {ui_file}")
            return
            
        self.patient_id = patient_id  # 存储病人 ID
        self.selected_image_path = None  # 初始化存储选中图像路径
        self.file_uploader = FileUploader()
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
        self.imageTable.setColumnCount(4)  # 四列（图像名、类型、时间、路径）
        self.imageTable.setHorizontalHeaderLabels(["名称", "类型", "时间", "路径"])
        # 让三列宽度均分表格的宽度
        header = self.imageTable.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

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
                self.viewer = XRayViewer(image_array, parent_window=self)
            elif dimension == 3:
                self.viewer = CTViewer(self.image, parent=self)  # 传递 self 作为 parent
                self.hide()
            else:
                QMessageBox.warning(self, "不支持的图像", "选中的图像格式不支持。")
                return
            self.viewer.show()
            # self.hide()  # 隐藏 MedicalImageViewer 窗口
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
