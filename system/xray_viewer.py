# xray_viewer.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QMessageBox, QInputDialog, QLabel
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import vtk.util.numpy_support as vtk_np
import numpy as np
from utils.file_upload import FileUploader
from datetime import datetime
import os
import SimpleITK as sitk
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from utils.progress_dialog import UploadProgressDialog
from db_manager import get_connection
from PyQt5.QtGui import QImage, QPixmap

class UploadThread(QThread):
    upload_finished = pyqtSignal(bool, str)  # 上传完成信号
    upload_progress = pyqtSignal(float)      # 上传进度信号

    def __init__(self, file_path, patient_id, image_type):
        super().__init__()
        self.file_path = file_path
        self.patient_id = patient_id
        self.image_type = image_type
        self.file_uploader = FileUploader()

    def run(self):
        try:
            relative_path = self.file_uploader.upload_medical_image(
                self.file_path,
                self.patient_id,
                self.image_type,
                self.update_progress
            )
            self.upload_finished.emit(True, relative_path)
        except Exception as e:
            self.upload_finished.emit(False, str(e))

    def update_progress(self, progress):
        self.upload_progress.emit(progress)

class XRayViewer(QWidget):
    def __init__(self, image_array, parent=None, patient_id=None):
        super().__init__(parent)
        self.parent_window = parent
        self.patient_id = patient_id
        self.file_uploader = FileUploader()
        self.image_array = image_array
        self.initUI()

    def initUI(self):
        # 创建主布局
        layout = QVBoxLayout()
        
        # 创建VTK部件和渲染器
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # 移除交互器的所有功能
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(None)

        # 如果有图像，显示它
        if self.image_array is not None:
            vtk_image = self.numpy_to_vtk_image(self.image_array)
            self.display_image(vtk_image)

        # 添加上传按钮
        self.upload_button = QPushButton("上传X光片", self)
        self.upload_button.clicked.connect(self.upload_xray_image)

        # 设置布局
        layout.addWidget(self.vtkWidget)
        layout.addWidget(self.upload_button)
        self.setLayout(layout)

        # 初始化VTK部件
        self.vtkWidget.Initialize()
        self.vtkWidget.Start()

    def numpy_to_vtk_image(self, image):
        # 处理可能的多通道
        print(f"图像维度: {image.ndim}")
        if image.ndim == 3:
            if image.shape[2] == 2:
                # 合并通道
                image = image.mean(axis=2)
            elif image.shape[2] == 3:
                # RGB图像
                image = image[:, :, ::-1]  # 转换RGB为BGR
            else:
                raise ValueError('不支持的通道数')

        # 归一化图像数据
        normalized_image = self.normalize_image(image)

        # 转换为VTK图像数据
        vtk_data_array = vtk_np.numpy_to_vtk(
            num_array=normalized_image.ravel(),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(image.shape[1], image.shape[0], 1)
        vtk_image.GetPointData().SetScalars(vtk_data_array)

        return vtk_image

    def normalize_image(self, image_array):
        min_val = image_array.min()
        max_val = image_array.max()
        if max_val - min_val == 0:
            normalized = np.zeros(image_array.shape, dtype=np.uint8)
        else:
            # 使用gamma校正进行归一化
            gamma = 0.25
            normalized = np.power(image_array/np.max(image_array), gamma) * 255
            normalized = normalized.astype(np.uint8)

        return normalized

    def display_image(self, vtk_image):
        # 移除之前的项目
        self.renderer.RemoveAllViewProps()

        # 创建翻转过滤器来垂直翻转图像
        flip_filter = vtk.vtkImageFlip()
        flip_filter.SetInputData(vtk_image)
        flip_filter.SetFilteredAxes(1)  # 1 = Y轴 (上下翻转)

        # 更新过滤器以应用更改
        flip_filter.Update()

        # 使用翻转后的图像创建actor
        flipped_image = flip_filter.GetOutput()

        # 创建图像actor
        image_actor = vtk.vtkImageActor()
        image_actor.SetInputData(flipped_image)

        # 调整窗宽窗位
        image_actor.GetProperty().SetColorWindow(255)
        image_actor.GetProperty().SetColorLevel(127)

        # 将actor添加到渲染器
        self.renderer.AddActor(image_actor)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def resizeEvent(self, event):
        """当窗口大小改变时调整图像大小"""
        super().resizeEvent(event)
        if hasattr(self, 'image_array'):
            vtk_image = self.numpy_to_vtk_image(self.image_array)
            self.display_image(vtk_image)

    def upload_xray_image(self):
        try:
            if self.image_array is None:
                QMessageBox.warning(self, "警告", "请先加载X光片！")
                return

            # 如果没有病人ID，弹出输入框
            if not self.patient_id:
                patient_id, ok = QInputDialog.getText(
                    self, 
                    "输入病人ID", 
                    "请输入病人ID:",
                    text=""
                )
                if ok and patient_id:
                    self.patient_id = patient_id
                else:
                    return

            # 保存临时文件
            temp_file = "temp_xray.mha"
            
            # 将numpy数组转换为SimpleITK图像
            sitk_image = sitk.GetImageFromArray(self.image_array)
            sitk.WriteImage(sitk_image, temp_file, True)  # True表示使用压缩

            # 创建并显示进度对话框
            self.progress_dialog = UploadProgressDialog(self)
            self.progress_dialog.show()

            # 创建并启动上传线程
            self.upload_thread = UploadThread(temp_file, self.patient_id, 'xray')
            self.upload_thread.upload_finished.connect(self.on_upload_finished)
            self.upload_thread.upload_progress.connect(self.progress_dialog.update_progress)
            self.upload_button.setEnabled(False)
            self.upload_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"准备上传失败：{str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def on_upload_finished(self, success, message):
        """上传完成的回调函数"""
        try:
            self.upload_button.setEnabled(True)
            
            if success:
                image_name = f"XRAY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mha"
                self._save_to_database(image_name, message, 'XRAY')
                if self.progress_dialog:
                    self.progress_dialog.close()
                QMessageBox.information(self, "成功", f"X光片上传成功！\n病人ID: {self.patient_id}")
            else:
                if self.progress_dialog:
                    self.progress_dialog.close()
                QMessageBox.critical(self, "错误", f"上传失败：{message}")

        finally:
            # 清理临时文件
            temp_file = "temp_xray.mha"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # 清理进度对话框
            if self.progress_dialog:
                self.progress_dialog.deleteLater()
                self.progress_dialog = None

    def _save_to_database(self, image_name, image_path, modality):
        """保存图像信息到数据库"""
        connection = get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO patient_images 
                (patient_id, image_name, image_path, modality) 
                VALUES (%s, %s, %s, %s)
            """, (
                self.patient_id,
                image_name,
                image_path,
                modality
            ))
            connection.commit()
        finally:
            cursor.close()
            connection.close()
