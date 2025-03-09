from PyQt5.QtCore import QThread, pyqtSignal
import SimpleITK as sitk
from utils.file_upload import FileUploader
import os

class DownloadThread(QThread):
    progress = pyqtSignal(float)
    finished = pyqtSignal(object, bool, str)  # (image, success, message)

    def __init__(self, image_path, temp_path):
        super().__init__()
        self.image_path = image_path
        self.temp_path = temp_path
        self.file_uploader = FileUploader()

    def run(self):
        try:
            # 下载文件
            self.file_uploader.get_medical_image(
                self.image_path, 
                self.temp_path,
                progress_callback=lambda p: self.progress.emit(p)
            )

            # 读取图像
            image = sitk.ReadImage(self.temp_path)
            self.finished.emit(image, True, "成功")

        except Exception as e:
            self.finished.emit(None, False, str(e)) 