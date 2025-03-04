# xray_viewer.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import vtk.util.numpy_support as vtk_np
import numpy as np


class XRayViewer(QWidget):
    def __init__(self, image_array, parent_window=None):
        super().__init__()
        self.image_array = image_array
        self.parent_window = parent_window  # ✅ 记录 MedicalImageViewer 窗口
        self.initUI()

    def initUI(self):
        self.resize(800, 600)  # 默认 800x600
        self.setMinimumSize(600, 400)  # 最小 600x400

        height, width = self.image_array.shape[:2]  # 获取图像尺寸
        screen = QApplication.primaryScreen().availableGeometry()

        new_width = min(width + 100, screen.width())
        new_height = min(height + 100, screen.height())

        self.resize(new_width, new_height)  # ✅ 适应图像大小

        # Create VTK widget and renderer
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # Remove all function of the interactor
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(None)

        # Normalize and display image
        vtk_image = self.numpy_to_vtk_image(self.image_array)
        self.display_image(vtk_image)

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.vtkWidget)
        self.setLayout(layout)

        # Initialize VTK widget
        self.vtkWidget.Initialize()
        self.vtkWidget.Start()

    def closeEvent(self, event):
        """ ✅ 关闭 XRayViewer 时释放 VTK 资源 """
        try:
            if hasattr(self, "vtkWidget"):
                self.vtkWidget.GetRenderWindow().Finalize()  # ✅ 释放 OpenGL 资源
                self.vtkWidget.SetParent(None)
                self.vtkWidget.deleteLater()  # ✅ 确保对象被正确销毁
                print("VTK 资源已释放")

            if self.parent_window:
                print("返回 MedicalImageViewer")
                self.parent_window.show()
                self.parent_window.activateWindow()

            event.accept()  # 允许窗口关闭
        except Exception as e:
            print(f"关闭窗口时出错: {e}")

    def numpy_to_vtk_image(self, image):
        # Handle possible multiple channels
        print(image.ndim)
        if image.ndim == 3:
            if image.shape[2] == 2:
                # Combine channels into one
                image = image.mean(axis=2)
            elif image.shape[2] == 3:
                # RGB image
                image = image[:, :, ::-1]  # Convert RGB to BGR for VTK
            else:
                raise ValueError('Unsupported number of channels')

        # Normalize image data
        normalized_image = self.normalize_image(image)

        # Convert to VTK image data
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
            # Use gamma correction to apply normalization
            gamma = 0.25
            normalized = np.power(image_array / np.max(image_array), gamma) * 255
            normalized = normalized.astype(np.uint8)

        return normalized

    def display_image(self, vtk_image):
        # Remove previous items
        self.renderer.RemoveAllViewProps()

        # Create a flip filter to flip the image vertically
        flip_filter = vtk.vtkImageFlip()
        flip_filter.SetInputData(vtk_image)
        flip_filter.SetFilteredAxes(1)  # 1 = Y-axis (上下翻转)

        # Update the filter to apply the changes
        flip_filter.Update()

        # Use the flipped image for the actor
        flipped_image = flip_filter.GetOutput()

        # Create image actor
        image_actor = vtk.vtkImageActor()
        image_actor.SetInputData(flipped_image)

        # Adjust window/level if necessary
        image_actor.GetProperty().SetColorWindow(255)
        image_actor.GetProperty().SetColorLevel(127)

        # Add actor to the renderer
        self.renderer.AddActor(image_actor)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
