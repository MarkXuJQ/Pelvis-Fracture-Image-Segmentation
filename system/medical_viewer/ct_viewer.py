import sys
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QMessageBox, QPushButton, QInputDialog, QProgressBar, QLabel
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import SimpleITK as sitk
import numpy as np
import vtk.util.numpy_support as vtk_np
import vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkInteractionWidgets import vtkResliceCursor, vtkResliceCursorWidget, vtkResliceCursorLineRepresentation
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkActor, vtkPolyDataMapper, vtkCamera
from vtkmodules.vtkIOImage import vtkImageImport
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.util.vtkConstants import VTK_FLOAT
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from utils.file_upload import FileUploader
from datetime import datetime
from system.database.db_manager import get_connection
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from utils.progress_dialog import UploadProgressDialog


# Fix layering issues with PyQt5 and VTK
os.environ["QT_MAC_WANTS_LAYER"] = "1"


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


class CTViewer(QWidget):
    def __init__(self, sitk_image, parent=None, render_model=False, patient_id=None):
        super().__init__(parent)
        self.parent_window = parent
        self.render_model = render_model
        self.patient_id = patient_id
        self.sitk_image = sitk_image
        
        # 重要: 在加载UI前不要设置任何布局
        self.setup_ui()
        
        # 初始化 VTK 相关
        self.image_data = self.sitk_to_vtk_image(sitk_image)
        self.reslice_cursor = vtkResliceCursor()
        self.reslice_cursor.SetCenter(self.image_data.GetCenter())
        self.reslice_cursor.SetImage(self.image_data)
        self.reslice_cursor.SetThickMode(False)
        self.reslice_widgets = []
        self.reslice_representations = []
        self.file_uploader = FileUploader()

        # 设置滑块和3D视图
        self.setup_sliders()
        self.setup_buttons()
        self.setup_reslice_views()
        self.synchronize_views()
        self.setup_model_view()
        
        # 如果需要渲染3D模型
        if self.render_model:
            self.generate_and_display_model()

        # 添加鼠标交互状态
        self.is_panning = False
        self.last_mouse_pos = None
        self.zoom_factor = 1.0
        self.window_level = None
        self.window_width = None

        # 设置所有视图的鼠标交互
        self.setup_mouse_interaction()

        # 添加上传按钮
        self.setup_upload_button()
        
        # 移除之前的进度条和状态标签
        self.progress_dialog = None

    def setup_ui(self):
        """加载UI文件"""
        try:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(current_dir))  # 上两级目录
            
            # 尝试多个可能的UI文件路径
            ui_paths = [
                os.path.join(current_dir, "ui", "ct_viewer.ui"),     # 脚本同级ui目录
                os.path.join(base_dir, "ui", "ct_viewer.ui"),        # 系统根目录下的ui
                os.path.join(os.getcwd(), "ui", "ct_viewer.ui"),     # 当前工作目录下的ui
                os.path.join(os.getcwd(), "system", "ui", "ct_viewer.ui"),  # system/ui下
                "d:\\pelvis\\system\\ui\\ct_viewer.ui"               # 硬编码路径(从日志看出)
            ]
            
            ui_file = None
            for path in ui_paths:
                if os.path.exists(path):
                    ui_file = path
                    break
            
            if ui_file:
                print(f"加载UI文件: {ui_file}")
                uic.loadUi(ui_file, self)
                print(f"已成功加载UI文件: {ui_file}")
            else:
                QMessageBox.warning(self, "警告", "找不到UI文件，使用默认布局")
                self.create_default_ui()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载UI文件失败: {str(e)}")
            self.create_default_ui()

    def create_default_ui(self):
        """创建默认UI"""
        print("创建默认UI布局")
        main_layout = QHBoxLayout(self)
        
        # 左侧控制面板
        controls_layout = QVBoxLayout()
        self.Generate_Model = QPushButton("生成3D模型", self)
        self.Generate_Model.setObjectName("Generate_Model")
        controls_layout.addWidget(self.Generate_Model)
        
        self.Back = QPushButton("返回", self)
        self.Back.setObjectName("Back")
        controls_layout.addWidget(self.Back)
        
        main_layout.addLayout(controls_layout, 1)
        
        # 视图布局
        views_layout = QGridLayout()
        
        # 创建轴向视图容器
        axial_container = QWidget()
        axial_layout = QVBoxLayout(axial_container)
        axial_label = QLabel("轴向视图")
        self.vtkWidget_axial = QVTKRenderWindowInteractor(axial_container)
        self.vtkWidget_axial.setObjectName("vtkWidget_axial")
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setObjectName("axial_slider")
        axial_layout.addWidget(axial_label)
        axial_layout.addWidget(self.vtkWidget_axial)
        axial_layout.addWidget(self.axial_slider)
        
        # 创建冠状视图容器
        coronal_container = QWidget()
        coronal_layout = QVBoxLayout(coronal_container)
        coronal_label = QLabel("冠状视图")
        self.vtkWidget_coronal = QVTKRenderWindowInteractor(coronal_container)
        self.vtkWidget_coronal.setObjectName("vtkWidget_coronal")
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setObjectName("coronal_slider")
        coronal_layout.addWidget(coronal_label)
        coronal_layout.addWidget(self.vtkWidget_coronal)
        coronal_layout.addWidget(self.coronal_slider)
        
        # 创建矢状视图容器
        sagittal_container = QWidget()
        sagittal_layout = QVBoxLayout(sagittal_container)
        sagittal_label = QLabel("矢状视图")
        self.vtkWidget_sagittal = QVTKRenderWindowInteractor(sagittal_container)
        self.vtkWidget_sagittal.setObjectName("vtkWidget_sagittal")
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setObjectName("sagittal_slider")
        sagittal_layout.addWidget(sagittal_label)
        sagittal_layout.addWidget(self.vtkWidget_sagittal)
        sagittal_layout.addWidget(self.sagittal_slider)
        
        # 创建3D模型视图容器
        model_container = QWidget()
        model_layout = QVBoxLayout(model_container)
        model_label = QLabel("3D模型")
        self.model_vtkWidget = QVTKRenderWindowInteractor(model_container)
        self.model_vtkWidget.setObjectName("model_vtkWidget")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_vtkWidget)
        
        # 添加到网格布局
        views_layout.addWidget(axial_container, 0, 0)
        views_layout.addWidget(model_container, 0, 1)
        views_layout.addWidget(coronal_container, 1, 0)
        views_layout.addWidget(sagittal_container, 1, 1)
        
        main_layout.addLayout(views_layout, 4)
        self.setLayout(main_layout)

    def setup_buttons(self):
        """设置按钮连接"""
        # 查找并连接生成模型按钮
        generate_model_btn = self.findChild(QPushButton, 'Generate_Model')
        if generate_model_btn:
            generate_model_btn.clicked.connect(self.generate_model)
        
        # 查找并连接返回按钮
        back_btn = self.findChild(QPushButton, 'Back')
        if back_btn:
            back_btn.clicked.connect(self.back_to_MainWindow)

    def setup_upload_button(self):
        """设置上传按钮"""
        self.upload_button = QPushButton("上传CT图像", self)
        self.upload_button.clicked.connect(self.upload_ct_image)
        
        # 尝试找到现有布局并添加按钮
        if hasattr(self, 'verticalLayout') and self.verticalLayout:
            self.verticalLayout.addWidget(self.upload_button)
        elif hasattr(self, 'layout') and self.layout():
            # 找到最左侧的垂直布局
            left_layout = None
            layout = self.layout()
            
            # 如果是水平布局，尝试获取第一个子布局
            if isinstance(layout, QHBoxLayout) and layout.count() > 0:
                item = layout.itemAt(0)
                if item and item.layout():
                    left_layout = item.layout()
            
            if left_layout:
                left_layout.addWidget(self.upload_button)
            else:
                layout.addWidget(self.upload_button)

    def setup_reslice_views(self):
        # Initialize VTK widgets
        self.findChild(QVTKRenderWindowInteractor, 'vtkWidget_axial').Initialize()
        self.findChild(QVTKRenderWindowInteractor, 'vtkWidget_coronal').Initialize()
        self.findChild(QVTKRenderWindowInteractor, 'vtkWidget_sagittal').Initialize()

        # Setup using the widgets from UI file
        self.vtkWidget_axial = self.setup_reslice_view(self.findChild(QVTKRenderWindowInteractor, 'vtkWidget_axial'), 2)
        self.vtkWidget_coronal = self.setup_reslice_view(
            self.findChild(QVTKRenderWindowInteractor, 'vtkWidget_coronal'), 1)
        self.vtkWidget_sagittal = self.setup_reslice_view(
            self.findChild(QVTKRenderWindowInteractor, 'vtkWidget_sagittal'), 0)

    def sitk_to_vtk_image(self, sitk_image):
        # Convert SimpleITK image to VTK image
        size = sitk_image.GetSize()
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        image_array = sitk.GetArrayFromImage(sitk_image)  # Convert SITK image to NumPy array
        image_array = np.transpose(image_array.astype(np.float32), (2, 1, 0)).flatten(order='F')
        vtk_image = vtkImageData()
        vtk_image.SetDimensions(size)
        vtk_image.SetSpacing(spacing)
        vtk_image.SetOrigin(origin)
        vtk_image.AllocateScalars(VTK_FLOAT, 1)
        vtk_data_array = vtk.util.numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        vtk_data_array[:] = image_array  # Assign pixel data to VTK image
        return vtk_image

    def setup_sliders(self):
        # Get sliders from UI
        self.axial_slider = self.findChild(QSlider, 'axial_slider')
        self.coronal_slider = self.findChild(QSlider, 'coronal_slider')
        self.sagittal_slider = self.findChild(QSlider, 'sagittal_slider')

        # Debug print to verify sliders are found
        if not all([self.axial_slider, self.coronal_slider, self.sagittal_slider]):
            print("Warning: Not all sliders were found in the UI")
            return

        # Get image dimensions and center positions
        dims = self.image_data.GetDimensions()
        spacing = self.image_data.GetSpacing()

        # This allows navigation to both sides of the center
        self.axial_slider.setRange(-dims[2] // 2, dims[2] // 2)
        self.coronal_slider.setRange(-dims[1] // 2, dims[1] // 2)
        self.sagittal_slider.setRange(-dims[0] // 2, dims[0] // 2)

        # Set initial values to center (0)
        self.axial_slider.setValue(0)
        self.coronal_slider.setValue(0)
        self.sagittal_slider.setValue(0)

        # Connect slider value changes to update functions with offset calculation
        self.axial_slider.valueChanged.connect(
            lambda value: self.update_slice_position(2, value + dims[2] // 2))
        self.coronal_slider.valueChanged.connect(
            lambda value: self.update_slice_position(1, -value))
        self.sagittal_slider.valueChanged.connect(
            lambda value: self.update_slice_position(0, -value))

    def update_slice_position(self, orientation, value):
        center = list(self.reslice_cursor.GetCenter())
        dims = self.image_data.GetDimensions()
        spacing = self.image_data.GetSpacing()

        # Calculate physical position based on image spacing
        physical_pos = value * spacing[orientation]
        center[orientation] = physical_pos

        self.reslice_cursor.SetCenter(center)

        # Update all views
        for widget in self.reslice_widgets:
            widget.Render()

    def setup_reslice_view(self, placeholder_widget, orientation):
        # Create and configure renderer for the view
        renderer = vtkRenderer()
        placeholder_widget.GetRenderWindow().AddRenderer(renderer)

        # Configure the reslice cursor representation and widget
        reslice_representation = vtkResliceCursorLineRepresentation()
        reslice_widget = vtkResliceCursorWidget()
        reslice_widget.SetInteractor(placeholder_widget)
        reslice_widget.SetRepresentation(reslice_representation)
        reslice_representation.GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(self.reslice_cursor)
        reslice_representation.GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(orientation)

        # Set window/level for the reslice representation
        scalar_range = self.image_data.GetScalarRange()
        reslice_representation.SetWindowLevel(scalar_range[1] - scalar_range[0],
                                              (scalar_range[1] + scalar_range[0]) / 2)

        # Input data for reslice view
        reslice_representation.GetReslice().SetInputData(self.image_data)
        reslice_representation.GetResliceCursor().SetImage(self.image_data)

        # Enable the reslice widget and add it to lists
        reslice_widget.SetEnabled(1)
        reslice_widget.On()
        self.reslice_widgets.append(reslice_widget)
        self.reslice_representations.append(reslice_representation)

        # Set camera orientation based on the slice plane
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        center = self.image_data.GetCenter()
        camera.SetFocalPoint(center)
        if orientation == 2:
            camera.SetPosition(center[0], center[1], center[2] + 1000)
            camera.SetViewUp(0, -1, 0)
        elif orientation == 1:
            camera.SetPosition(center[0], center[1] - 1000, center[2])
            camera.SetViewUp(0, 0, 1)
        elif orientation == 0:
            camera.SetPosition(center[0] - 1000, center[1], center[2])
            camera.SetViewUp(0, 0, 1)
        camera.SetParallelScale(
            max([dim * spc for dim, spc in zip(self.image_data.GetDimensions(), self.image_data.GetSpacing())]) / 2.0)

        return placeholder_widget

    def synchronize_views(self):
        # Sync views so that interaction in one updates others
        for reslice_widget in self.reslice_widgets:
            reslice_widget.AddObserver("InteractionEvent", self.on_interaction)
            reslice_widget.AddObserver("EndInteractionEvent", self.on_interaction)

    def on_interaction(self, caller, event):
        # Get current cursor position
        center = self.reslice_cursor.GetCenter()
        dims = self.image_data.GetDimensions()

        # Update slider positions based on cursor position
        # For axial view
        self.axial_slider.setValue(int(center[2] / self.image_data.GetSpacing()[2]) - dims[2] // 2)

        # For coronal and sagittal views (using negative mapping)
        self.coronal_slider.setValue(-int(center[1] / self.image_data.GetSpacing()[1]))
        self.sagittal_slider.setValue(-int(center[0] / self.image_data.GetSpacing()[0]))

        # Update all views
        for reslice_widget in self.reslice_widgets:
            if reslice_widget != caller:
                reslice_widget.Render()

    def setup_model_view(self):
        # Use the model widget from UI file
        self.model_vtkWidget = self.findChild(QVTKRenderWindowInteractor, 'model_vtkWidget')
        self.model_renderer = vtkRenderer()
        render_window = self.model_vtkWidget.GetRenderWindow()
        render_window.AddRenderer(self.model_renderer)
        self.model_vtkWidget.Initialize()

    def generate_and_display_model(self):
        # Generate a 3D model from the image data and display it
        self.model_renderer.RemoveAllViewProps()
        poly_data = self.generate_3d_model()
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtkActor()
        actor.SetMapper(mapper)

        # Add actor to the renderer
        self.model_renderer.AddActor(actor)
        self.model_renderer.SetBackground(0.68, 0.85, 0.9)
        self.model_renderer.ResetCamera()
        self.model_vtkWidget.GetRenderWindow().Render()

    def generate_3d_model(self):
        # Generate 3D model using thresholding and marching cubes for surface extraction
        image_array = sitk.GetArrayFromImage(self.sitk_image).astype(np.float32)
        thresholded_image = np.where((image_array >= 300) & (image_array <= 3000), image_array, 0)
        shape = thresholded_image.shape
        flat_image_array = thresholded_image.flatten(order="C")

        importer = vtkImageImport()
        importer.CopyImportVoidPointer(flat_image_array, flat_image_array.nbytes)
        importer.SetDataScalarType(VTK_FLOAT)
        importer.SetNumberOfScalarComponents(1)
        importer.SetWholeExtent(0, shape[2] - 1, 0, shape[1] - 1, 0, shape[0] - 1)
        importer.SetDataExtent(0, shape[2] - 1, 0, shape[1] - 1, 0, shape[0] - 1)
        importer.SetDataSpacing(self.sitk_image.GetSpacing())
        importer.SetDataOrigin(self.sitk_image.GetOrigin())
        importer.Update()

        # Run marching cubes algorithm to generate a surface model
        vtk_image = importer.GetOutput()
        contour_filter = vtkMarchingCubes()
        contour_filter.SetInputData(vtk_image)
        contour_filter.SetValue(0, 300)
        contour_filter.Update()
        return contour_filter.GetOutput()

    def closeEvent(self, event):
        # Close event to finalize and clean up reslice widgets
        for widget in [self.vtkWidget_axial, self.vtkWidget_coronal, self.vtkWidget_sagittal, self.model_vtkWidget]:
            if widget:
                widget.Finalize()
                del widget
        super().closeEvent(event)

    def setup_mouse_interaction(self):
        """Setup mouse interaction for all views"""
        for widget in [self.vtkWidget_axial, self.vtkWidget_coronal, self.vtkWidget_sagittal]:
            interactor = widget.GetRenderWindow().GetInteractor()

            # Add observers for mouse events
            interactor.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
            interactor.AddObserver("LeftButtonReleaseEvent", self.on_left_button_release)
            interactor.AddObserver("MouseMoveEvent", self.on_mouse_move)
            interactor.AddObserver("MouseWheelForwardEvent", self.on_mouse_wheel_forward)
            interactor.AddObserver("MouseWheelBackwardEvent", self.on_mouse_wheel_backward)
            interactor.AddObserver("RightButtonPressEvent", self.on_right_button_press)
            interactor.AddObserver("RightButtonReleaseEvent", self.on_right_button_release)

            # Set interaction style
            style = interactor.GetInteractorStyle()
            style.SetCurrentStyleToTrackballCamera()

    def on_left_button_press(self, obj, event):
        """Handle left button press for panning"""
        self.is_panning = True
        self.last_mouse_pos = obj.GetEventPosition()

    def on_left_button_release(self, obj, event):
        """Handle left button release"""
        self.is_panning = False
        self.last_mouse_pos = None

    def on_mouse_move(self, obj, event):
        """Handle mouse movement for panning and window/level adjustment"""
        if not self.last_mouse_pos:
            return

        current_pos = obj.GetEventPosition()
        dx = current_pos[0] - self.last_mouse_pos[0]
        dy = current_pos[1] - self.last_mouse_pos[1]

        if self.is_panning:
            # Pan the camera
            camera = obj.GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
            fp = list(camera.GetFocalPoint())
            pos = list(camera.GetPosition())

            camera.SetFocalPoint(fp[0] - dx, fp[1] - dy, fp[2])
            camera.SetPosition(pos[0] - dx, pos[1] - dy, pos[2])

        elif self.window_level is not None:
            # Adjust window/level
            self.window_width += dx * 10
            self.window_level += dy * 10
            self.update_window_level()

        self.last_mouse_pos = current_pos
        obj.GetRenderWindow().Render()

    def on_mouse_wheel_forward(self, obj, event):
        """Handle mouse wheel forward for zooming in"""
        self.zoom_factor *= 1.1
        self.apply_zoom(obj)

    def on_mouse_wheel_backward(self, obj, event):
        """Handle mouse wheel backward for zooming out"""
        self.zoom_factor /= 1.1
        self.apply_zoom(obj)

    def on_right_button_press(self, obj, event):
        """Handle right button press for window/level adjustment"""
        self.window_level = self.reslice_representations[0].GetWindowLevel()
        self.window_width = self.reslice_representations[0].GetWindow()
        self.last_mouse_pos = obj.GetEventPosition()

    def on_right_button_release(self, obj, event):
        """Handle right button release"""
        self.window_level = None
        self.last_mouse_pos = None

    def apply_zoom(self, obj):
        """Apply zoom to the camera"""
        camera = obj.GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
        camera.SetParallelScale(camera.GetParallelScale() / self.zoom_factor)
        self.zoom_factor = 1.0  # Reset zoom factor
        obj.GetRenderWindow().Render()

    def update_window_level(self):
        """Update window/level for all views"""
        for rep in self.reslice_representations:
            rep.SetWindowLevel(self.window_width, self.window_level)
            rep.GetResliceCursorWidget().Render()

    def upload_ct_image(self):
        try:
            if not hasattr(self, 'sitk_image'):
                QMessageBox.warning(self, "警告", "请先加载CT图像！")
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
            temp_file = "temp_ct.mha"
            sitk.WriteImage(self.sitk_image, temp_file)

            # 创建并显示进度对话框
            self.progress_dialog = UploadProgressDialog(self)
            self.progress_dialog.show()

            # 创建并启动上传线程
            self.upload_thread = UploadThread(temp_file, self.patient_id, 'ct')
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
                image_name = f"CT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mha"
                # 打印上传路径用于调试
                print(f"文件上传成功，路径为: {message}")
                # 存储数据库记录前确认路径格式
                db_path = message
                # 如果是Windows路径格式，转换为统一格式
                if '\\' in db_path:
                    db_path = db_path.replace('\\', '/')
                
                self._save_to_database(image_name, db_path, 'CT')
                if self.progress_dialog:
                    self.progress_dialog.close()
                QMessageBox.information(self, "成功", f"CT图像上传成功！\n病人ID: {self.patient_id}\n保存路径: {db_path}")
            else:
                if self.progress_dialog:
                    self.progress_dialog.close()
                QMessageBox.critical(self, "错误", f"上传失败：{message}")

        finally:
            # 清理临时文件
            temp_file = "temp_ct.mha"
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

    def generate_model(self):
        """生成3D模型"""
        self.render_model = True
        self.generate_and_display_model()
        
    def back_to_MainWindow(self):
        """返回到主窗口"""
        # 在这里动态导入，避免循环导入
        try:
            from image_viewer_window import MedicalImageViewer
            if hasattr(self, 'medical_image_viewer'):
                return self.medical_image_viewer
            else:
                self.medical_image_viewer = MedicalImageViewer(self.patient_id)
                return self.medical_image_viewer
        except ImportError:
            try:
                from system.image_viewer_window import MedicalImageViewer
                if hasattr(self, 'medical_image_viewer'):
                    return self.medical_image_viewer
                else:
                    self.medical_image_viewer = MedicalImageViewer(self.patient_id)
                    return self.medical_image_viewer
            except ImportError:
                QMessageBox.warning(self, "警告", "无法返回主窗口，找不到相关模块")
                return None
