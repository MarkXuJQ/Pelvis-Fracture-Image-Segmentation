import sys
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QMessageBox
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import SimpleITK as sitk
import numpy as np
import vtk.util.numpy_support as vtk_np
import vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkInteractionWidgets import vtkResliceCursor, vtkResliceCursorWidget, \
    vtkResliceCursorLineRepresentation
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkActor, vtkPolyDataMapper, vtkCamera
from vtkmodules.vtkIOImage import vtkImageImport
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.util.vtkConstants import VTK_FLOAT
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2

# Ensures compatibility with macOS for layering issues with PyQt5 and VTK
os.environ["QT_MAC_WANTS_LAYER"] = "1"


class CTViewer(QWidget):
    def __init__(self, sitk_image, render_model=False):
        super().__init__()
        self.sitk_image = sitk_image
        self.render_model = render_model
        self.image_data = self.sitk_to_vtk_image(sitk_image)
        self.reslice_cursor = vtkResliceCursor()
        self.reslice_cursor.SetCenter(self.image_data.GetCenter())
        self.reslice_cursor.SetImage(self.image_data)
        self.reslice_cursor.SetThickMode(False)
        self.reslice_widgets = []
        self.reslice_representations = []

        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ui_file = os.path.join(current_dir, "system", "ui", "ct_viewer.ui")
        
        uic.loadUi(ui_file, self)

        self.setup_sliders()
        self.Generate_Model.clicked.connect(self.generate_model)
        self.Back.clicked.connect(self.back_to_MainWindow)
        # Setup views using the widgets from UI file
        self.setup_reslice_views()
        self.synchronize_views()
        self.setup_model_view()
        if self.render_model:
            self.generate_and_display_model()

        # Add mouse interaction states
        self.is_panning = False
        self.last_mouse_pos = None
        self.zoom_factor = 1.0
        self.window_level = None
        self.window_width = None

        # Setup mouse interaction for all views
        self.setup_mouse_interaction()

    def back_to_MainWindow(self):
        # 在需要时才导入 DoctorUI
        from doctor_window import DoctorUI
        
        main_window = self.parent()
        self.close()
        main_window.close()
        self.main_window = DoctorUI(1)
        self.main_window.show()

    def generate_model(self):
        self.render_model = True
        self.generate_and_display_model()

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


# Main execution of the application
if __name__ == "__main__":
    app = QApplication(sys.argv)

    image_path = r"ct_seg\data\PENGWIN_CT_train_images\001.mha"  # Path to medical image file
    sitk_image = sitk.ReadImage(image_path)  # Read the image using SimpleITK
    viewer = CTViewer(sitk_image)  # Create an instance of the viewer with the image
    viewer.show()  # Display the viewer window
    sys.exit(app.exec_())  # Execute the application