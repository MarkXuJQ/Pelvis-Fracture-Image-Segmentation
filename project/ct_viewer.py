import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import SimpleITK as sitk
import numpy as np
import vtk.util.numpy_support as vtk_np

import vtk

# VTK Imports
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkInteractionWidgets import (
    vtkResliceCursor,
    vtkResliceCursorWidget,
    vtkResliceCursorLineRepresentation,
)
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkActor,
    vtkPolyDataMapper,
    vtkImageActor,
    vtkCamera,
)
from vtkmodules.vtkIOImage import vtkImageImport  # Corrected import
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.util.vtkConstants import VTK_FLOAT, VTK_UNSIGNED_CHAR
import vtkmodules.vtkInteractionStyle  # Necessary for interactor style
import vtkmodules.vtkRenderingOpenGL2  # Necessary for rendering

# Ensure that VTK's OpenGL context is correctly set up on macOS
os.environ["QT_MAC_WANTS_LAYER"] = "1"


class CTViewer(QWidget):
    def __init__(self, sitk_image, render_model=False):
        super().__init__()

        # Store the SimpleITK image
        self.sitk_image = sitk_image
        self.render_model = render_model

        # Convert SimpleITK image to VTK format
        self.image_data = self.sitk_to_vtk_image(sitk_image)

        # Print image data info
        print("Image Dimensions:", self.image_data.GetDimensions())
        print("Scalar Range:", self.image_data.GetScalarRange())
        print("Scalar Type:", self.image_data.GetScalarTypeAsString())

        # Initialize reslice cursor for interactive crosshairs
        self.reslice_cursor = vtkResliceCursor()
        self.reslice_cursor.SetCenter(self.image_data.GetCenter())
        self.reslice_cursor.SetImage(self.image_data)
        self.reslice_cursor.SetThickMode(False)

        # Store reslice widgets and representations
        self.reslice_widgets = []
        self.reslice_representations = []

        # Create the UI elements programmatically
        self.init_ui()

        # Set up reslice views (axial, coronal, sagittal)
        self.setup_reslice_views()

        # Synchronize the views
        self.synchronize_views()

        # Set up the 3D model view
        self.setup_model_view()

        # Initialize model rendering
        if self.render_model:
            self.generate_and_display_model()

    def init_ui(self):
        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create a horizontal layout for the reslice views
        reslice_layout = QHBoxLayout()

        # Create placeholders for the VTK widgets
        self.vtkWidget_axial = QWidget()
        self.vtkWidget_coronal = QWidget()
        self.vtkWidget_sagittal = QWidget()

        # Set minimum sizes for the widgets (optional)
        self.vtkWidget_axial.setMinimumSize(200, 200)
        self.vtkWidget_coronal.setMinimumSize(200, 200)
        self.vtkWidget_sagittal.setMinimumSize(200, 200)

        # Add the placeholders to the reslice layout
        reslice_layout.addWidget(self.vtkWidget_axial)
        reslice_layout.addWidget(self.vtkWidget_coronal)
        reslice_layout.addWidget(self.vtkWidget_sagittal)

        # Add the reslice layout to the main layout
        main_layout.addLayout(reslice_layout)

        # Create a placeholder for the 3D model view
        self.model_vtkWidget = QWidget()
        self.model_vtkWidget.setMinimumSize(600, 400)  # Adjust size as needed

        # Add the model view to the main layout
        main_layout.addWidget(self.model_vtkWidget)

        # Set the main layout
        self.setLayout(main_layout)

    def setup_reslice_views(self):
        # Set up reslice views (axial, coronal, sagittal)
        self.vtkWidget_axial = self.setup_reslice_view(self.vtkWidget_axial, 2)    # Axial (Z-axis)
        self.vtkWidget_coronal = self.setup_reslice_view(self.vtkWidget_coronal, 1)  # Coronal (Y-axis)
        self.vtkWidget_sagittal = self.setup_reslice_view(self.vtkWidget_sagittal, 0) # Sagittal (X-axis)

    def sitk_to_vtk_image(self, sitk_image):
        # Get the image dimensions
        size = sitk_image.GetSize()  # (x_size, y_size, z_size)
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        
        # Convert SimpleITK image to a NumPy array
        image_array = sitk.GetArrayFromImage(sitk_image)  # Shape: (z_size, y_size, x_size)
        
        # Ensure the data is in the correct format
        image_array = image_array.astype(np.float32)
        
        # Transpose the array to (x_size, y_size, z_size)
        image_array = np.transpose(image_array, (2, 1, 0))
        
        # Flatten the array in Fortran order (column-major)
        flat_image_array = image_array.flatten(order='F')
        
        # Create vtkImageData object
        vtk_image = vtkImageData()
        vtk_image.SetDimensions(size)
        vtk_image.SetSpacing(spacing)
        vtk_image.SetOrigin(origin)
        vtk_image.AllocateScalars(VTK_FLOAT, 1)
        
        # Copy the data into the VTK image
        vtk_data_array = vtk.util.numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        vtk_data_array[:] = flat_image_array  # Copy the data
        
        return vtk_image

    def setup_reslice_view(self, placeholder_widget, orientation):
        # Create a QVTKRenderWindowInteractor
        render_window_interactor = QVTKRenderWindowInteractor(placeholder_widget)

        # Set layout for the placeholder widget
        layout = QVBoxLayout(placeholder_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(render_window_interactor)

        # Create a renderer and add it to the render window
        renderer = vtkRenderer()
        render_window_interactor.GetRenderWindow().AddRenderer(renderer)

        # Create the reslice cursor representation and widget
        reslice_representation = vtkResliceCursorLineRepresentation()
        reslice_widget = vtkResliceCursorWidget()
        reslice_widget.SetInteractor(render_window_interactor)
        reslice_widget.SetRepresentation(reslice_representation)

        # Set the reslice cursor via the cursor algorithm
        reslice_cursor_algorithm = reslice_representation.GetResliceCursorActor().GetCursorAlgorithm()
        reslice_cursor_algorithm.SetResliceCursor(self.reslice_cursor)
        reslice_cursor_algorithm.SetReslicePlaneNormal(orientation)

        # Configure the reslice representation
        scalar_range = self.image_data.GetScalarRange()
        window = scalar_range[1] - scalar_range[0]
        level = (scalar_range[1] + scalar_range[0]) / 2
        reslice_representation.SetWindowLevel(window, level)
        reslice_representation.GetReslice().SetInputData(self.image_data)
        reslice_representation.GetResliceCursor().SetImage(self.image_data)

        # Adjust the crosshair line properties
        reslice_cursor_actor = reslice_representation.GetResliceCursorActor()
        axes = []
        if orientation == 2:  # Axial (Z-axis)
            axes = [0, 1]  # X and Y axes are visible
        elif orientation == 1:  # Coronal (Y-axis)
            axes = [0, 2]  # X and Z axes are visible
        elif orientation == 0:  # Sagittal (X-axis)
            axes = [1, 2]  # Y and Z axes are visible

        for i in axes:
            centerline_property = reslice_cursor_actor.GetCenterlineProperty(i)
            centerline_property.SetLineWidth(3.0)
            # Optional: Set color
            # centerline_property.SetColor(1.0, 0.0, 0.0)  # Red

        # Set the interactor style to vtkInteractorStyleImage
        interactor_style = vtk.vtkInteractorStyleImage()
        render_window_interactor.SetInteractorStyle(interactor_style)

        # Disable panning and rotation by overriding interaction methods
        interactor_style.OnPan = lambda *args: None
        interactor_style.OnRotate = lambda *args: None
        interactor_style.OnZoom = lambda *args: None
        interactor_style.OnMouseWheelForward = lambda *args: None
        interactor_style.OnMouseWheelBackward = lambda *args: None
        interactor_style.OnLeftButtonDown = lambda *args: None
        interactor_style.OnLeftButtonUp = lambda *args: None
        interactor_style.OnMiddleButtonDown = lambda *args: None
        interactor_style.OnMiddleButtonUp = lambda *args: None
        interactor_style.OnRightButtonDown = lambda *args: None
        interactor_style.OnRightButtonUp = lambda *args: None

        # Initialize the widget
        reslice_widget.SetEnabled(1)
        reslice_widget.On()

        # Store the widgets for synchronization
        self.reslice_widgets.append(reslice_widget)
        self.reslice_representations.append(reslice_representation)

        # Reset the camera to fit the image
        renderer.ResetCamera()

        # Adjust the camera for the specific orientation
        camera = renderer.GetActiveCamera()
        center = self.image_data.GetCenter()
        camera.SetFocalPoint(center)

        if orientation == 2:  # Axial (Z-axis)
            camera.SetPosition(center[0], center[1], center[2] + 1000)
            camera.SetViewUp(0, -1, 0)
        elif orientation == 1:  # Coronal (Y-axis)
            camera.SetPosition(center[0], center[1] - 1000, center[2])
            camera.SetViewUp(0, 0, 1)
        elif orientation == 0:  # Sagittal (X-axis)
            camera.SetPosition(center[0] - 1000, center[1], center[2])
            camera.SetViewUp(0, 0, 1)

        # Adjust the parallel scale to fill the window
        dims = self.image_data.GetDimensions()
        spacing = self.image_data.GetSpacing()
        pixel_size = [dims[i] * spacing[i] for i in range(2)]
        parallel_scale = max(pixel_size) / 2.0
        camera.SetParallelScale(parallel_scale)

        # Initialize the interactor
        render_window_interactor.Initialize()

        # Render the scene
        render_window_interactor.Render()

        return render_window_interactor





    def synchronize_views(self):
        for reslice_widget in self.reslice_widgets:
            reslice_widget.AddObserver("InteractionEvent", self.on_interaction)
            reslice_widget.AddObserver("EndInteractionEvent", self.on_interaction)

    def on_interaction(self, caller, event):
        for reslice_widget in self.reslice_widgets:
            if reslice_widget != caller:
                reslice_widget.Render()

    def setup_model_view(self):
        # Create a QVTKRenderWindowInteractor
        render_window_interactor = QVTKRenderWindowInteractor(self.model_vtkWidget)

        # Set layout for the model placeholder widget
        layout = QVBoxLayout(self.model_vtkWidget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(render_window_interactor)

        # Create a renderer and add to the render window
        self.model_renderer = vtkRenderer()
        render_window = render_window_interactor.GetRenderWindow()
        render_window.AddRenderer(self.model_renderer)

        # Initialize the interactor
        render_window_interactor.Initialize()

        self.model_vtkWidget = render_window_interactor

    def generate_and_display_model(self):
        # Clear any existing actors
        self.model_renderer.RemoveAllViewProps()

        # Generate the 3D model
        poly_data = self.generate_3d_model()

        # Create mapper and actor
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtkActor()
        actor.SetMapper(mapper)

        # Add actor to the renderer
        self.model_renderer.AddActor(actor)
        self.model_renderer.SetBackground(0.68, 0.85, 0.9)  # Light blue background
        self.model_renderer.ResetCamera()
        self.model_vtkWidget.GetRenderWindow().Render()

    def generate_3d_model(self):
        # Apply threshold to isolate certain structures (e.g., bones)
        lower_threshold = 0.5  # Adjust based on your image data
        upper_threshold = 1.0  # Adjust based on your image data

        # Convert the SimpleITK image to a numpy array
        image_array = sitk.GetArrayFromImage(self.sitk_image)
        image_array = image_array.astype(np.float32)

        # Apply threshold
        thresholded_image = np.where(
            (image_array >= lower_threshold) & (image_array <= upper_threshold), image_array, 0
        )

        # Convert numpy array to vtkImageData
        shape = thresholded_image.shape  # (slices, height, width)
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
        vtk_image = importer.GetOutput()

        # Extract a 3D surface from the thresholded volume using Marching Cubes algorithm
        contour_filter = vtkMarchingCubes()
        contour_filter.SetInputData(vtk_image)
        contour_filter.SetValue(0, lower_threshold)  # Isosurface value
        contour_filter.Update()

        # Get the polydata (3D surface) from the contour filter
        poly_data = contour_filter.GetOutput()

        return poly_data

    def closeEvent(self, event):
        # Clean up VTK widgets
        for widget in [self.vtkWidget_axial, self.vtkWidget_coronal, self.vtkWidget_sagittal, self.model_vtkWidget]:
            if widget:
                widget.Finalize()
                del widget

        # Call the base class close method
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Replace the synthetic image with reading an actual .mha file
    # Specify the path to your .mha file
    image_path = r'D:\pelvis-source\001.mha'
    # Read the image using SimpleITK
    sitk_image = sitk.ReadImage(image_path)

    # Create and show the CTViewer
    viewer = CTViewer(sitk_image)
    viewer.show()

    sys.exit(app.exec_())

