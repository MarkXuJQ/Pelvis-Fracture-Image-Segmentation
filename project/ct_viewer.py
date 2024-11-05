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
        # For testing, create a synthetic image instead
        image = vtkImageData()
        image.SetDimensions(100, 100, 100)
        image.AllocateScalars(VTK_UNSIGNED_CHAR, 1)

        # Initialize with gradient values
        dims = image.GetDimensions()
        for z in range(dims[2]):
            for y in range(dims[1]):
                for x in range(dims[0]):
                    pixel = (x + y + z) % 256
                    image.SetScalarComponentFromDouble(x, y, z, 0, pixel)

        return image

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
        reslice_representation.GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(self.reslice_cursor)
        reslice_representation.GetResliceCursorActor().GetCursorAlgorithm().SetReslicePlaneNormal(orientation)

        # Configure the reslice representation
        scalar_range = self.image_data.GetScalarRange()
        window = scalar_range[1] - scalar_range[0]
        level = (scalar_range[1] + scalar_range[0]) / 2
        reslice_representation.SetWindowLevel(window, level)
        reslice_representation.GetReslice().SetInputData(self.image_data)
        reslice_representation.GetResliceCursor().SetImage(self.image_data)

        # Enable the widget
        reslice_widget.SetEnabled(1)
        reslice_widget.On()

        # Store the widgets for synchronization
        self.reslice_widgets.append(reslice_widget)
        self.reslice_representations.append(reslice_representation)

        # Initialize the interactor
        render_window_interactor.Initialize()

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

# You can add your main application code here to test the CTViewer class
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create a dummy SimpleITK image for testing
    sitk_image = sitk.Image(100, 100, 100, sitk.sitkFloat32)
    sitk_image.SetSpacing([1.0, 1.0, 1.0])
    sitk_image.SetOrigin([0.0, 0.0, 0.0])

    # Fill the image with gradient values
    for z in range(100):
        for y in range(100):
            for x in range(100):
                sitk_image[x, y, z] = (x + y + z) % 256

    # Create and show the CTViewer
    viewer = CTViewer(sitk_image)
    viewer.show()

    sys.exit(app.exec_())
