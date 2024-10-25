from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUi
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import SimpleITK as sitk
import numpy as np
import vtk.util.numpy_support as vtk_np

class CTViewer(QWidget):
    def __init__(self, sitk_image, render_model=False):
        super().__init__()

        # Load the UI
        loadUi('project/ui/ct_viewer.ui', self)

        self.sitk_image = sitk_image
        self.render_model = render_model

        # Replace placeholders with QVTKRenderWindowInteractor
        self.vtkWidget_axial = QVTKRenderWindowInteractor(self)
        self.vtkWidget_coronal = QVTKRenderWindowInteractor(self)
        self.vtkWidget_sagittal = QVTKRenderWindowInteractor(self)
        self.model_vtkWidget = QVTKRenderWindowInteractor(self)

        # Set them in the correct layout positions
        axial_placeholder = self.findChild(QWidget, 'vtkWidget_axial')
        coronal_placeholder = self.findChild(QWidget, 'vtkWidget_coronal')
        sagittal_placeholder = self.findChild(QWidget, 'vtkWidget_sagittal')
        model_placeholder = self.findChild(QWidget, 'model_vtkWidget')

        # Replace the placeholders with actual VTK widgets
        self.layout().replaceWidget(axial_placeholder, self.vtkWidget_axial)
        self.layout().replaceWidget(coronal_placeholder, self.vtkWidget_coronal)
        self.layout().replaceWidget(sagittal_placeholder, self.vtkWidget_sagittal)
        self.layout().replaceWidget(model_placeholder, self.model_vtkWidget)

        axial_placeholder.deleteLater()
        coronal_placeholder.deleteLater()
        sagittal_placeholder.deleteLater()
        model_placeholder.deleteLater()

        # Initialize VTK widgets and set image interactor style
        self.init_vtk_widget(self.vtkWidget_axial)
        self.init_vtk_widget(self.vtkWidget_coronal)
        self.init_vtk_widget(self.vtkWidget_sagittal)
        self.model_vtkWidget.Initialize()  # For model widget, rotation might be acceptable

        # Set up renderers
        self.axial_renderer = vtk.vtkRenderer()
        self.vtkWidget_axial.GetRenderWindow().AddRenderer(self.axial_renderer)

        self.coronal_renderer = vtk.vtkRenderer()
        self.vtkWidget_coronal.GetRenderWindow().AddRenderer(self.coronal_renderer)

        self.sagittal_renderer = vtk.vtkRenderer()
        self.vtkWidget_sagittal.GetRenderWindow().AddRenderer(self.sagittal_renderer)

        self.model_renderer = vtk.vtkRenderer()
        self.model_vtkWidget.GetRenderWindow().AddRenderer(self.model_renderer)

        # Convert SimpleITK image to NumPy array
        self.image_array = sitk.GetArrayFromImage(self.sitk_image)

        # Get dimensions
        self.dimensions = self.image_array.shape  # (Depth, Height, Width)

        # Initialize slice indices and slider ranges
        self.slice_indices = [
            self.dimensions[0] // 2,  # Axial
            self.dimensions[1] // 2,  # Coronal
            self.dimensions[2] // 2   # Sagittal
        ]

        self.slider_axial.setRange(0, self.dimensions[0] - 1)
        self.slider_coronal.setRange(0, self.dimensions[1] - 1)
        self.slider_sagittal.setRange(0, self.dimensions[2] - 1)

        # Connect sliders to update functions
        self.slider_axial.valueChanged.connect(self.update_axial_view)
        self.slider_coronal.valueChanged.connect(self.update_coronal_view)
        self.slider_sagittal.valueChanged.connect(self.update_sagittal_view)
        
        
# Set slider initial values to the middle
        self.slider_axial.setValue(self.slice_indices[0])
        self.slider_coronal.setValue(self.slice_indices[1])
        self.slider_sagittal.setValue(self.slice_indices[2])

        # Initialize views
        self.update_views()
        self.initModelWindow()

    def init_vtk_widget(self, vtk_widget):
        vtk_widget.Initialize()
        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleImage())  # Lock rotation

    def update_views(self):
        self.update_axial_view(self.slice_indices[0])
        self.update_coronal_view(self.slice_indices[1])
        self.update_sagittal_view(self.slice_indices[2])

    def update_axial_view(self, value):
        self.slice_indices[0] = value
        slice_data = self.image_array[value, :, :]
        vtk_image = self.numpy_to_vtk_image(slice_data)
        self.display_image(vtk_image, self.axial_renderer)

    def update_coronal_view(self, value):
        self.slice_indices[1] = value
        slice_data = self.image_array[:, value, :]
        vtk_image = self.numpy_to_vtk_image(slice_data)
        self.display_image(vtk_image, self.coronal_renderer)

    def update_sagittal_view(self, value):
        self.slice_indices[2] = value
        slice_data = self.image_array[:, :, value]
        vtk_image = self.numpy_to_vtk_image(slice_data)
        self.display_image(vtk_image, self.sagittal_renderer)

    def numpy_to_vtk_image(self, image):
        # Convert NumPy array to VTK image
        image = np.flipud(image)
        vtk_data_array = vtk_np.numpy_to_vtk(
            num_array=image.ravel(),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(image.shape[1], image.shape[0], 1)
        vtk_image.GetPointData().SetScalars(vtk_data_array)
        return vtk_image

    def display_image(self, vtk_image, renderer):
        if renderer is None:
            print("Renderer is None, cannot render the image.")
            return

        # Remove previous items
        renderer.RemoveAllViewProps()

        # Create image actor
        image_actor = vtk.vtkImageActor()
        image_actor.SetInputData(vtk_image)

        # Add actor to the renderer
        renderer.AddActor(image_actor)

        if not hasattr(renderer, 'camera_initialized') or not renderer.camera_initialized:
            camera = renderer.GetActiveCamera()
            camera.ParallelProjectionOn()  # Use parallel projection for a flat view
            renderer.ResetCamera()
            renderer.camera_initialized = True

        renderer.GetRenderWindow().Render()

    def initModelWindow(self):
        if self.render_model:
            self.generate_and_display_model()
        else:
            self.display_placeholder_cube()

    def generate_and_display_model(self):
        # Generate the 3D model
        poly_data = self.generate_3d_model()
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # Add actor to the renderer
        self.model_renderer.AddActor(actor)
        self.model_renderer.SetBackground(0.68, 0.85, 0.9)  # Light blue background
        self.model_renderer.ResetCamera()
        self.model_vtkWidget.GetRenderWindow().Render()

    def display_placeholder_cube(self):
        # Create a cube source
        cube_source = vtk.vtkCubeSource()
        cube_source.Update()

        # Create a mapper for the cube
        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputConnection(cube_source.GetOutputPort())

        # Create a cube actor
        cube_actor = vtk.vtkActor()
        cube_actor.SetMapper(cube_mapper)

        # Set the cube to wireframe
        cube_actor.GetProperty().SetRepresentationToWireframe()
        cube_actor.GetProperty().SetColor(1, 1, 1)  # White color

        # Add the cube actor to the renderer
        self.model_renderer.AddActor(cube_actor)

        # Set background color to light blue
        self.model_renderer.SetBackground(0.68, 0.85, 0.9)  # Light blue background

        self.model_renderer.ResetCamera()
        self.model_vtkWidget.GetRenderWindow().Render()

    def generate_3d_model(self):
        # Convert the SimpleITK image to a VTK image
        sitk_image = sitk.Cast(self.sitk_image, sitk.sitkFloat32)
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(sitk_image.GetSize())

        spacing = sitk_image.GetSpacing()
        vtk_image.SetSpacing(spacing)

        origin = sitk_image.GetOrigin()
        vtk_image.SetOrigin(origin)

        # Convert SimpleITK image to numpy array
        image_array = sitk.GetArrayFromImage(sitk_image)

        # Apply threshold to isolate certain structures (e.g., bones)
        lower_threshold = 300  # example threshold for bones in Hounsfield units
        upper_threshold = 3000  # example upper limit for bones
        thresholded_image = np.where((image_array >= lower_threshold) & (image_array <= upper_threshold), image_array, 0)

        vtk_data_array = vtk_np.numpy_to_vtk(
        num_array=thresholded_image.ravel(),
        deep=True,
        array_type=vtk.VTK_FLOAT
        )

        vtk_image.GetPointData().SetScalars(vtk_data_array)

        # Extract a 3D surface from the thresholded volume using Marching Cubes algorithm
        contour_filter = vtk.vtkMarchingCubes()
        contour_filter.SetInputData(vtk_image)
        contour_filter.SetValue(0, lower_threshold)  # Isosurface value (use the lower threshold)
        contour_filter.Update()

        # Get the polydata (3D surface) from the contour filter
        poly_data = contour_filter.GetOutput()

        return poly_data

    def close(self):
        # Clean up VTK widgets
        if hasattr(self, 'vtkWidget_axial'):
            self.vtkWidget_axial[0].Finalize()
            del self.vtkWidget_axial
            
        if hasattr(self, 'vtkWidget_coronal'):
            self.vtkWidget_coronal[0].Finalize()
            del self.vtkWidget_coronal

        if hasattr(self, 'vtkWidget_sagittal'):
            self.vtkWidget_sagittal[0].Finalize()
            del self.vtkWidget_sagittal

        if hasattr(self, 'model_vtkWidget'):
            self.model_vtkWidget.Finalize()
            del self.model_vtkWidget

        # Call the base class close method
        super().close()