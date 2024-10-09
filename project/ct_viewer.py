import vtk
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QLabel, QSizePolicy
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
import SimpleITK as sitk
import vtk.util.numpy_support as vtk_np

class CTViewer(QWidget):
    def __init__(self, sitk_image, render_model=False):
        super().__init__()
        self.sitk_image = sitk_image
        self.render_model = render_model
        self.crosshair_visible = False

        # Convert SimpleITK image to NumPy array
        self.image_array = sitk.GetArrayFromImage(self.sitk_image)

        # Get dimensions
        self.dimensions = self.image_array.shape  # (Depth, Height, Width)

        # Initialize slice indices
        self.slice_indices = [
            self.dimensions[0] // 2,  # Axial
            self.dimensions[1] // 2,  # Coronal
            self.dimensions[2] // 2   # Sagittal
        ]

        # Crosshair line sources
        self.crosshair_lines_axial = []
        self.crosshair_lines_coronal = []
        self.crosshair_lines_sagittal = []
        
        self.initUI()
        self.initModelWindow()

    def initUI(self):
        # Create VTK widgets and renderers for each view
        self.vtkWidget_axial = self.create_vtk_widget()
        self.vtkWidget_coronal = self.create_vtk_widget()
        self.vtkWidget_sagittal = self.create_vtk_widget()

        # Create sliders for each view
        self.slider_axial = self.create_slider(0, self.dimensions[0] - 1, self.slice_indices[0], self.update_axial_view)
        self.slider_coronal = self.create_slider(1, self.dimensions[1] - 1, self.slice_indices[1], self.update_coronal_view)
        self.slider_sagittal = self.create_slider(2, self.dimensions[2] - 1, self.slice_indices[2], self.update_sagittal_view)

        # Set layouts
        self.setLayouts()

        # Initialize views
        self.update_views()

    def create_vtk_widget(self):
        vtk_widget = QVTKRenderWindowInteractor(self)
        renderer = vtk.vtkRenderer()
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        return vtk_widget, renderer

    def create_slider(self, dim, max_value, initial_value, update_function):
        slider = QSlider()
        slider.setOrientation(1)  # Horizontal
        slider.setMinimum(0)
        slider.setMaximum(max_value)
        slider.setValue(initial_value)
        slider.valueChanged.connect(update_function)
        return slider

    def setLayouts(self):
        axial_layout = self.create_view_layout('Axial View', self.vtkWidget_axial[0], self.slider_axial)
        coronal_layout = self.create_view_layout('Coronal View', self.vtkWidget_coronal[0], self.slider_coronal)
        sagittal_layout = self.create_view_layout('Sagittal View', self.vtkWidget_sagittal[0], self.slider_sagittal)

        # Initialize the model window widget in the top-right corner
        self.model_vtkWidget = QVTKRenderWindowInteractor(self)
        self.model_renderer = vtk.vtkRenderer()
        self.model_vtkWidget.GetRenderWindow().AddRenderer(self.model_renderer)

        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel('3D Model'))
        model_layout.addWidget(self.model_vtkWidget)

        # Create a grid layout
        grid_layout = QGridLayout()
        grid_layout.addLayout(axial_layout, 0, 0)   # Top Left
        grid_layout.addLayout(model_layout, 0, 1)   # Top Right
        grid_layout.addLayout(coronal_layout, 1, 0) # Bottom Left
        grid_layout.addLayout(sagittal_layout, 1, 1) # Bottom Right

        # Set stretch factors
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        # Set the grid layout as the main layout
        self.setLayout(grid_layout)

        # Initialize VTK widgets
        self.initialize_vtk_widgets()

    def create_view_layout(self, title, vtk_widget, slider):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(title))
        layout.addWidget(vtk_widget)
        layout.addWidget(slider)
        return layout

    def initialize_vtk_widgets(self):
        self.model_vtkWidget.Initialize()
        self.vtkWidget_axial[0].Initialize()
        self.vtkWidget_coronal[0].Initialize()
        self.vtkWidget_sagittal[0].Initialize()

        self.model_vtkWidget.Start()
        self.vtkWidget_axial[0].Start()
        self.vtkWidget_coronal[0].Start()
        self.vtkWidget_sagittal[0].Start()

    def update_views(self):
        self.update_axial_view(self.slice_indices[0])
        self.update_coronal_view(self.slice_indices[1])
        self.update_sagittal_view(self.slice_indices[2])
        self.update_crosshairs()

    def update_axial_view(self, value):
        self.slice_indices[0] = value
        slice_data = self.image_array[value, :, :]
        vtk_image = self.numpy_to_vtk_image(slice_data)
        self.display_image(vtk_image, self.vtkWidget_axial[1])
        self.update_crosshairs()

    def update_coronal_view(self, value):
        self.slice_indices[1] = value
        slice_data = self.image_array[:, value, :]
        vtk_image = self.numpy_to_vtk_image(slice_data)
        self.display_image(vtk_image, self.vtkWidget_coronal[1])
        self.update_crosshairs()

    def update_sagittal_view(self, value):
        self.slice_indices[2] = value
        slice_data = self.image_array[:, :, value]
        vtk_image = self.numpy_to_vtk_image(slice_data)
        self.display_image(vtk_image, self.vtkWidget_sagittal[1])
        self.update_crosshairs()

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

    def display_image(self, vtk_image, vtk_widget):
        # Remove previous items
        vtk_widget.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveAllViewProps()

        # Create image actor
        image_actor = vtk.vtkImageActor()
        image_actor.SetInputData(vtk_image)

        # Add actor to the renderer
        vtk_widget.GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(image_actor)
        vtk_widget.GetRenderWindow().GetRenderers().GetFirstRenderer().ResetCamera()
        vtk_widget.GetRenderWindow().Render()
    def create_crosshair(self):
        # Check if crosshairs already exist
        if not self.crosshair_visible:
            # Create crosshairs in each view
            self.crosshair_lines_axial = self.create_crosshair_lines(self.vtkWidget_axial[1])
            self.crosshair_lines_coronal = self.create_crosshair_lines(self.vtkWidget_coronal[1])
            self.crosshair_lines_sagittal = self.create_crosshair_lines(self.vtkWidget_sagittal[1])

            self.crosshair_visible = True  # Set the visibility flag to true
        else:
            # Hide crosshairs by removing them from the renderers
            for line in self.crosshair_lines_axial + self.crosshair_lines_coronal + self.crosshair_lines_sagittal:
                line[0].RemoveAllViewProps()  # Remove lines from renderers
                line[1].RemoveAllViewProps()

            self.crosshair_visible = False  # Set the visibility flag to false

        self.update_views()  # Update views to reflect the changes


    def create_crosshair_lines(self, renderer):
        # Create two lines for the crosshair
        line1 = vtk.vtkLineSource()
        line2 = vtk.vtkLineSource()

        # Create mappers and actors for both lines
        mapper1 = vtk.vtkPolyDataMapper()
        mapper1.SetInputConnection(line1.GetOutputPort())
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper1)
        actor1.GetProperty().SetColor(1, 0, 0)  # Red color
        renderer.AddActor(actor1)

        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputConnection(line2.GetOutputPort())
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        actor2.GetProperty().SetColor(1, 0, 0)  # Red color
        renderer.AddActor(actor2)

        return (line1, actor1), (line2, actor2)


    def update_crosshairs(self):
        # Check if crosshairs are visible
        if not self.crosshair_visible:
            return

        dims = self.dimensions  # (Depth, Height, Width)
        x = self.slice_indices[2]  # Sagittal index (Width)
        y = self.slice_indices[1]  # Coronal index (Height)

        # Update crosshairs in axial view
        self.crosshair_lines_axial[0][0].SetPoint1(x, 0, 0)
        self.crosshair_lines_axial[0][0].SetPoint2(x, dims[1], 0)
        self.crosshair_lines_axial[1][0].SetPoint1(0, y, 0)
        self.crosshair_lines_axial[1][0].SetPoint2(dims[2], y, 0)

        # Update crosshairs in coronal view
        z = self.slice_indices[0]  # Axial index (Depth)
        self.crosshair_lines_coronal[0][0].SetPoint1(x, 0, z)
        self.crosshair_lines_coronal[0][0].SetPoint2(x, dims[1], z)
        self.crosshair_lines_coronal[1][0].SetPoint1(0, y, z)
        self.crosshair_lines_coronal[1][0].SetPoint2(dims[2], y, z)

        # Update crosshairs in sagittal view
        self.crosshair_lines_sagittal[0][0].SetPoint1(0, y, z)
        self.crosshair_lines_sagittal[0][0].SetPoint2(dims[1], y, z)
        self.crosshair_lines_sagittal[1][0].SetPoint1(x, 0, z)
        self.crosshair_lines_sagittal[1][0].SetPoint2(x, dims[0], z)

        # Render updates
        for vtk_widget in [self.vtkWidget_axial[1], self.vtkWidget_coronal[1], self.vtkWidget_sagittal[1]]:
            vtk_widget.GetRenderWindow().Render()


    def initModelWindow(self):
        if self.render_model:
            self.model_renderer.ResetCamera()  # Adjust camera if needed
            # Additional model rendering logic can go here


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

        # # Create axes actor
        # axes = vtk.vtkAxesActor()
        # axes.SetTotalLength(1.0, 1.0, 1.0)
        # axes.SetShaftTypeToLine()
        # axes.SetAxisLabels(True)
        # axes.SetCylinderRadius(0.02)

        # # Customize axes labels
        # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 0, 0)  # Red X
        # axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 1, 0)  # Green Y
        # axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 1)  # Blue Z

        # Add the cube actor and axes to the renderer
        self.model_renderer.AddActor(cube_actor)
        # self.model_renderer.AddActor(axes)

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
            self.vtkWidget_axial.Finalize()
            del self.vtkWidget_axial

        if hasattr(self, 'vtkWidget_coronal'):
            self.vtkWidget_coronal.Finalize()
            del self.vtkWidget_coronal

        if hasattr(self, 'vtkWidget_sagittal'):
            self.vtkWidget_sagittal.Finalize()
            del self.vtkWidget_sagittal

        if hasattr(self, 'model_vtkWidget'):
            self.model_vtkWidget.Finalize()
            del self.model_vtkWidget

        # Call the base class close method
        super().close()