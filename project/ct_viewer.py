# ct_viewer.py

import vtk
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
import SimpleITK as sitk
import vtk.util.numpy_support as vtk_np

class CTViewer(QWidget):
    def __init__(self, sitk_image):
        super().__init__()
        self.sitk_image = sitk_image

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

    def initUI(self):
        # Create VTK widgets and renderers for each view
        self.vtkWidget_axial = QVTKRenderWindowInteractor(self)
        self.vtkWidget_coronal = QVTKRenderWindowInteractor(self)
        self.vtkWidget_sagittal = QVTKRenderWindowInteractor(self)

        self.renderer_axial = vtk.vtkRenderer()
        self.renderer_coronal = vtk.vtkRenderer()
        self.renderer_sagittal = vtk.vtkRenderer()

        self.vtkWidget_axial.GetRenderWindow().AddRenderer(self.renderer_axial)
        self.vtkWidget_coronal.GetRenderWindow().AddRenderer(self.renderer_coronal)
        self.vtkWidget_sagittal.GetRenderWindow().AddRenderer(self.renderer_sagittal)

        # Create crosshairs for each view
        self.crosshair_lines_axial = self.create_crosshairs(self.renderer_axial)
        self.crosshair_lines_coronal = self.create_crosshairs(self.renderer_coronal)
        self.crosshair_lines_sagittal = self.create_crosshairs(self.renderer_sagittal)

        # Initialize the views
        self.update_views()

        # Create sliders for each view
        self.slider_axial = QSlider()
        self.slider_axial.setOrientation(1)  # Horizontal
        self.slider_axial.setMinimum(0)
        self.slider_axial.setMaximum(self.dimensions[0] - 1)
        self.slider_axial.setValue(self.slice_indices[0])
        self.slider_axial.valueChanged.connect(self.update_axial_view)

        self.slider_coronal = QSlider()
        self.slider_coronal.setOrientation(1)
        self.slider_coronal.setMinimum(0)
        self.slider_coronal.setMaximum(self.dimensions[1] - 1)
        self.slider_coronal.setValue(self.slice_indices[1])
        self.slider_coronal.valueChanged.connect(self.update_coronal_view)

        self.slider_sagittal = QSlider()
        self.slider_sagittal.setOrientation(1)
        self.slider_sagittal.setMinimum(0)
        self.slider_sagittal.setMaximum(self.dimensions[2] - 1)
        self.slider_sagittal.setValue(self.slice_indices[2])
        self.slider_sagittal.valueChanged.connect(self.update_sagittal_view)

        # Layouts
        layout = QVBoxLayout()
        views_layout = QHBoxLayout()

        # Axial view layout
        axial_layout = QVBoxLayout()
        axial_layout.addWidget(QLabel('Axial View'))
        axial_layout.addWidget(self.vtkWidget_axial)
        axial_layout.addWidget(self.slider_axial)

        # Coronal view layout
        coronal_layout = QVBoxLayout()
        coronal_layout.addWidget(QLabel('Coronal View'))
        coronal_layout.addWidget(self.vtkWidget_coronal)
        coronal_layout.addWidget(self.slider_coronal)

        # Sagittal view layout
        sagittal_layout = QVBoxLayout()
        sagittal_layout.addWidget(QLabel('Sagittal View'))
        sagittal_layout.addWidget(self.vtkWidget_sagittal)
        sagittal_layout.addWidget(self.slider_sagittal)

        # Add layouts to views_layout
        views_layout.addLayout(axial_layout)
        views_layout.addLayout(coronal_layout)
        views_layout.addLayout(sagittal_layout)

        # Add views_layout to main layout
        layout.addLayout(views_layout)
        self.setLayout(layout)

        # Initialize VTK widgets
        self.vtkWidget_axial.Initialize()
        self.vtkWidget_coronal.Initialize()
        self.vtkWidget_sagittal.Initialize()

        self.vtkWidget_axial.Start()
        self.vtkWidget_coronal.Start()
        self.vtkWidget_sagittal.Start()

    def create_crosshairs(self, renderer):
        # Create two lines for the crosshair
        line1 = vtk.vtkLineSource()
        line2 = vtk.vtkLineSource()

        # Create mappers and actors
        mapper1 = vtk.vtkPolyDataMapper()
        mapper1.SetInputConnection(line1.GetOutputPort())
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper1)
        actor1.GetProperty().SetColor(1, 0, 0)  # Red color

        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputConnection(line2.GetOutputPort())
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        actor2.GetProperty().SetColor(1, 0, 0)  # Red color

        # Add actors to the renderer
        renderer.AddActor(actor1)
        renderer.AddActor(actor2)

        return (line1, line2)  # Return line sources to update positions later

    def update_views(self):
        self.update_axial_view(self.slice_indices[0])
        self.update_coronal_view(self.slice_indices[1])
        self.update_sagittal_view(self.slice_indices[2])
        self.update_crosshairs()

    def update_axial_view(self, value):
        self.slice_indices[0] = value
        slice_data = self.image_array[value, :, :]
        vtk_image = self.numpy_to_vtk_image(slice_data)
        self.display_image(vtk_image, self.renderer_axial, self.vtkWidget_axial)
        self.update_crosshairs()

    def update_coronal_view(self, value):
        self.slice_indices[1] = value
        slice_data = self.image_array[:, value, :]
        vtk_image = self.numpy_to_vtk_image(slice_data)
        self.display_image(vtk_image, self.renderer_coronal, self.vtkWidget_coronal)
        self.update_crosshairs()

    def update_sagittal_view(self, value):
        self.slice_indices[2] = value
        slice_data = self.image_array[:, :, value]
        vtk_image = self.numpy_to_vtk_image(slice_data)
        self.display_image(vtk_image, self.renderer_sagittal, self.vtkWidget_sagittal)
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

    def display_image(self, vtk_image, renderer, vtk_widget):
        # Remove previous items
        renderer.RemoveAllViewProps()

        # Create image actor
        image_actor = vtk.vtkImageActor()
        image_actor.SetInputData(vtk_image)

        # Add actor to the renderer
        renderer.AddActor(image_actor)
        renderer.ResetCamera()
        vtk_widget.GetRenderWindow().Render()

    def update_crosshairs(self):
        dims = self.dimensions  # (Depth, Height, Width)

        # Update crosshairs in axial view
        x = self.slice_indices[2]  # Sagittal index (Width)
        y = self.slice_indices[1]  # Coronal index (Height)
        # Vertical line (X position)
        self.crosshair_lines_axial[0].SetPoint1(x, 0, 0)
        self.crosshair_lines_axial[0].SetPoint2(x, dims[1], 0)
        # Horizontal line (Y position)
        self.crosshair_lines_axial[1].SetPoint1(0, y, 0)
        self.crosshair_lines_axial[1].SetPoint2(dims[2], y, 0)

        # Update crosshairs in coronal view
        x = self.slice_indices[2]  # Sagittal index (Width)
        z = self.slice_indices[0]  # Axial index (Depth)
        # Vertical line (X position)
        self.crosshair_lines_coronal[0].SetPoint1(x, 0, 0)
        self.crosshair_lines_coronal[0].SetPoint2(x, dims[0], 0)
        # Horizontal line (Z position)
        self.crosshair_lines_coronal[1].SetPoint1(0, z, 0)
        self.crosshair_lines_coronal[1].SetPoint2(dims[2], z, 0)

        # Update crosshairs in sagittal view
        y = self.slice_indices[1]  # Coronal index (Height)
        z = self.slice_indices[0]  # Axial index (Depth)
        # Vertical line (Y position)
        self.crosshair_lines_sagittal[0].SetPoint1(y, 0, 0)
        self.crosshair_lines_sagittal[0].SetPoint2(y, dims[0], 0)
        # Horizontal line (Z position)
        self.crosshair_lines_sagittal[1].SetPoint1(0, z, 0)
        self.crosshair_lines_sagittal[1].SetPoint2(dims[1], z, 0)

        # Render the views to update the crosshairs
        self.vtkWidget_axial.GetRenderWindow().Render()
        self.vtkWidget_coronal.GetRenderWindow().Render()
        self.vtkWidget_sagittal.GetRenderWindow().Render()
