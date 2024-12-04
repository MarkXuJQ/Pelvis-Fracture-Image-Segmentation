import sys
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QSlider
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import SimpleITK as sitk
import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkImageSlice, vtkImageSliceMapper
from vtkmodules.vtkImagingCore import vtkImageReslice
from vtkmodules.vtkInteractionWidgets import vtkLineWidget2, vtkLineRepresentation
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2

class CTViewer(QWidget):
    def __init__(self, sitk_image, render_model=False):
        super().__init__()
        self.sitk_image = sitk_image
        self.render_model = render_model
        self.image_data = self.sitk_to_vtk_image(sitk_image)
        
        # Load the UI file
        uic.loadUi('system/ui/ct_viewer.ui', self)
        
        self.setup_sliders()
        self.setup_views()
        
        if self.render_model:
            self.setup_model_view()
            self.generate_and_display_model()

    def setup_views(self):
        # Initialize VTK widgets
        self.vtkWidget_axial = self.findChild(QVTKRenderWindowInteractor, 'vtkWidget_axial')
        self.vtkWidget_coronal = self.findChild(QVTKRenderWindowInteractor, 'vtkWidget_coronal')
        self.vtkWidget_sagittal = self.findChild(QVTKRenderWindowInteractor, 'vtkWidget_sagittal')
        
        # Setup each view
        self.setup_slice_view(self.vtkWidget_axial, 2)    # Axial
        self.setup_slice_view(self.vtkWidget_coronal, 1)  # Coronal
        self.setup_slice_view(self.vtkWidget_sagittal, 0) # Sagittal

    def setup_slice_view(self, vtk_widget, orientation):
        # Initialize the widget
        vtk_widget.Initialize()
        renderer = vtkRenderer()
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        
        # Create the reslice mapper
        reslice = vtkImageReslice()
        reslice.SetInputData(self.image_data)
        reslice.SetOutputDimensionality(2)
        reslice.SetResliceAxes(self.get_reslice_axes(orientation))
        reslice.SetInterpolationModeToLinear()
        
        # Create mapper and slice
        mapper = vtkImageSliceMapper()
        mapper.SetInputConnection(reslice.GetOutputPort())
        
        image_slice = vtkImageSlice()
        image_slice.SetMapper(mapper)
        
        # Set window/level for better visualization
        image_slice.GetProperty().SetColorWindow(2000)
        image_slice.GetProperty().SetColorLevel(0)
        
        renderer.AddViewProp(image_slice)
        renderer.ResetCamera()
        
        # Store the objects we need to update later
        vtk_widget.reslice = reslice
        vtk_widget.image_slice = image_slice
        
        # Add line widget (reslice cursor)
        line_widget = vtkLineWidget2()
        line_widget.SetInteractor(vtk_widget.GetRenderWindow().GetInteractor())
        line_representation = line_widget.GetRepresentation()
        line_representation.PlaceWidget(self.image_data.GetBounds())
        line_widget.On()
        
        # Set the line widget to update the reslice plane
        line_widget.AddObserver("InteractionEvent", lambda obj, event: self.update_reslice_plane(obj, reslice, orientation))
        
        # Store the line widget
        vtk_widget.line_widget = line_widget
        
        return vtk_widget

    def get_reslice_axes(self, orientation):
        # Create appropriate axes matrix for each view
        axes = vtk.vtkMatrix4x4()
        if orientation == 0:    # Sagittal
            axes.DeepCopy((
                0, 0, -1, 0,
                0, 1, 0, 0,
                1, 0, 0, 0,
                0, 0, 0, 1))
        elif orientation == 1:  # Coronal
            axes.DeepCopy((
                1, 0, 0, 0,
                0, 0, -1, 0,
                0, 1, 0, 0,
                0, 0, 0, 1))
        else:                   # Axial (orientation == 2)
            axes.DeepCopy((
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1))
        return axes

    def setup_sliders(self):
        # Get sliders from UI
        self.axial_slider = self.findChild(QSlider, 'axial_slider')
        self.coronal_slider = self.findChild(QSlider, 'coronal_slider')
        self.sagittal_slider = self.findChild(QSlider, 'sagittal_slider')
        
        # Get image dimensions
        dims = self.image_data.GetDimensions()
        
        # Set ranges for sliders
        self.axial_slider.setRange(0, dims[2]-1)
        self.coronal_slider.setRange(0, dims[1]-1)
        self.sagittal_slider.setRange(0, dims[0]-1)
        
        # Set initial values to middle of the image
        self.axial_slider.setValue(dims[2]//2)
        self.coronal_slider.setValue(dims[1]//2)
        self.sagittal_slider.setValue(dims[0]//2)
        
        # Connect sliders to update functions
        self.axial_slider.valueChanged.connect(lambda val: self.update_slice_position(2, val))
        self.coronal_slider.valueChanged.connect(lambda val: self.update_slice_position(1, val))
        self.sagittal_slider.valueChanged.connect(lambda val: self.update_slice_position(0, val))
        
    def update_reslice_plane(self, obj, reslice, orientation):
        rep = obj.GetRepresentation()
        point1, point2 = [0, 0, 0], [0, 0, 0]
        rep.GetPoint1WorldPosition(point1)
        rep.GetPoint2WorldPosition(point2)
        
        normal = [point2[i] - point1[i] for i in range(3)]
        vtk.vtkMath.Normalize(normal)
        
        origin = point1
        
        matrix = vtk.vtkMatrix4x4()
        matrix.DeepCopy((
            normal[0], normal[1], normal[2], origin[0],
            normal[1], -normal[0], 0, origin[1],
            normal[2], 0, normal[0], origin[2],
            0, 0, 0, 1
        ))
        
        reslice.SetResliceAxes(matrix)
        reslice.Update()

    def update_slice_position(self, orientation, position):
        widgets = {
            0: self.vtkWidget_sagittal,
            1: self.vtkWidget_coronal,
            2: self.vtkWidget_axial
        }
        
        widget = widgets[orientation]
        center = list(self.image_data.GetCenter())
        spacing = self.image_data.GetSpacing()
        
        # Update the slice position
        center[orientation] = position * spacing[orientation]
        widget.reslice.SetResliceAxesOrigin(center)
        
        # Update the line widget position
        rep = widget.line_widget.GetRepresentation()
        point1, point2 = [0, 0, 0], [0, 0, 0]
        rep.GetPoint1WorldPosition(point1)
        rep.GetPoint2WorldPosition(point2)
        point1[orientation] = center[orientation]
        point2[orientation] = center[orientation]
        rep.SetPoint1WorldPosition(point1)
        rep.SetPoint2WorldPosition(point2)
        
        # Render the updated view
        widget.GetRenderWindow().Render()

    def sitk_to_vtk_image(self, sitk_image):
        # Convert SimpleITK image to VTK image
        size = sitk_image.GetSize()
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        
        # Convert SITK image to NumPy array
        image_array = sitk.GetArrayFromImage(sitk_image)
        
        # Transpose the array to match VTK's ordering (x,y,z)
        image_array = np.transpose(image_array.astype(np.float32), (2, 1, 0))
        
        # Create VTK image data
        vtk_image = vtkImageData()
        vtk_image.SetDimensions(size)
        vtk_image.SetSpacing(spacing)
        vtk_image.SetOrigin(origin)
        vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)
        
        # Copy data from numpy array to vtk image
        vtk_array = vtk_image.GetPointData().GetScalars()
        vtk_array_mem = vtk_np.vtk_to_numpy(vtk_array)
        vtk_array_mem[:] = image_array.flatten(order='F')
        
        return vtk_image

    def closeEvent(self, event):
        # Clean up
        for widget in [self.vtkWidget_axial, self.vtkWidget_coronal, self.vtkWidget_sagittal]:
            if widget:
                widget.Finalize()
        super().closeEvent(event)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    image_path = '/Users/markxu/Downloads/001.mha'  # Path to medical image file
    sitk_image = sitk.ReadImage(image_path)  # Read the image using SimpleITK
    viewer = CTViewer(sitk_image)  # Create an instance of the viewer with the image
    viewer.show()  # Display the viewer window
    sys.exit(app.exec_())  # Execute the application