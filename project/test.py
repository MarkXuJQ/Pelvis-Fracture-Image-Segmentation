import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from vtkmodules.vtkInteractionWidgets import (
    vtkResliceCursorWidget,
    vtkResliceCursorLineRepresentation,
    vtkResliceCursor
)
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkImageActor
from vtkmodules.util.vtkConstants import VTK_UNSIGNED_CHAR
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2

class TestApp(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)

        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Create a renderer and add it to the render window
        self.renderer = vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # Create an image data object and initialize it with some values
        image = vtkImageData()
        image.SetDimensions(100, 100, 100)
        image.AllocateScalars(VTK_UNSIGNED_CHAR, 1)

        # Fill the image with gradient values
        dims = image.GetDimensions()
        for z in range(dims[2]):
            for y in range(dims[1]):
                for x in range(dims[0]):
                    pixel = (x + y + z) % 256
                    image.SetScalarComponentFromDouble(x, y, z, 0, pixel)

        # Create the reslice cursor
        reslice_cursor = vtkResliceCursor()
        reslice_cursor.SetCenter(image.GetCenter())
        reslice_cursor.SetImage(image)

        # Create the reslice cursor representation and widget
        reslice_representation = vtkResliceCursorLineRepresentation()
        reslice_widget = vtkResliceCursorWidget()
        reslice_widget.SetInteractor(self.iren)
        reslice_widget.SetRepresentation(reslice_representation)

        # Set the reslice cursor via the cursor algorithm
        reslice_representation.GetResliceCursorActor().GetCursorAlgorithm().SetResliceCursor(reslice_cursor)

        # Configure the reslice representation
        reslice_representation.SetWindowLevel(255, 127)
        reslice_representation.GetReslice().SetInputData(image)
        reslice_representation.GetResliceCursor().SetImage(image)

        # Add an image actor to display the image
        image_actor = vtkImageActor()
        image_actor.GetMapper().SetInputData(image)
        self.renderer.AddActor(image_actor)

        # Reset the camera to include the image
        self.renderer.ResetCamera()

        # Enable the widget
        reslice_widget.SetEnabled(1)
        reslice_widget.On()

        # Initialize and start the interactor
        self.iren.Initialize()
        # self.iren.Start()  # Do not call Start() here in PyQt applications

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestApp()
    window.show()
    sys.exit(app.exec_())
