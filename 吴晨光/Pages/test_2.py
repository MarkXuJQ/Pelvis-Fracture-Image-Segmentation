import sys
import vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import nibabel as nib
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Medical Image Viewer")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)


        self.layout = QVBoxLayout(self.central_widget)

        # Create QVTKRenderWindowInteractors for each orientation
        self.interactors = {
            "Axial": QVTKRenderWindowInteractor(parent=self.central_widget),
            "Sagittal": QVTKRenderWindowInteractor(parent=self.central_widget),
            "Coronal": QVTKRenderWindowInteractor(parent=self.central_widget)
        }

        for interactor in self.interactors.values():
            self.layout.addWidget(interactor)

        # Load NIFTI file
        self.filename = "D:/graduate_design/project/abdomen_dataset/abdomen/imagesTr/img0007.nii.gz"
        self.load_nifti()

        # Initialize slices
        self.slice_positions = {
            "Axial": self.dims[2] // 2,
            "Sagittal": self.dims[0] // 2,
            "Coronal": self.dims[1] // 2
        }
        self.update_slices()

    def load_nifti(self):
        nifti_image = nib.load(self.filename)
        self.image_data = nifti_image.get_fdata()
        self.dims = self.image_data.shape

    def update_slices(self):
        for orientation, interactor in self.interactors.items():
            slice_position = self.slice_positions[orientation]
            slice_data = np.take(self.image_data, slice_position, axis=self.get_axis(orientation)).T

            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(slice_data.shape[1], slice_data.shape[0], 1)
            vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

            vtk_array = vtk_image.GetPointData().GetScalars()
            vtk_array.SetArray(slice_data.flatten(), slice_data.size, 1)

            viewer = vtk.vtkImageViewer2()
            viewer.SetInputData(vtk_image)
            viewer.SetRenderWindow(interactor.GetRenderWindow())
            viewer.SetSliceOrientation(self.get_vtk_orientation(orientation))
            viewer.SetSlice(slice_position)
            viewer.Render()

        for interactor in self.interactors.values():
            interactor.Render()

    def get_axis(self, orientation):
        axes = {"Axial": 2, "Sagittal": 0, "Coronal": 1}
        return axes[orientation]

    def get_vtk_orientation(self, orientation):
        orientations = {"Axial": vtk.vtkImageViewer2.SLICE_ORIENTATION_XY,
                        "Sagittal": vtk.vtkImageViewer2.SLICE_ORIENTATION_XZ,
                        "Coronal": vtk.vtkImageViewer2.SLICE_ORIENTATION_YZ}
        return orientations[orientation]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
