import vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Medical Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        self.layout = QVBoxLayout(self.centralWidget)

        self.vtkWidget = QVTKRenderWindowInteractor(self.centralWidget)
        self.layout.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.loadDICOMButton = QPushButton("Load DICOM", self.centralWidget)
        self.loadDICOMButton.clicked.connect(self.load_dicom)
        self.layout.addWidget(self.loadDICOMButton)

        self.loadNIFTIButton = QPushButton("Load NIfTI", self.centralWidget)
        self.loadNIFTIButton.clicked.connect(self.load_nifti)
        self.layout.addWidget(self.loadNIFTIButton)

        self.show()

    def load_dicom(self):
        # Add your DICOM loading code here
        pass

    def load_nifti(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "", "NIfTI Files (*.nii.gz)")
        if filename:
            reader = vtk.vtkNIFTIImageReader()
            reader.SetFileName(filename)
            reader.Update()

            self.imageActor = vtk.vtkImageActor()
            self.imageActor.SetInputData(reader.GetOutput())

            self.ren.AddActor(self.imageActor)
            self.ren.ResetCamera()

            self.iren.Initialize()
            self.iren.Start()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
