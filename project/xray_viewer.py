# xray_viewer.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class XRayViewer(QWidget):
    def __init__(self, image_array):
        super().__init__()
        self.image_array = image_array
        self.initUI()

    def initUI(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(self.image_array, cmap='gray')
        self.ax.axis('off')

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def close(self):
        # Clean up VTK widget
        self.vtkWidget.Finalize()
        del self.vtkWidget

        self.close()  # Call the base class close method if necessary
