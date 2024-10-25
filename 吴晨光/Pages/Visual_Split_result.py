import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QWidget
from PyQt5.uic import loadUi

from PyQt5.QtWidgets import *

from PyQt5 import QtCore, QtGui, QtWidgets

from DicomVis_ok import DicomVis

class Visual_Wndow(QMainWindow):
    def __init__(self):
        super(Visual_Wndow, self).__init__()
        # 加载UI文件
        loadUi("吴晨光/15-相关材料/毕设相关源码/ui/visual_split_result_1.0.ui", self)
        # 设置主窗口的布局为垂直布局
        self.centralwidget.setLayout(QVBoxLayout())



        self.dicomVisWidget = DicomVis()
        self.horizontalLayout.insertWidget(1, self.dicomVisWidget,6)

        # 设置布局管理器的缩放策略为Expanding，以确保所有部件都会自适应窗口大小变化
        self.centralwidget.layout().setContentsMargins(0, 0, 0, 0)
        self.centralwidget.layout().setSpacing(0)
        self.centralwidget.layout().setStretch(0, 1)
        # 设置各个控件的大小策略为Expanding，以便它们能够自适应窗口大小
        self.setWidgetsSizePolicy()
        self.DicomDir.clicked.connect(self.on_AddDicomBtn_clicked)
        self.niiFile.clicked.connect(self.on_AddNiiBtn_clicked)
        self.selectNii.clicked.connect(self.on_Add2DNIIBtn_clicked)
       # load_Nii_from_path

    def on_AddDicomBtn_clicked(self):
        dicompath = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")

        if dicompath == "":
            print("\n取消选择")
            return
        self.dicomVisWidget.load_dicom_from_path(dicompath)

    def on_Add2DNIIBtn_clicked(self):
        segdialog = QtWidgets.QFileDialog()
        segdialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        segdialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        file, _ = segdialog.getOpenFileName(segdialog, "Open file", "", "files (*.nii *nii.gz" + ")")

        if file == "":
            print("\n取消选择")
            return

        self.dicomVisWidget.load_Nii_from_path(file)

    def on_AddNiiBtn_clicked(self):
        segdialog = QtWidgets.QFileDialog()
        segdialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        segdialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        file, _ = segdialog.getOpenFileName(segdialog, "Open file", "", "files (*.nii *nii.gz" + ")")

        if file == "":
            print("\n取消选择")
            return

        self.dicomVisWidget.GetImageDataFromPath(file)

    def setWidgetsSizePolicy(self):
        for widget in self.findChildren(QWidget):
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
    def close(self):
        self.dicomVisWidget.close()

    def closeEvent(self, event):
        self.close()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Visual_Wndow()
    mainWindow.show()
    sys.exit(app.exec_())
