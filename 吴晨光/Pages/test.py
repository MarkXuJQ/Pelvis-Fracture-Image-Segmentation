import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QWidget
from PyQt5.uic import loadUi

from PyQt5.QtWidgets import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal,QSize, Qt

from DicomVis_ok import DicomVis

class Visual_Wndow(QMainWindow):
    goBackSignal_info = pyqtSignal()
    def __init__(self):
        super(Visual_Wndow, self).__init__()
        # 加载UI文件
        loadUi("吴晨光/15-相关材料/毕设相关源码/ui/visual_split_result_1.0.ui", self)
        # 设置主窗口的布局为垂直布局
        self.centralwidget.setLayout(QVBoxLayout())

        # 设置进度条全亮
        self.progress_1.setChecked(True)
        self.progress_2.setChecked(True)
        self.progress_3.setChecked(True)
        self.progress_4.setChecked(True)
        self.progress_5.setChecked(True)

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
       
        self.back.clicked.connect(self.goBackToPatientInfo)
    # load_Nii_from_path
    def on_AddDicomBtn_clicked(self):
        dicompath = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "")

        if dicompath == "":
            print("\n取消选择")
            return
        self.dicomVisWidget.load_dicom_from_path(dicompath)

    # 界面二来的执行流
    def load_nii_from_widget_2(self, image_path):
        segdialog = QtWidgets.QFileDialog()
        segdialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        segdialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        # if image_path == "":
        #     print("\n取消选择")
        #     return
        self.dicomVisWidget.load_Nii_from_path(image_path)
    
    def on_Add2DNIIBtn_clicked(self):
        segdialog = QtWidgets.QFileDialog()
        segdialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        segdialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        file, _ = segdialog.getOpenFileName(segdialog, "Open file", "", "files (*.nii *nii.gz" + ")")

        if file == "":
            print("\n取消选择")
            return

        self.dicomVisWidget.load_Nii_from_path(file)

    # 界面二来的执行流（label）
    def load_label_from_widget_2(self, label_path):
        segdialog = QtWidgets.QFileDialog()
        segdialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        segdialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        self.dicomVisWidget.GetImageDataFromPath(label_path)


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

    # 当用户点击back时发出信号
    def goBackToPatientInfo(self):
        self.goBackSignal_info.emit()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     mainWindow = Visual_Wndow()
#     mainWindow.show()
#     sys.exit(app.exec_())
