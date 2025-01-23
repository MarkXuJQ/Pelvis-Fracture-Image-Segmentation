from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QStackedWidget, QWidget, QSplitter


#from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class DoctorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # 加载 .ui 文件
        uic.loadUi("ui/doctor_window.ui", self)
        page1 = self.findChild(QWidget, "rightSplitter")  # 获取页面1对象
        self.stackedWidget.setCurrentWidget(page1)  # 设置显示页面1

        # 设置默认显示页面1 (ctViewerPage)
        #self.stackedWidget.setCurrentIndex(0)  # 显示页面1
        print(self.stackedWidget)  # 检查是否加载成功
        #self.show()
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = DoctorUI()
    sys.exit(app.exec_())
