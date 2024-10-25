from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import  QMainWindow, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHBoxLayout, QSizePolicy,QHeaderView,QMessageBox,QListWidgetItem
import os
import shutil
import sys
from PyQt5.QtCore import QObject, pyqtSignal,QSize, Qt

class Patient_Image(QMainWindow):
    goBackSignal = pyqtSignal()#返回界面一的信号
    navigateSignal_visual = pyqtSignal() #跳转至界面三的信号
    file_label_path_signal = pyqtSignal(str) #传label文件名到main
    file_image_path_signal = pyqtSignal(str) #传image文件名到main
    flash_image = pyqtSignal()#刷新路径
    flash_label = pyqtSignal()#刷新路径
    def __init__(self, patient_name):
        super().__init__()

        # 使用loadUi加载UI文件
        loadUi('吴晨光/15-相关材料/毕设相关源码/ui/patient_image.ui', self)

        # 设置 progress_1 被选中
        self.progress_1.setChecked(True)

        # 为 ListWidget_1 添加信号和槽连接
        self.listWidget_1.itemClicked.connect(self.on_listWidget_1_itemClicked)

        # 为listwidget_2添加信号和槽链接
        self.listWidget_2.itemClicked.connect(self.on_listWidget_2_itemClicked)

        # 为 model_select_1 到 model_select_4 添加信号和槽连接
        # self.model_select_1.toggled.connect(self.on_model_select_toggled)
        self.model_select_2.toggled.connect(self.on_model_select_toggled)
        self.model_select_3.toggled.connect(self.on_model_select_toggled)
        self.model_select_4.toggled.connect(self.on_model_select_toggled)

        # 为 split 按钮添加点击事件连接
        self.split.clicked.connect(self.on_split_clicked)

        # 加载病人文件夹内容到 listWidget_1 中
        self.loadPatientFiles(patient_name)

        self.loadPatientLabels(patient_name)

        #为back按钮添加点击事件
        self.back.clicked.connect(self.goBackToPatientManage)      

        # 为visual添加带点击事件连接
        self.visualize.clicked.connect(self.navigateTovisual_result)  


     # 加载病人文件夹内容到 listWidget_1 中
    def loadPatientFiles(self, patient_name):
        # 获取病人文件夹路径
        patient_folder_path = os.path.join('patients', patient_name)
        # print("list——1：",patient_folder_path)
        # 检查文件夹是否存在
        if os.path.exists(patient_folder_path) and os.path.isdir(patient_folder_path):
            # 清空 listWidget_1 中的内容
            self.listWidget_1.clear()

            # 获取文件夹中的文件和子文件夹
            files_and_folders = os.listdir(patient_folder_path)

            # 将文件和子文件夹名称添加到 listWidget_1 中
            for item in files_and_folders:
                file_path = os.path.join(patient_folder_path, item)
                if os.path.isfile(file_path) and \
                    (item.endswith('.nii.gz') or item.endswith('.dcm')) and \
                    'label' not in item.lower():
                        self.listWidget_1.addItem(item)

    #listwidget_2显示label文件夹 
    def loadPatientLabels(self, patient_name):
            # 获取病人文件夹路径
        patient_folder_path = os.path.join('patients', patient_name)
        # print(" patient_folder_path    ",patient_folder_path)

        # 检查文件夹是否存在
        if os.path.exists(patient_folder_path) and os.path.isdir(patient_folder_path):
            # 清空列表视图中的内容
            self.listWidget_2.clear()
            # 获取label子文件夹的路径
            label_folder_path = os.path.join(patient_folder_path, 'label')
            print(" label_folder_path    ",label_folder_path)

            # 检查label子文件夹是否存在
            if os.path.exists(label_folder_path) and os.path.isdir(label_folder_path):
                # 获取label子文件夹中的所有文件名
                label_files = os.listdir(label_folder_path)

                # 将文件名添加到列表视图中
                for file_name in label_files:
                    # item = QListWidgetItem(file_name)
                    self.listWidget_2.addItem(file_name)
        self.listWidget_2.update()

    # 当 ListWidget_1 中的某一行被选中时，progress_2 被选中
    def on_listWidget_1_itemClicked(self, item):
        self.progress_2.setChecked(True)
        self.flash_image.emit()
        # 获取选中行的文件名
        selected_file_name = item.text()
        # 发射自定义信号，发送选中行的文件名
        self.file_image_path_signal.emit(selected_file_name)

    # 当 ListWidget_2 中的某一行被选中时，progress_2 被选中
    def on_listWidget_2_itemClicked(self, item):
        self.progress_2.setChecked(True)
        self.flash_label.emit()
         # 获取选中行的文件名
        selected_file_name = item.text()
        # 发射自定义信号，发送选中行的文件名
        self.file_label_path_signal.emit(selected_file_name)

    # 当 model_select_1 到 model_select_4 中的某个被选中时，progress_3 被选中
    def on_model_select_toggled(self):
        
        if self.model_select_2.isChecked() or \
           self.model_select_3.isChecked() or self.model_select_4.isChecked():
            self.progress_3.setChecked(True)

    # 当 split 按钮被点击时，progress_4 被选中
    def on_split_clicked(self):
        
        self.progress_4.setChecked(True)

    # 当用户点击返回按钮时发出信号
    def goBackToPatientManage(self):
        # self.progress_1.setChecked(False)
        self.progress_2.setChecked(False)
        self.progress_3.setChecked(False)
        self.progress_4.setChecked(False)
        self.progress_5.setChecked(False)
        self.model_select_1.setChecked(False)
        self.model_select_2.setChecked(False)
        self.model_select_3.setChecked(False)
        self.model_select_4.setChecked(False)
        
        self.goBackSignal.emit()

    # 当用户点击visualize时发出信号（跳转至页面三）
    def navigateTovisual_result(self):
        self.navigateSignal_visual.emit()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = Patient_Image()
#     window.show()
#     sys.exit(app.exec_())