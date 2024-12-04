import sys
from PyQt5.QtWidgets import QApplication
from Patient_Manage import Patient_Manage
from Patient_Image import Patient_Image
from test import Visual_Wndow
import os
class MainApplication:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.patient_manage_page = Patient_Manage()
        self.patient_info_page = None
        self.visual_result_page = Visual_Wndow()

        # 连接界面一的信号到页面跳转的槽函数
        self.patient_manage_page.navigateSignal.connect(self.navigateToPatient_Image)

        # 链接界面三的信号到页面跳转的槽函数
        self.visual_result_page.goBackSignal_info.connect(self.goBackToPatientInfo)

        # 链接界面二传过来的image文件名的槽函数
        # self.patient_info_page.file_image_path_signal.connect(self.calculate_file_path)
        # 显示界面一
        self.patient_manage_page.show()
        sys.exit(self.app.exec_())

    def navigateToPatient_Image(self):
        # 获取当前选中行的病人姓名
        global current_row
        current_row = self.patient_manage_page.tableWidget.currentRow()
        # cur_row = current_row
        patient_name_item = self.patient_manage_page.tableWidget.item(current_row, 0)
        global file_image_path
        file_image_path = os.path.join(file_image_path, patient_name_item.text())
        file_image_path = file_image_path.replace('\\', '/')
        global file_label_path
        file_label_path = os.path.join(file_label_path, patient_name_item.text(),"label")
        file_label_path = file_label_path.replace('\\', '/')
        # print(patient_name_item, current_row)
        if patient_name_item is not None:
            patient_name = patient_name_item.text()
            # print(patient_name+'\n')

            # 如果界面二未被实例化，则实例化并加载病人文件
            if self.patient_info_page is None:
                self.patient_info_page = Patient_Image(patient_name)
                self.patient_info_page.loadPatientFiles(patient_name)
                self.patient_info_page.goBackSignal.connect(self.goBackToPatientManage)
                # 链接界面二的信号到页面跳转的槽函数
                self.patient_info_page.navigateSignal_visual.connect(self.navigateTovisual_result)
                # 链接界面二的刷新函数
                self.patient_info_page.flash_image.connect(self.flash_image)
                self.patient_info_page.flash_label.connect(self.flash_label)
                # 链接界面二传过来的image文件名的槽函数
                self.patient_info_page.file_image_path_signal.connect(self.calculate_image_path)
                # 链接界面二传过来的label文件名的槽函数
                self.patient_info_page.file_label_path_signal.connect(self.calculate_label_path)
                self.patient_info_page.show()
            else:
                # 如果界面二已经被实例化，则直接加载病人文件
                self.patient_info_page.loadPatientFiles(patient_name)
                self.patient_info_page.show()
            # print("label_file:",file_label_path)

            # 隐藏界面一
            self.patient_manage_page.hide()

    # 返回界面1
    def goBackToPatientManage(self):
        # 隐藏界面二
        self.patient_info_page.hide()
        self.patient_info_page = None
        # 显示界面一
        self.patient_manage_page.show()
        global file_image_path
        global file_label_path
        file_image_path = "E:/pytorch/吴晨光毕设源码/15-相关材料/毕设相关源码/patients"
        file_label_path = "E:/pytorch/吴晨光毕设源码/15-相关材料/毕设相关源码/patients"

    # 界面二通过visual到界面三
    def navigateTovisual_result(self):
        self.patient_manage_page.hide()
        self.visual_result_page.show()
        global file_image_path
        global file_label_path
        # file_image_path = os.path.dirname(file_image_path)
        # file_label_path = os.path.dirname(file_label_path)
        # print(file_image_path)
        self.visual_result_page.load_nii_from_widget_2(file_image_path)
        self.visual_result_page.load_label_from_widget_2(file_label_path)



    # 界面三返回界面二
    def goBackToPatientInfo(self):
        # cur_row = current_row
        # 获取当前选中行的病人姓名
        # current_row = self.patient_manage_page.tableWidget.currentRow()
        patient_name_item = self.patient_manage_page.tableWidget.item(current_row, 0)
        # print(patient_name_item.text(), current_row)
        if patient_name_item is not None:
            patient_name = patient_name_item.text()
            # print(patient_name+'\n')
            # 如果界面二已经被实例化，则直接加载病人文件
            self.patient_info_page.loadPatientFiles(patient_name)
            self.patient_info_page.show()
            # 隐藏界面三
            self.visual_result_page.hide()
        global file_image_path
        global file_label_path
        file_image_path = "E:/pytorch/吴晨光毕设源码/15-相关材料/毕设相关源码/patients"
        file_label_path = "E:/pytorch/吴晨光毕设源码/15-相关材料/毕设相关源码/patients"

    # 计算路径
    def calculate_image_path(self, file_name):
        global file_image_path
        # global file_label_path
        file_image_path = os.path.join(file_image_path, file_name)
        # file_label_path = os.path.join(file_label_path, file_name)
        file_image_path = file_image_path.replace('\\', '/')
        # file_label_path = file_label_path.replace('\\', '/')
        print("file_path_image:", file_image_path)
        # print("file_path_label:", file_label_path)


    # 计算路径
    def calculate_label_path(self, file_name):
        global file_label_path
        # global file_label_path
        file_label_path = os.path.join(file_label_path, file_name)
        # file_label_path = os.path.join(file_label_path, file_name)
        file_label_path = file_label_path.replace('\\', '/')
        # file_label_path = file_label_path.replace('\\', '/')
        print("file_label_path:", file_label_path)

    # 刷新路径
    def flash_image(self):
        global file_image_path

        file_image_path = "E:/pytorch/吴晨光毕设源码/15-相关材料/毕设相关源码/patients"
        file_image_path = os.path.join(file_image_path, self.patient_manage_page.tableWidget.item(current_row, 0).text())

    # 刷新路径
    def flash_label(self):
        global file_label_path

        file_label_path = "E:/pytorch/吴晨光毕设源码/15-相关材料/毕设相关源码/patients"
        file_label_path = os.path.join(file_label_path, self.patient_manage_page.tableWidget.item(current_row, 0).text(),"label")
    
if __name__ == "__main__":
    current_row = -1
    file_image_path = "E:/pytorch/吴晨光毕设源码/15-相关材料/毕设相关源码/patients"
    file_label_path = "E:/pytorch/吴晨光毕设源码/15-相关材料/毕设相关源码/patients"
    main_app = MainApplication()
