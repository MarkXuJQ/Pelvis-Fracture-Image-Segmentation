import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt, QFileInfo
from PyQt5.QtGui import QPixmap, QRegion
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox, QFileDialog, \
    QLabel
import sys

from requests import Session

import sys
from sqlalchemy import Column, String
import pyodbc
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTableWidget, \
    QTableWidgetItem, QDialog, QMessageBox
from PyQt5.QtWidgets import QComboBox, QLineEdit, QFormLayout, QErrorMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 数据库连接设置
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=LAPTOP-5NGQ4BFB;DATABASE=PropertyManagementSystem;Trusted_Connection=yes"
conn = pyodbc.connect(connection_string)
engine = create_engine("mssql+pyodbc://", creator=lambda: conn)

# 测试 SQLAlchemy 连接
try:
    with engine.connect() as connection:
        print("SQLAlchemy 连接成功")
except Exception as e:
    print("SQLAlchemy 连接失败:", e)

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()


# 病人表
class Patient(Base):
    __tablename__ = 'patients'
    patient_id = Column(String(20), primary_key=True)
    patient_name = Column(String(50))
    patient_password = Column(String(20))
    phone = Column(String(11))


class AddPatientDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("新增病人信息")

        # 创建表单布局
        form_layout = QFormLayout()

        # 创建输入框和标签
        self.patient_id_input = QLineEdit(self)
        self.name_input = QLineEdit(self)
        self.age_input = QLineEdit(self)
        self.gender_input = QComboBox(self)
        self.gender_input.addItems(["男", "女"])  # 男、女
        self.disease_input = QLineEdit(self)
        self.contact_input = QLineEdit(self)

        # 将控件加入表单布局
        form_layout.addRow("病人ID:", self.patient_id_input)
        form_layout.addRow("姓名:", self.name_input)
        form_layout.addRow("年龄:", self.age_input)
        form_layout.addRow("性别:", self.gender_input)
        form_layout.addRow("疾病:", self.disease_input)
        form_layout.addRow("联系方式:", self.contact_input)

        # 创建提交按钮
        submit_button = QPushButton("提交", self)
        submit_button.clicked.connect(self.submit)

        # 将按钮加入布局
        form_layout.addWidget(submit_button)

        # 设置布局
        self.setLayout(form_layout)

    def submit(self):
        """提交数据并关闭对话框"""
        patient_id = self.patient_id_input.text().strip()
        name = self.name_input.text().strip()
        age = self.age_input.text().strip()
        gender = self.gender_input.currentText()
        disease = self.disease_input.text().strip()
        contact = self.contact_input.text().strip()

        if not patient_id or not name or not age or not disease or not contact:
            QMessageBox.warning(self, "错误", "所有字段都是必填项！")
            return

        try:
            age = int(age)  # 将年龄转换为整数
        except ValueError:
            QMessageBox.warning(self, "错误", "年龄必须为整数！")
            return

        # 构建病人信息字典
        new_patient = {
            "patient_id": patient_id,
            "name": name,
            "age": age,
            "gender": gender,
            "disease": disease,
            "contact": contact
        }

        # 插入新病人信息到数据库
        self.insert_new_patient(new_patient)

        # 关闭对话框
        self.accept()

    def insert_new_patient(self, patient_info):
        """将新病人信息插入数据库"""
        session = Session()
        try:
            new_patient = Patient(
                patient_id=patient_info["patient_id"],
                name=patient_info["name"],
                age=patient_info["age"],
                gender=patient_info["gender"],
                disease=patient_info["disease"],
                contact=patient_info["contact"]
            )
            session.add(new_patient)
            session.commit()
        except Exception as e:
            print(f"Error inserting new patient: {e}")
            session.rollback()  # 如果出错回滚事务
        finally:
            session.close()

class PatientManageWindow(QMainWindow):
    def __init__(self,table,list):
        super(PatientManageWindow, self).__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(current_dir, "ui", "patient_message.ui")

        # 调试输出
        print(f"Current directory: {current_dir}")
        print(f"UI file path: {ui_file}")
        print(f"UI file exists: {os.path.exists(ui_file)}")

        uic.loadUi(ui_file, self)
        with open('ui/button_style.qss', 'r', encoding='utf-8') as f:
            self.setStyleSheet(f.read())
        self.setWindowTitle("Patient Management")
        self.setGeometry(0, 0, 1800, 900)
        self.tableWidget=table
        self.listWidget = list
        #self.initUI()
        self.avatarLabel = self.findChild(QLabel, 'avatarLabel')
        pixmap = QPixmap("../image/plan/头像测试.jpg")
        self.avatarLabel.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
        # 创建一个圆形的QRegion作为mask
        # mask = QRegion(0, 0, 100, 100, QRegion.Ellipse)
        #
        # # 设置QLabel的图像和mask
        # self.avatarLabel.setPixmap(pixmap)
        # self.avatarLabel.setMask(mask)  # 应用圆形mask
        # # 设置QLabel的固定大小为100x100，避免布局问题
        # self.avatarLabel.setFixedSize(100, 100)

    def initUI(self):
        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Set up layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Add buttons to the layout
        self.add_patient_button = QPushButton("Add Patient")
        self.load_patient_button = QPushButton("Load Patient")
        self.view_patient_button = QPushButton("View Patient Data")

        layout.addWidget(self.add_patient_button)
        layout.addWidget(self.load_patient_button)
        layout.addWidget(self.view_patient_button)

        # Connect buttons to methods (placeholders for now)
        self.add_patient_button.clicked.connect(self.add_patient)
        self.load_patient_button.clicked.connect(self.load_patient)
        self.view_patient_button.clicked.connect(self.view_patient)

    def add_patient(self):
        print('ok')
        dialog = AddPatientDialog()
        if dialog.exec_() == QDialog.Accepted:
            # 如果提交成功，刷新表格
            self.refresh_table()

    def load_patient(self):
        # Open file dialog to select image
        options = QFileDialog.Options()
        file_types = "All Files (*);;DICOM Files (*.dcm);;NIfTI Files (*.nii *.nii.gz);;NRRD Files (*.nrrd);;MetaImage Files (*.mha *.mhd)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Patient File", "", file_types, options=options)
        if file_path:
            file_name = QFileInfo(file_path).fileName()  # 获取文件名
            self.listWidget.clearSelection()
            self.listWidget.addItem(file_name)  # 将文件名添加到 listWidget
            self.listWidget.setFocus()  # 设置焦点到 listWidget
            # 选择当前新添加的项
            item = self.listWidget.item(self.listWidget.count() - 1)
            item.setSelected(True)  # 选中新添加的项

    def view_patient(self):
        # Placeholder for viewing patient data
        print("View Patient Data button clicked")

    def on_delete_patient_info(self):
        # 检查是否焦点在 QListWidget 上
        # if self.listWidget.hasFocus():
        #     # 获取选中的项
        #     selected_items = self.listWidget.selectedItems()
        #     for item in selected_items:
        #         # 删除选中的项
        #         self.listWidget.takeItem(self.listWidget.row(item))
        #         # 可以在这里执行同步删除数据库中的记录
        #         print(f"Deleted item from QListWidget: {item.text()}")
        #
        # # 检查是否焦点在 QTableWidget 上
        # elif self.tableWidget.hasFocus():
        # 获取表格的行数
        rows = self.tableWidget.rowCount()
        for row in range(rows - 1, -1, -1):  # 从最后一行开始遍历，避免删除时改变行索引
            # 获取复选框所在的单元格项
            item = self.tableWidget.item(row, 0)  # 假设复选框在第1列
            if item and item.checkState() == Qt.Checked:  # 检查复选框是否被选中
                # 删除选中的行
                self.tableWidget.removeRow(row)
                # 在这里可以同步删除数据库中的记录
                # db.delete_patient(item_id)  # 假设有一个方法来删除数据
                print(f"Deleted row {row} from QTableWidget")
    def delete_selected_file(self):
        # Get the selected item in the listWidget
        selected_item = self.listWidget.currentItem()
        if selected_item:  # If an item is selected
            self.listWidget.takeItem(self.listWidget.row(selected_item))  # Remove the selected item from the list
            print(f"Deleted file: {selected_item.text()}")  # Print the name of the deleted file
        else:
            print("No file selected for deletion.")  # If no item is selected, print a message


def main():
    app = QApplication(sys.argv)
    main_window = PatientManageWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
