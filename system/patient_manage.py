import os
from datetime import datetime

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt, QFileInfo, QDate, QTimer
from PyQt5.QtGui import QPixmap, QRegion, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox, QFileDialog, \
    QLabel, QListWidgetItem, QHBoxLayout, QHeaderView
import sys

from requests import Session
from system.database.db_manager import patients, fracturehistories
import sys
from sqlalchemy import Column, String
import pyodbc
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTableWidget, \
    QTableWidgetItem, QDialog, QMessageBox
from PyQt5.QtWidgets import QComboBox, QLineEdit, QFormLayout, QErrorMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import pymysql
from pymysql import Error
from system.database.db_config import db_config

# 创建数据库连接
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
Session = sessionmaker(bind=engine)
session = Session()


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
        self.gender_input.addItems(['male', 'female', 'other'])  # 男、女
        self.id_number_input = QLineEdit(self)
        # self.disease_input = QLineEdit(self)
        self.phone_input = QLineEdit(self)
        self.contactPerson_input = QLineEdit(self)
        self.contactPhone_input = QLineEdit(self)

        # 创建出生年月日的 QComboBox
        self.birth_date_input = QComboBox(self)
        # 填充年份、月份、日期（这里示例填充2020年1月1日至2023年12月31日的日期）
        for year in range(1900, 2024):  # 填充年份
            self.birth_date_input.addItem(str(year))
        self.birth_month_input = QComboBox(self)
        for month in range(1, 13):  # 填充月份
            self.birth_month_input.addItem(str(month))

        self.birth_day_input = QComboBox(self)
        for day in range(1, 32):  # 填充日期
            self.birth_day_input.addItem(str(day))

        # 将控件加入表单布局
        form_layout.addRow("病人ID:", self.patient_id_input)
        form_layout.addRow("姓名:", self.name_input)
        form_layout.addRow("年龄:", self.age_input)
        form_layout.addRow("性别:", self.gender_input)
        form_layout.addRow("联系电话:", self.phone_input)
        # form_layout.addRow("疾病:", self.disease_input)
        form_layout.addRow("紧急联系人:", self.contactPerson_input)
        form_layout.addRow("紧急联系人电话:", self.contactPhone_input)
        form_layout.addRow("身份证号码:", self.id_number_input)

        # 出生日期部分使用 QComboBox 来选择
        form_layout.addRow("出生年份:", self.birth_date_input)
        form_layout.addRow("出生月份:", self.birth_month_input)
        form_layout.addRow("出生日期:", self.birth_day_input)

        # 创建提交按钮
        submit_button = QPushButton("提交", self)
        submit_button.clicked.connect(self.submit)

        # 将按钮加入布局
        form_layout.addWidget(submit_button)

        # 设置布局
        self.setLayout(form_layout)

    def submit(self):
        """提交数据并关闭对话框"""
        # 获取用户输入的数据
        patient_id = self.patient_id_input.text().strip()
        name = self.name_input.text().strip()
        age = self.age_input.text().strip()
        gender = self.gender_input.currentText()
        # disease = self.disease_input.text().strip()
        phone = self.phone_input.text().strip()
        contact_person = self.contactPerson_input.text().strip()  # 紧急联系人
        contact_phone = self.contactPhone_input.text().strip()  # 紧急联系人电话
        id_number = self.id_number_input.text().strip()  # 身份证号码

        # 获取出生年月日
        birth_year = self.birth_date_input.currentText()  # 选择的出生年份
        birth_month = self.birth_month_input.currentText()  # 选择的出生月份
        birth_day = self.birth_day_input.currentText()  # 选择的出生日期

        # 验证所有必填字段是否已填
        if not patient_id or not name or not age or not contact_person or not phone or not contact_phone or not id_number:
            QMessageBox.warning(self, "错误", "所有字段都是必填项！")
            return

        # 验证年龄是否为整数
        try:
            age = int(age)  # 将年龄转换为整数
        except ValueError:
            QMessageBox.warning(self, "错误", "年龄必须为整数！")
            return

        # 验证出生年月日是否完整
        if not birth_year or not birth_month or not birth_day:
            QMessageBox.warning(self, "错误", "出生年月日必须完整！")
            return
            # 合成完整的出生日期
        try:
            birth_date = QDate(int(birth_year), int(birth_month), int(birth_day))
            birth_date_str = birth_date.toString("yyyy-MM-dd")  # 转换为数据库可接受的日期格式
        except ValueError:
            QMessageBox.warning(self, "错误", "无效的出生日期！")
            return
        # 构建病人信息字典
        new_patient = {
            "patient_id": patient_id,
            "patient_name": name,
            "age": age,
            "gender": gender,
            "phone_number": phone,
            "contact_person": contact_person,
            "contact_phone": contact_phone,
            "id_card": id_number,
            "date_of_birth": birth_date_str,  # 使用合成的出生日期
            "password_hash": 1234
        }

        # 插入新病人信息到数据库
        self.insert_new_patient(new_patient)

        # 关闭对话框
        self.accept()

    def insert_new_patient(self, patient_info):
        """将新病人信息插入数据库"""
        session = Session()
        try:
            # 创建新的病人对象，插入数据库
            new_patient = patients(
                patient_id=patient_info["patient_id"],
                patient_name=patient_info["patient_name"],
                age=patient_info["age"],
                gender=patient_info["gender"],
                phone_number=patient_info["phone_number"],
                contact_person=patient_info["contact_person"],
                contact_phone=patient_info["contact_phone"],  # 紧急联系人电话
                id_card=patient_info["id_card"],
                date_of_birth=patient_info["date_of_birth"],
                password_hash=patient_info["password_hash"]
            )
            session.add(new_patient)
            session.commit()  # 提交事务
        except Exception as e:
            print(f"Error inserting new patient: {e}")
            session.rollback()  # 如果出错回滚事务
        finally:
            session.close()  # 关闭数据库会话


class PatientManageWindow(QMainWindow):
    def __init__(self, table, list, is_from_open_patient):
        super(PatientManageWindow, self).__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(current_dir, "ui", "patient_message.ui")

        # 调试输出
        print(f"Current directory: {current_dir}")
        print(f"UI file path: {ui_file}")
        print(f"UI file exists: {os.path.exists(ui_file)}")

        uic.loadUi(ui_file, self)

        self.setWindowTitle("Patient Management")
        self.setGeometry(0, 0, 1400, 800)
        self.tableWidget = table
        self.listWidget = list
        # self.initUI()
        self.nameLineEdit = self.findChild(QLineEdit, 'nameLineEdit')
        self.ageLineEdit = self.findChild(QLineEdit, 'ageLineEdit')
        self.genderLineEdit = self.findChild(QLineEdit, 'genderLineEdit')
        self.phoneLineEdit = self.findChild(QLineEdit, 'phoneLineEdit')
        self.idLineEdit = self.findChild(QLineEdit, 'idLineEdit')
        self.birthdayLineEdit = self.findChild(QLineEdit, 'birthdayLineEdit')
        self.contactPersonLineEdit = self.findChild(QLineEdit, 'contactPersonLineEdit')
        self.contactPhoneLineEdit = self.findChild(QLineEdit, 'contactPhoneLineEdit')

        self.medicalHistorytable = self.findChild(QTableWidget, 'medicalHistoryTable')

        self.backbutton = self.findChild(QPushButton, 'backButton')
        self.backbutton.clicked.connect(self.back)
        # 设置 QLabel 的大小
        self.avatarLabel_2.setFixedSize(250, 250)  # 设定合适的宽高
        self.avatarLabel_2 = self.findChild(QLabel, 'avatarLabel_2')
        pixmap = QPixmap("../image/plan/头像测试.jpg")
        self.avatarLabel_2.setPixmap(pixmap.scaled(self.avatarLabel_2.size(), Qt.IgnoreAspectRatio))
        self.is_from_open_patient = is_from_open_patient
        if self.is_from_open_patient:
            self.view_patient()

    def back(self):
        self.hide()  # 关闭当前窗口 (CTViewer)
        self.is_from_open_patient = False
        if self.checkbox_item:
            self.checkbox_item.setCheckState(Qt.Unchecked)  # 取消选中

    def add_patient(self):
        print('ok')
        dialog = AddPatientDialog()
        if dialog.exec_() == QDialog.Accepted:
            # 如果提交成功，刷新表格
            print("成功")
            self.add_log("新增病人成功！", "操作成功", "警告")
            self.refresh_table()

    def get_patient_by_choose(self, choose, choice1, choice2=None):
        """筛选病人信息"""
        session = Session()  # 获取数据库会话
        print(choose, choice1)
        try:
            if choose == "编号":
                patient_info = session.query(patients).filter(patients.patient_id == choice1).all()

            elif choose == "性别":
                print(6)
                print(type(choice1), choice1)  # 输出查询传入的值
                print(type(patients.gender), patients.gender)  # 输出数据库中读取的性别字段

                # 查询病人信息，根据病人性别进行筛选
                patient_info = session.query(patients).filter(patients.gender == choice1).all()
            elif choose == "编号和性别":
                if choice1.isdigit():  # 判断病人ID是否为数字
                    # 查询病人信息，根据病人ID和性别进行筛选
                    patient_info = session.query(patients).filter(patients.patient_id == choice1,
                                                                  patients.gender == choice2).all()
                else:
                    print("Invalid patient ID")
                    return []
            # 将查询结果转换为字典列表
            patient_list = []
            for patient in patient_info:
                patient_list.append({
                    "patient_id": patient.patient_id,
                    "patient_name": patient.patient_name,
                    "gender": patient.gender,
                })

            return patient_list

        except Exception as e:
            print(f"Error fetching patient info by level: {e}")
            return []

        finally:
            session.close()

    def fill_patient_table(self, patient_list):
        """将病人信息填充到QTableWidget"""
        self.tableWidget.setRowCount(len(patient_list))  # 设置表格行数
        print(patient_list)
        for row, patient in enumerate(patient_list):
            self.tableWidget.setItem(row, 1, QTableWidgetItem(str(patient["patient_id"])))
            self.tableWidget.setItem(row, 2, QTableWidgetItem(str(patient["patient_name"])))
            self.tableWidget.setItem(row, 3, QTableWidgetItem(str(patient["gender"])))

    def get_all_patient_info(self):
        session = Session()  # 获取数据库会话
        try:
            patient_info = session.query(patients).all()

            # 将病人信息转换为字典列表，以便后续在表格中显示
            patient_list = []
            for patient in patient_info:
                patient_list.append({
                    "patient_id": patient.patient_id,
                    "patient_name": patient.patient_name,
                    "age": patient.age,
                    "gender": patient.gender,
                    "id_card": patient.id_card,
                    "date_of_birth": patient.date_of_birth,
                    "phone_number": patient.phone_number,
                    "password_hash": patient.password_hash,
                    "contact_person": patient.contact_person,
                    "contact_phone": patient.contact_phone,

                })

            # 返回字典列表
            return patient_list

        except Exception as e:
            print(f"Error fetching housing info: {e}")
            return []

        finally:
            # 确保关闭会话
            session.close()

    def view_patient(self):
        """显示通用的个人信息"""
        selected_rows = []
        for row in range(self.tableWidget.rowCount()):
            checkbox_item = self.tableWidget.item(row, 0)  # 复选框在表格的第一列（索引为0）
            if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                selected_rows.append(row)
                self.checkbox_item = checkbox_item
        selected_row = selected_rows[0]  # 获取选中的行
        patient_id = self.tableWidget.item(selected_row, 1).text()
        print(patient_id)
        session = Session()  # 获取数据库会话
        print("?")
        # 从数据库获取病人的个人信息
        patient_info = session.query(patients).filter_by(patient_id=patient_id).first()
        print("?")
        # patient_fracture_info = session.query(FractureHistories).filter_by(patient_id=patient_id).all()
        print("?")
        print(patient_info)
        # 如果获取到了病人信息
        if patient_info:
            print(6)
            print("wer")
            # 假设你在 .ui 文件中已经设置了文本框，比如 nameLineEdit, ageLineEdit 等
            self.nameLineEdit.setText(patient_info.patient_name)  # 设置病人的名字
            self.ageLineEdit.setText(str(patient_info.age))  # 设置病人的年龄
            self.genderLineEdit.setText(patient_info.gender)  # 设置病人的性别
            self.phoneLineEdit.setText(str(patient_info.phone_number))  # 设置病人的电话
            self.idLineEdit.setText(patient_info.id_card)  # 设置病人的id number
            self.birthdayLineEdit.setText(str(patient_info.date_of_birth))  # 设置病人的生日
            self.contactPersonLineEdit.setText(patient_info.contact_person)  # 设置紧急联系人
            self.contactPhoneLineEdit.setText(patient_info.contact_phone)  # 设置紧急联系电话
            # 设置文本框为只读模式
            self.nameLineEdit.setReadOnly(True)
            self.ageLineEdit.setReadOnly(True)
            self.genderLineEdit.setReadOnly(True)
            self.phoneLineEdit.setReadOnly(True)
            self.idLineEdit.setReadOnly(True)
            self.birthdayLineEdit.setReadOnly(True)
            self.contactPersonLineEdit.setReadOnly(True)
            self.contactPhoneLineEdit.setReadOnly(True)

            # 调整每一列的宽度
            # self.medicalHistorytable.setColumnWidth(0, 150)  # 设置第一列宽度为100
            self.medicalHistorytable.setColumnWidth(0, 150)  # 设置第二列宽度为200
            # self.medicalHistorytable.setColumnWidth(1, 180)  # 设置第三列宽度为150
            self.medicalHistorytable.setColumnWidth(1, 180)  # 设置第三列宽度为150
            self.medicalHistorytable.setColumnWidth(2, 150)  # 设置第三列宽度为150
            self.medicalHistorytable.setColumnWidth(3, 200)  # 设置第三列宽度为150
            # 最后一列自动填充剩余宽度
            self.medicalHistorytable.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
            # 使用 relationship 直接访问病人的骨折历史
            patient_fracture_info = patient_info.fracturehistories
            if patient_fracture_info:
                # 按 fracture_date 降序排序（从最新到最晚）
                patient_fracture_info = sorted(patient_fracture_info, key=lambda fracture: fracture.fracture_date,
                                               reverse=True)

                # 填充病人的骨折历史到表格中
                self.medicalHistorytable.setRowCount(len(patient_fracture_info))  # 设置行数为骨折记录数
                for row, fracture in enumerate(patient_fracture_info):
                    self.medicalHistorytable.setItem(row, 0, QTableWidgetItem(str(fracture.fracture_date)))  # 骨折日期
                    # self.medicalHistorytable.setItem(row, 1, QTableWidgetItem(fracture.diagnosis_hospital))
                    self.medicalHistorytable.setItem(row, 1, QTableWidgetItem(str(fracture.fracture_location)))  # 骨折部位
                    # self.medicalHistorytable.setItem(row, 2, QTableWidgetItem(fracture.fracture_type))  # 骨折类型
                    self.medicalHistorytable.setItem(row, 2, QTableWidgetItem(str(fracture.severity_level)))  # 骨折严重程度
                    self.medicalHistorytable.setItem(row, 3, QTableWidgetItem(str(fracture.diagnosis_details)))  # 诊断描述

                    # 创建按钮
                    view_image_button = QPushButton("查看图像")
                    # 将按钮添加到表格的最后一列
                    self.medicalHistorytable.setCellWidget(row, 4, view_image_button)

                    # 为按钮设置点击事件
                    # view_image_button.clicked.connect(lambda checked, f=fracture: self.view_fracture_image(f))

            # self.medicalHistorytable.setEnabled(False)  # 禁用表格交互

    def view_fracture_image(self, fracture):
        """点击按钮时查看骨折图像"""
        # 假设骨折图像路径保存在 fracture.image_path 中
        image_path = fracture.image_path  # 这里假设你已经在 FractureHistories 中添加了 image_path 字段
        if image_path:
            # 创建一个对话框来展示图像
            dialog = QDialog(self)
            dialog.setWindowTitle("骨折图像")

            # 创建标签来显示图像
            label = QLabel(dialog)
            pixmap = QPixmap(image_path)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            dialog.setLayout(QVBoxLayout())
            dialog.layout().addWidget(label)

            # 设置对话框大小
            dialog.setFixedSize(pixmap.width(), pixmap.height())
            dialog.exec_()
        else:
            print("没有找到骨折图像！")

    def delete_patient_info(self):
        """删除病人信息"""
        print("删除")
        rows = self.tableWidget.rowCount()
        for row in range(rows - 1, -1, -1):  # 从最后一行开始遍历，避免删除时改变行索引
            # 获取复选框所在的单元格项
            item = self.tableWidget.item(row, 0)  # 假设复选框在第1列
            if item and item.checkState() == Qt.Checked:  # 检查复选框是否被选中
                patientID = self.tableWidget.item(row, 1).text()  # 获取当前行的病人ID
                session = Session()  # 获取数据库会话
                try:
                    patient_to_delete = session.query(patients).filter_by(patient_id=patientID).first()
                    print(patient_to_delete)
                    if patient_to_delete:
                        # insert_maintenance_record(self.user_id, housingID, f"删除房屋{housingID}")
                        print(556)
                        session.delete(patient_to_delete)
                        session.commit()  # 提交删除操作
                        # 弹出提示框显示删除成功
                        QMessageBox.information(self, "删除成功", "该病人信息已成功删除！", QMessageBox.Ok)
                        # self.refresh_message_page() #关于操作日志提示信息的显示

                        print(f"删除病人信息：{patientID}")
                    else:
                        print(f"未找到病人ID：{patientID}")
                except Exception as e:
                    print(f"Error deleting patient: {e}")
                    session.rollback()  # 如果出错回滚事务
                finally:
                    session.close()
                # 删除选中的行
                self.tableWidget.removeRow(row)

        # 调用这个方法来添加日志信息
        self.add_log("删除病人成功！", "操作成功", "警告")
        self.refresh_table()

    def add_log(self, message, log_type, icon_type):
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 创建卡片布局
        card_layout = QVBoxLayout()
        card_widget = QWidget()
        card_widget.setLayout(card_layout)

        # 设置卡片边框
        card_widget.setStyleSheet(
            "QWidget { "
            "   border: 1px solid #cccccc; "
            "   border-radius: 5px; "
            "   padding: 10px; "
            "   margin: 5px; "
            "   background-color: #f9f9f9; "
            "}"
        )

        # 创建操作信息
        info_layout = QHBoxLayout()

        # 添加警告图标
        icon = QIcon()
        if icon_type == "警告":
            icon.addFile("../image/plan/头像测试.jpg")  # 替换成警告图标的路径
        else:
            icon.addFile("../image/plan/头像测试.jpg")  # 替换成成功图标的路径

        label_icon = QLabel()
        label_icon.setPixmap(icon.pixmap(80, 80))  # 设置图标大小
        info_layout.addWidget(label_icon)

        # 添加日志信息和时间
        log_info = QLabel(f"{message} ({current_time})")
        log_info.setStyleSheet(
            "QLabel { "
            "   font-size: 14px; "
            "   color: #333333; "
            "   padding-left: 10px; "
            "}"
        )
        info_layout.addWidget(log_info)

        # 将信息加入卡片布局
        card_layout.addLayout(info_layout)

        # 创建 QListWidgetItem，并设置其为卡片样式
        item = QListWidgetItem()
        item.setSizeHint(card_widget.sizeHint())
        self.listWidget.addItem(item)
        self.listWidget.setItemWidget(item, card_widget)

        # 设置定时器，3秒后删除该项
        # QTimer.singleShot(10000, lambda: self.remove_log_item(item))

    def remove_log_item(self, item):
        row = self.listWidget.row(item)
        self.listWidget.takeItem(row)

    def refresh_table(self):
        """刷新病人信息表格"""
        patient_list = self.get_all_patient_info()
        self.fill_patient_table(patient_list)
        self.create_checkBox()

    def create_checkBox(self):
        # 获取当前表格的行数
        rows = self.tableWidget.rowCount()
        for row in range(rows):
            self.tableWidget.setRowHeight(row, 60)  # 设置每一行的高度为40
            checkBoxItem = QTableWidgetItem()  # 创建一个表格项
            checkBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)  # 设置为可选中且启用
            checkBoxItem.setCheckState(Qt.Unchecked)  # 设置复选框初始状态为未选中

            # 将复选框添加到表格的第一列
            self.tableWidget.setItem(row, 0, checkBoxItem)

            self.tableWidget.blockSignals(False)  # 启用信号
            # 连接itemChanged信号到槽函数
            # self.tableWidget.itemChanged.connect(self.on_checkbox_state_changed)


def main():
    app = QApplication(sys.argv)
    main_window = PatientManageWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
