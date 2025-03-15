import re

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QMessageBox, QHBoxLayout
from sqlalchemy.exc import IntegrityError

from system.WebSocket import Session
from database.db_manager import doctors, patients


class AddDoctorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新增医生")
        self.setFixedSize(500, 500)

        # **布局**
        layout = QVBoxLayout()

        # *医生 ID（自动生成，不可编辑）**
        self.idLabel = QLabel("医生 ID:")
        self.idField = QLineEdit()
        self.idField.setText(generate_doctor_id())
        self.idField.setReadOnly(True)  # 不可修改
        layout.addWidget(self.idLabel)
        layout.addWidget(self.idField)

        # **医生姓名**
        self.name_label = QLabel("医生姓名:")
        self.name_input = QLineEdit()
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)

        # **电话号码**
        self.phone_label = QLabel("联系方式:")
        self.phone_input = QLineEdit()
        layout.addWidget(self.phone_label)
        layout.addWidget(self.phone_input)

        # **科室选择**
        self.specialty_label = QLabel("科室:")
        self.specialty_combo = QComboBox()
        self.specialty_combo.addItems(["Orthopedics", "Cardiology", "Pediatrics", "Dermatology", "Neurology", "Gastroenterology"])
        layout.addWidget(self.specialty_label)
        layout.addWidget(self.specialty_combo)

        # **确认/取消按钮**
        self.confirm_button = QPushButton("确认")
        self.cancel_button = QPushButton("取消")

        self.confirm_button.clicked.connect(self.add_doctor)
        self.cancel_button.clicked.connect(self.close)

        layout.addWidget(self.confirm_button)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def add_doctor(self):
        """验证并添加医生"""
        name = self.name_input.text().strip()
        phone = self.phone_input.text().strip()
        specialty = self.specialty_combo.currentText()

        if not name:
            QMessageBox.warning(self, "错误", "医生姓名不能为空！")
            return
        if not phone.isdigit() or len(phone) != 10:
            QMessageBox.warning(self, "错误", "电话号码必须是10位数字！")
            return

        session = Session()
        new_id = generate_doctor_id()  # 生成 Dxxx 格式 ID
        print(f"生成的医生 ID: {new_id}")  # ✅ 先打印看看

        new_doctor = doctors(
            doctor_id=new_id,
            doctor_name=name,
            phone=phone,
            specialty=specialty,
            password="88888888"  # 默认密码
        )

        try:
            session.add(new_doctor)
            session.commit()

            # **避免 PyQt5 崩溃**
            QTimer.singleShot(0, lambda: QMessageBox.information(self, "成功", f"医生 {name} 添加成功！"))

            self.accept()  # 关闭窗口
        except IntegrityError:
            session.rollback()
            QMessageBox.critical(self, "错误", "数据库插入失败，可能是 ID 重复！")


def generate_doctor_id():
    session = Session()

    # 获取所有 doctor_id 并转换成整数
    doctor_ids = session.query(doctors.doctor_id).all()
    doctor_ids = [int(id_[0]) for id_ in doctor_ids if id_[0].isdigit()]  # 过滤非数字 ID

    if doctor_ids:
        new_id = str(max(doctor_ids) + 1)  # 获取最大 ID +1
    else:
        new_id = "1"  # 如果表为空，则从 "1" 开始

    return new_id


def generate_patient_id():
    """自动生成新的病人 ID（格式：P00001、P00002...）"""
    session = Session()
    last_patient = session.query(patients).order_by(patients.patient_id.desc()).first()

    if last_patient:
        last_id = int(last_patient.patient_id[1:])  # 去掉 'P'，转为整数
        new_id = f"P{last_id + 1:05d}"  # 递增并格式化
    else:
        new_id = "P00001"  # 第一个病人

    return new_id

class AddPatientDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新增病人")
        self.setFixedSize(500, 500)

        layout = QVBoxLayout(self)

        # **病人 ID（自动生成，不可编辑）**
        self.idLabel = QLabel("病人 ID:")
        self.idField = QLineEdit()
        self.idField.setText(generate_patient_id())
        self.idField.setReadOnly(True)  # 不可修改

        # **病人姓名**
        self.nameLabel = QLabel("姓名:")
        self.nameField = QLineEdit()

        # # **密码（默认值）**
        # self.passwordLabel = QLabel("密码:")
        # self.passwordField = QLineEdit()
        # self.passwordField.setText("88888888")
        # self.passwordField.setReadOnly(True)

        # **按钮**
        self.confirmButton = QPushButton("确定")
        self.cancelButton = QPushButton("取消")

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.confirmButton)
        buttonLayout.addWidget(self.cancelButton)

        # **布局**
        layout.addWidget(self.idLabel)
        layout.addWidget(self.idField)
        layout.addWidget(self.nameLabel)
        layout.addWidget(self.nameField)
        # layout.addWidget(self.passwordLabel)
        # layout.addWidget(self.passwordField)
        layout.addLayout(buttonLayout)

        # **按钮连接**
        self.confirmButton.clicked.connect(self.add_patient)
        self.cancelButton.clicked.connect(self.close)

    def add_patient(self):
        """新增病人到数据库"""
        patient_id = self.idField.text()
        name = self.nameField.text().strip()

        if not name:
            QMessageBox.warning(self, "警告", "请输入病人姓名！")
            return

        new_patient = patients(
            patient_id=patient_id,
            patient_name=name,
            password="88888888"
        )

        try:
            session = Session()
            session.add(new_patient)
            session.commit()
            QMessageBox.information(self, "成功", f"病人 {name} 添加成功！")
            self.accept()  # 关闭窗口
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据库插入失败: {str(e)}")
            session.rollback()
        finally:
            session.close()

class EditDoctorDialog(QDialog):
    def __init__(self, parent, doctor):
        super().__init__(parent)
        self.doctor = doctor
        self.setWindowTitle("编辑医生信息")
        self.setFixedSize(500, 500)

        layout = QVBoxLayout()

        # **医生 ID（不可编辑）**
        self.id_label = QLabel(f"ID: {doctor.doctor_id}")
        layout.addWidget(self.id_label)

        # **姓名**
        self.name_edit = QLineEdit(doctor.doctor_name)
        layout.addWidget(QLabel("姓名:"))
        layout.addWidget(self.name_edit)

        # **联系方式**
        self.phone_edit = QLineEdit(doctor.phone)
        layout.addWidget(QLabel("联系方式:"))
        layout.addWidget(self.phone_edit)

        # **科室（下拉选择框）**
        self.specialty_combo = QComboBox()
        self.specialty_combo.addItems(["Orthopedics", "Cardiology", "Pediatrics", "Dermatology", "Neurology", "Gastroenterology"])
        self.specialty_combo.setCurrentText(doctor.specialty)
        layout.addWidget(QLabel("科室:"))
        layout.addWidget(self.specialty_combo)

        # **按钮**
        self.save_button = QPushButton("保存")
        self.cancel_button = QPushButton("取消")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # **信号连接**
        self.save_button.clicked.connect(self.save_changes)
        self.cancel_button.clicked.connect(self.reject)

    def save_changes(self):
        """保存医生修改信息"""
        new_name = self.name_edit.text().strip()
        new_phone = self.phone_edit.text().strip()

        if not new_name or not new_phone:
            QMessageBox.warning(self, "错误", "姓名和联系方式不能为空！")
            return

        if len(new_phone) != 10 or not new_phone.isdigit():
            QMessageBox.warning(self, "错误", "联系方式必须是 10 位数字！")
            return

        session = Session()
        try:
            # **获取对象并修改属性**
            doctor = session.query(doctors).filter_by(doctor_id=self.doctor.doctor_id).first()
            if doctor:
                doctor.doctor_name = new_name
                doctor.phone = new_phone
                doctor.specialty = self.specialty_combo.currentText()

                session.commit()  # **确保数据提交**
                print(f"✅ 修改成功: {doctor.doctor_id}, {doctor.doctor_name}, {doctor.phone}")  # **调试**

                QMessageBox.information(self, "成功", "医生信息已更新！")
                self.accept()  # **关闭窗口**
            else:
                QMessageBox.warning(self, "错误", "医生记录未找到！")
        except Exception as e:
            session.rollback()
            QMessageBox.critical(self, "错误", f"更新失败：{str(e)}")
        finally:
            session.close()


class EditPatientDialog(QDialog):
    def __init__(self, parent, patient):
        super().__init__(parent)
        self.patient = patient
        self.setWindowTitle("编辑病人信息")
        self.setFixedSize(500, 500)

        layout = QVBoxLayout()

        # **病人 ID（不可编辑）**
        self.id_label = QLabel(f"ID: {patient.patient_id}")
        layout.addWidget(self.id_label)

        # **姓名**
        self.name_edit = QLineEdit(patient.patient_name)
        layout.addWidget(QLabel("姓名:"))
        layout.addWidget(self.name_edit)

        # **联系方式**
        self.phone_edit = QLineEdit(patient.phone_number)
        layout.addWidget(QLabel("联系方式:"))
        layout.addWidget(self.phone_edit)

        # **邮箱**
        self.email_edit = QLineEdit(patient.email)
        layout.addWidget(QLabel("邮箱:"))
        layout.addWidget(self.email_edit)

        # **年龄**
        self.age_edit = QLineEdit(str(patient.age))
        layout.addWidget(QLabel("年龄:"))
        layout.addWidget(self.age_edit)

        # **按钮**
        self.save_button = QPushButton("保存")
        self.cancel_button = QPushButton("取消")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # **信号连接**
        self.save_button.clicked.connect(self.save_changes)
        self.cancel_button.clicked.connect(self.reject)

    import re  # 用于检查邮箱格式

    def save_changes(self):
        """保存病人修改信息"""
        new_name = self.name_edit.text().strip()
        new_phone = self.phone_edit.text().strip()
        new_email = self.email_edit.text().strip()
        new_age = self.age_edit.text().strip()

        # **验证必填字段**
        if not new_name or not new_phone or not new_email or not new_age:
            QMessageBox.warning(self, "错误", "所有字段不能为空！")
            return

        # **验证电话号码（10 位数字）**
        if not new_phone.isdigit() or len(new_phone) != 10:
            QMessageBox.warning(self, "错误", "联系方式必须是 10 位数字！")
            return

        # **验证年龄（必须是 1-120 的整数）**
        if not new_age.isdigit() or not (1 <= int(new_age) <= 120):
            QMessageBox.warning(self, "错误", "年龄必须是 1 到 120 之间的整数！")
            return

        # **验证邮箱格式**
        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if not re.match(email_regex, new_email):
            QMessageBox.warning(self, "错误", "请输入有效的电子邮件地址！")
            return

        session = Session()
        try:
            # **确保 patient 数据在 session 内部获取**
            patient = session.query(patients).filter_by(patient_id=self.patient.patient_id).first()
            if patient:
                patient.patient_name = new_name
                patient.phone_number = new_phone
                patient.email = new_email
                patient.age = int(new_age)

                session.commit()

                QMessageBox.information(self, "成功", "病人信息已更新！")
                self.accept()
            else:
                QMessageBox.warning(self, "错误", "病人记录未找到！")
        except Exception as e:
            session.rollback()
            QMessageBox.critical(self, "错误", f"更新失败：{str(e)}")
        finally:
            session.close()

