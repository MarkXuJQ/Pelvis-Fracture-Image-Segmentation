import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QLabel, QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QTextEdit, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5 import uic

from login_window import LoginWindow

class LoginUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("登录系统")
        self.setGeometry(100, 100, 300, 200)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.username_label = QLabel("用户名:")
        layout.addWidget(self.username_label)
        self.username_input = QLineEdit()
        layout.addWidget(self.username_input)

        self.password_label = QLabel("密码:")
        layout.addWidget(self.password_label)
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)

        self.login_button = QPushButton("登录")
        layout.addWidget(self.login_button)
        self.login_button.clicked.connect(self.login)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        # 注释：验证用户信息并跳转到不同的UI
        print("用户名:", username)
        print("密码:", password)
        # 示例代码：成功后根据角色跳转到不同的UI（这里只是占位逻辑）
        # if 验证通过:
        #     if role == 'patient':
        #         self.patient_ui = PatientUI()
        #         self.patient_ui.show()
        #     elif role == 'doctor':
        #         self.doctor_ui = DoctorUI()
        #         self.doctor_ui.show()
        #     elif role == 'admin':
        #         self.admin_ui = AdminUI()
        #         self.admin_ui.show()

class PatientUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("病人管理系统")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()

        # 医学图像
        image_tab = QWidget()
        image_layout = QVBoxLayout()

        self.image_label = QLabel("医学图像:")
        image_layout.addWidget(self.image_label)

        self.image_table = QTableWidget(0, 2)
        self.image_table.setHorizontalHeaderLabels(["时间", "图像名称"])
        image_layout.addWidget(self.image_table)

        self.upload_button = QPushButton("上传图像")
        image_layout.addWidget(self.upload_button)
        self.upload_button.clicked.connect(self.upload_image)

        image_tab.setLayout(image_layout)
        tabs.addTab(image_tab, "医学图像")

        # 病史管理
        history_tab = QWidget()
        history_layout = QVBoxLayout()

        self.history_label = QLabel("病史记录:")
        history_layout.addWidget(self.history_label)

        self.history_table = QTableWidget(0, 2)
        self.history_table.setHorizontalHeaderLabels(["时间", "详细信息"])
        history_layout.addWidget(self.history_table)

        history_tab.setLayout(history_layout)
        tabs.addTab(history_tab, "病史管理")

        # 个人信息
        info_tab = QWidget()
        info_layout = QVBoxLayout()

        # 个人基本信息
        basic_info_layout = QVBoxLayout()
        self.name_label = QLabel("姓名: 张三")
        self.gender_label = QLabel("性别: 男")
        self.age_label = QLabel("年龄: 30")
        basic_info_layout.addWidget(self.name_label)
        basic_info_layout.addWidget(self.gender_label)
        basic_info_layout.addWidget(self.age_label)
        info_layout.addLayout(basic_info_layout)

        # 修改联系方式
        contact_info_layout = QVBoxLayout()

        contact_label = QLabel("修改联系方式:")
        contact_info_layout.addWidget(contact_label)

        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("联系电话")
        contact_info_layout.addWidget(self.phone_input)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("联系邮箱")
        contact_info_layout.addWidget(self.email_input)

        self.contact_input = QLineEdit()
        self.contact_input.setPlaceholderText("紧急联系人")
        contact_info_layout.addWidget(self.contact_input)

        contact_group = QWidget()
        contact_group.setLayout(contact_info_layout)
        info_layout.addWidget(contact_group)

        # 修改密码
        password_layout = QVBoxLayout()
        self.password_label = QLabel("修改密码:")
        password_layout.addWidget(self.password_label)

        self.new_password_input = QLineEdit()
        self.new_password_input.setEchoMode(QLineEdit.Password)
        self.new_password_input.setPlaceholderText("新密码")
        password_layout.addWidget(self.new_password_input)

        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input.setPlaceholderText("确认新密码")
        password_layout.addWidget(self.confirm_password_input)

        password_group = QWidget()
        password_group.setLayout(password_layout)
        info_layout.addWidget(password_group)

        self.save_button = QPushButton("保存信息")
        info_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_info)

        info_tab.setLayout(info_layout)
        tabs.addTab(info_tab, "个人信息")

        self.setCentralWidget(tabs)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.xpm *.jpg *.bmp *.tiff)")
        if file_name:
            # 注释：将图像上传到数据库的逻辑
            print("上传的图像文件路径:", file_name)

    def save_info(self):
        phone = self.phone_input.text()
        email = self.email_input.text()
        contact = self.contact_input.text()
        new_password = self.new_password_input.text()
        confirm_password = self.confirm_password_input.text()

        if new_password != confirm_password:
            print("密码不匹配！")
            return

        # 注释：保存信息到数据库的逻辑
        print("保存的信息: 电话:", phone, "邮箱:", email, "紧急联系人:", contact, "密码:", new_password)

class DoctorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医生管理系统")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()

        # 医学图像
        image_tab = QWidget()
        image_layout = QVBoxLayout()

        self.patient_image_label = QLabel("病人医学图像:")
        image_layout.addWidget(self.patient_image_label)

        self.patient_image_table = QTableWidget(0, 3)
        self.patient_image_table.setHorizontalHeaderLabels(["病人ID", "时间", "图像名称"])
        image_layout.addWidget(self.patient_image_table)

        self.process_button = QPushButton("处理图像")
        image_layout.addWidget(self.process_button)

        image_tab.setLayout(image_layout)
        tabs.addTab(image_tab, "医学图像")

        # 病人信息
        info_tab = QWidget()
        info_layout = QVBoxLayout()

        self.patient_info_label = QLabel("病人信息:")
        info_layout.addWidget(self.patient_info_label)

        self.patient_info_table = QTableWidget(0, 4)
        self.patient_info_table.setHorizontalHeaderLabels(["病人ID", "姓名", "性别", "年龄"])
        info_layout.addWidget(self.patient_info_table)

        self.view_history_button = QPushButton("查看病史")
        info_layout.addWidget(self.view_history_button)
        self.view_history_button.clicked.connect(self.view_history)

        info_tab.setLayout(info_layout)
        tabs.addTab(info_tab, "病人信息")

        # 个人信息
        personal_info_tab = QWidget()
        personal_info_layout = QVBoxLayout()

        # 欢迎信息
        self.welcome_label = QLabel("医生: XXX 您好！")
        personal_info_layout.addWidget(self.welcome_label)

        # 修改联系方式
        contact_info_layout = QVBoxLayout()
        contact_label = QLabel("修改联系方式:")
        contact_info_layout.addWidget(contact_label)

        self.doctor_phone_input = QLineEdit()
        self.doctor_phone_input.setPlaceholderText("联系电话")
        contact_info_layout.addWidget(self.doctor_phone_input)

        self.doctor_email_input = QLineEdit()
        self.doctor_email_input.setPlaceholderText("联系邮箱")
        contact_info_layout.addWidget(self.doctor_email_input)

        personal_info_layout.addLayout(contact_info_layout)

        # 修改密码
        password_layout = QVBoxLayout()
        self.doctor_password_label = QLabel("修改密码:")
        password_layout.addWidget(self.doctor_password_label)

        self.doctor_new_password_input = QLineEdit()
        self.doctor_new_password_input.setEchoMode(QLineEdit.Password)
        self.doctor_new_password_input.setPlaceholderText("新密码")
        password_layout.addWidget(self.doctor_new_password_input)

        self.doctor_confirm_password_input = QLineEdit()
        self.doctor_confirm_password_input.setEchoMode(QLineEdit.Password)
        self.doctor_confirm_password_input.setPlaceholderText("确认新密码")
        password_layout.addWidget(self.doctor_confirm_password_input)

        personal_info_layout.addLayout(password_layout)

        self.doctor_save_button = QPushButton("保存信息")
        personal_info_layout.addWidget(self.doctor_save_button)
        self.doctor_save_button.clicked.connect(self.save_doctor_info)

        personal_info_tab.setLayout(personal_info_layout)
        tabs.addTab(personal_info_tab, "个人信息")

        self.setCentralWidget(tabs)

    def view_history(self):
        # 注释：加载病人的病史记录逻辑
        print("查看病史功能触发")

    def save_doctor_info(self):
        phone = self.doctor_phone_input.text()
        email = self.doctor_email_input.text()
        new_password = self.doctor_new_password_input.text()
        confirm_password = self.doctor_confirm_password_input.text()

        if new_password != confirm_password:
            print("密码不匹配！")
            return

        # 注释：保存医生信息到数据库的逻辑
        print("保存的信息: 电话:", phone, "邮箱:", email, "密码:", new_password)

class AdminUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("管理员管理系统")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()

        # 管理医生信息
        doctor_tab = QWidget()
        doctor_layout = QVBoxLayout()

        self.doctor_table = QTableWidget(0, 4)
        self.doctor_table.setHorizontalHeaderLabels(["医生ID", "姓名", "电话", "专业"])
        doctor_layout.addWidget(self.doctor_table)

        self.add_doctor_button = QPushButton("添加医生")
        doctor_layout.addWidget(self.add_doctor_button)

        self.edit_doctor_button = QPushButton("编辑医生")
        doctor_layout.addWidget(self.edit_doctor_button)

        self.delete_doctor_button = QPushButton("删除医生")
        doctor_layout.addWidget(self.delete_doctor_button)

        doctor_tab.setLayout(doctor_layout)
        tabs.addTab(doctor_tab, "管理医生信息")

        # 管理病人信息
        patient_tab = QWidget()
        patient_layout = QVBoxLayout()

        self.patient_table = QTableWidget(0, 4)
        self.patient_table.setHorizontalHeaderLabels(["病人ID", "姓名", "电话", "年龄"])
        patient_layout.addWidget(self.patient_table)

        self.add_patient_button = QPushButton("添加病人")
        patient_layout.addWidget(self.add_patient_button)

        self.edit_patient_button = QPushButton("编辑病人")
        patient_layout.addWidget(self.edit_patient_button)

        self.delete_patient_button = QPushButton("删除病人")
        patient_layout.addWidget(self.delete_patient_button)

        patient_tab.setLayout(patient_layout)
        tabs.addTab(patient_tab, "管理病人信息")

        self.setCentralWidget(tabs)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     login_ui = LoginWindow()  # 替换为 login_window.LoginWindow
#     login_ui.show()
#     sys.exit(app.exec_())
