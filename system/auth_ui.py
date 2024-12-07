from PyQt5.QtWidgets import QRadioButton, QMainWindow, QVBoxLayout, QLineEdit, QLabel, QPushButton, QWidget, \
    QMessageBox, QComboBox, QHBoxLayout
from system.db_manager import register_user, verify_user


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("用户登录")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        # 登录界面
        self.user_id_label = QLabel("用户ID:")
        self.user_id_input = QLineEdit()

        self.password_label = QLabel("密码:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        # 用户类型选择（使用水平布局和单选按钮）
        self.user_type_label = QLabel("用户类型:")

        # 创建单选按钮
        self.patient_radio = QRadioButton("病人")
        self.doctor_radio = QRadioButton("医生")
        self.admin_radio = QRadioButton("管理员")

        # 默认选择病人
        self.patient_radio.setChecked(True)

        # 水平布局
        user_type_layout = QHBoxLayout()
        user_type_layout.addWidget(self.patient_radio)
        user_type_layout.addWidget(self.doctor_radio)
        user_type_layout.addWidget(self.admin_radio)

        # 登录和注册按钮
        self.login_button = QPushButton("登录")
        self.register_button = QPushButton("注册")

        # 添加到布局
        layout.addWidget(self.user_id_label)
        layout.addWidget(self.user_id_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.user_type_label)
        layout.addLayout(user_type_layout)  # 添加水平布局
        layout.addWidget(self.login_button)
        layout.addWidget(self.register_button)

        # 设置主窗口布局
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 连接按钮事件
        self.login_button.clicked.connect(self.handle_login)
        self.register_button.clicked.connect(self.show_register_window)

    def handle_login(self):
        user_id = self.user_id_input.text()
        password = self.password_input.text()

        # 判断选择的用户类型
        if self.patient_radio.isChecked():
            user_type = "病人"
        elif self.doctor_radio.isChecked():
            user_type = "医生"
        else:
            user_type = "管理员"

        if not user_id or not password:
            QMessageBox.warning(self, "错误", "请输入用户ID和密码")
            return

        user_type_value = "doctor" if user_type == "医生" else "patient" if user_type == "病人" else "admin"

        user_exists, message = verify_user(user_id, password, user_type_value)

        if not user_exists:
            QMessageBox.warning(self, "提示", "该用户不存在，请先注册")
        else:
            QMessageBox.information(self, "提示", message)
            print(f"登录成功，用户类型: {user_type}, 用户ID: {user_id}")
            if user_type == "医生":
                self.open_doctor_main(user_id)
            elif user_type == "病人":
                self.open_patient_main(user_id)
            else:
                self.open_admin_main(user_id)

    def open_doctor_main(self, doctor_id):
        print("进入医生主页面")
        self.close()

    def open_patient_main(self, patient_id):
        print("进入病人主页面")
        self.close()

    def open_admin_main(self, admin_id):
        print("进入管理员主页面")
        self.close()

    def show_register_window(self):
        # 实现注册窗口的显示
        self.register_window = RegisterWindow()
        self.register_window.show()



class RegisterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("用户注册")
        self.setGeometry(100, 100, 400, 600)

        layout = QVBoxLayout()

        # 注册界面
        self.register_id_label = QLabel("ID:")
        self.register_id_input = QLineEdit()

        self.register_name_label = QLabel("姓名:")
        self.register_name_input = QLineEdit()

        self.register_password_label = QLabel("密码:")
        self.register_password_input = QLineEdit()
        self.register_password_input.setEchoMode(QLineEdit.Password)

        # 用户类型选择框（使用 QComboBox）
        self.user_type_label = QLabel("用户类型:")
        self.user_type_combo = QComboBox()
        self.user_type_combo.addItem("病人")
        self.user_type_combo.addItem("医生")
        self.user_type_combo.addItem("管理员")

        self.register_phone_label = QLabel("电话号码:")
        self.register_phone_input = QLineEdit()

        self.submit_register_button = QPushButton("提交注册")

        # 添加到布局
        layout.addWidget(self.register_id_label)
        layout.addWidget(self.register_id_input)
        layout.addWidget(self.register_name_label)
        layout.addWidget(self.register_name_input)
        layout.addWidget(self.register_password_label)
        layout.addWidget(self.register_password_input)
        layout.addWidget(self.user_type_label)
        layout.addWidget(self.user_type_combo)
        layout.addWidget(self.register_phone_label)
        layout.addWidget(self.register_phone_input)
        layout.addWidget(self.submit_register_button)

        # 设置主窗口布局
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 连接按钮事件
        self.submit_register_button.clicked.connect(self.handle_register)

    def handle_register(self):
        user_id = self.register_id_input.text()
        name = self.register_name_input.text()
        password = self.register_password_input.text()
        phone = self.register_phone_input.text()
        user_type = self.user_type_combo.currentText()

        if not user_id or not name or not password or not phone:
            QMessageBox.warning(self, "错误", "所有字段都是必填项")
            return

        user_type_value = "doctor" if user_type == "医生" else "patient" if user_type == "病人" else "admin"

        success, message = register_user(user_id, name, password, phone, user_type_value)

        if success:
            QMessageBox.information(self, "提示", "注册成功！")
            self.close()  # 关闭注册窗口
        else:
            QMessageBox.warning(self, "错误", message)
