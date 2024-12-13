from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import uic
import os
from db_manager import verify_user  # 引入数据库验证方法
from three_UI import DoctorUI
from three_UI import PatientUI
from three_UI import AdminUI


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_file_path = os.path.join(os.path.dirname(__file__), 'ui', 'login_window.ui')

        # 检查路径是否正确
        print(f"UI 文件路径：{ui_file_path}")

        try:
            uic.loadUi(ui_file_path, self)
        except FileNotFoundError as e:
            print(f"错误：{e}")
            exit()

        self.setWindowTitle("用户登录")
        self.setGeometry(100, 100, 400, 300)

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
        self.main_window = DoctorUI()  # 调用医生主界面
        self.main_window.show()
        self.close()

    def open_patient_main(self, patient_id):
        print("进入病人主页面")
        self.main_window = PatientUI()  # 调用病人主界面
        self.main_window.show()
        self.close()

    def open_admin_main(self, admin_id):
        print("进入管理员主页面")
        self.main_window = AdminUI()  # 调用管理员主界面
        self.main_window.show()
        self.close()

    def show_register_window(self):
        # 注册窗口逻辑（假设已实现）
        from register_window import RegisterWindow
        self.register_window = RegisterWindow()
        self.register_window.show()


# if __name__ == "__main__":
#     app = QApplication([])
#     login_window = LoginWindow()
#     login_window.show()
#     app.exec_()
