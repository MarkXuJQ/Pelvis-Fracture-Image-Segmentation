import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton, QLineEdit
from PyQt5 import uic
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.db_manager import verify_user
from database.db_config import db_config
from doctor_window import DoctorUI
from admin_window import AdminUI

# 创建数据库连接
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
Session = sessionmaker(bind=engine)
session = Session()


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_file_path = os.path.join(os.path.dirname(__file__), 'ui', 'login_window.ui')

        try:
            uic.loadUi(ui_file_path, self)
        except FileNotFoundError as e:
            print(f"错误：{e}")
            exit()

        # 设置窗口基本属性
        self.setWindowTitle("用户登录")
        self.setGeometry(100, 100, 400, 300)

        # 连接按钮事件
        self.login_button.clicked.connect(self.handle_login)

        # 默认选择医生
        self.doctor_radio.setChecked(True)

    def handle_login(self):
        """ 处理用户登录逻辑 """
        user_id = self.user_id_input.text().strip()
        password = self.password_input.text().strip()

        if not user_id or not password:
            QMessageBox.warning(self, "错误", "请输入用户ID和密码")
            return

        # 确定用户类型（医生或管理员）
        user_type = "doctor" if self.doctor_radio.isChecked() else "admin"

        user_exists, message = verify_user(user_id, password, user_type)

        if not user_exists:
            QMessageBox.warning(self, "登录失败", message)  # 正确显示数据库返回的错误信息
        else:
            QMessageBox.information(self, "登录成功", message)
            print(f"登录成功，用户类型: {user_type}, 用户ID: {user_id}")

            if user_type == "doctor":
                self.open_doctor_main(user_id)
            else:
                self.open_admin_main(user_id)

    def open_doctor_main(self, doctor_id):
        """ 打开医生主界面 """
        print("进入医生主页面")
        self.main_window = DoctorUI(doctor_id)
        self.main_window.show()
        self.close()

    def open_admin_main(self, admin_id):
        """ 打开管理员主界面 """
        print("进入管理员主页面")
        self.main_window = AdminUI()
        self.main_window.show()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())
