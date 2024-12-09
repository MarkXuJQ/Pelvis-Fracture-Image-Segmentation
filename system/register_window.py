import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import uic
from system.db_manager import register_user  # 假设你已经在 db_manager.py 中实现了注册函数


class RegisterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_file_path = os.path.join(os.path.dirname(__file__), 'ui', 'register_window.ui')

        # 检查路径是否正确
        print(f"UI 文件路径：{ui_file_path}")

        try:
            uic.loadUi(ui_file_path, self)
        except FileNotFoundError as e:
            print(f"错误：{e}")
            exit()

        self.setWindowTitle("用户注册")

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

