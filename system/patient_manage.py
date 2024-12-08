from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
import sys

from requests import Session


class PatientManageWindow(QMainWindow):
    def __init__(self,table):
        super(PatientManageWindow, self).__init__()
        self.setWindowTitle("Patient Management")
        self.setGeometry(300, 300, 400, 300)
        self.tableWidget=table
        self.initUI()

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
        # Placeholder for adding a patient
        print("Add Patient button clicked")

    def load_patient(self):
        # Placeholder for loading patient data
        print("Load Patient button clicked")

    def view_patient(self):
        # Placeholder for viewing patient data
        print("View Patient Data button clicked")

    def on_delete_patient_info(self):
        """点击删除按钮时删除选中复选框的行"""
        # 获取所有行
        print(222)
        rows = self.tableWidget.rowCount()

        # 从最后一行开始往前遍历，避免删除时改变了行索引
        for row in range(rows - 1, -1, -1):
            item = self.tableWidget.item(row, 0)  # 获取复选框所在的单元格项
            if item and item.checkState() == Qt.Checked:  # 如果复选框被选中
                self.tableWidget.removeRow(row)  # 删除该行
                print(f"Deleted row {row}")
        #连接到数据库后，再同步删除数据库里面的信息


def main():
    app = QApplication(sys.argv)
    main_window = PatientManageWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
