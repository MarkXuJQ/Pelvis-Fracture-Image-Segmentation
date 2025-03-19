from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QPushButton, QMessageBox,
    QTableWidgetItem, QSizePolicy, QHeaderView
)
from database.db_manager import get_connection  # 确保路径正确

class ImageSelectionDialog(QDialog):
    def __init__(self, patient_id, parent=None):
        super().__init__(parent)
        self.patient_id = patient_id
        self.setWindowTitle("选择医学图像")
        self.resize(800, 800)
        self.selected_image_path = None  # 存储选中的图像路径

        # **创建主布局**
        layout = QVBoxLayout(self)

        # **创建表格**
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(4)  # 4列
        self.tableWidget.setHorizontalHeaderLabels(["图像名称", "模态", "路径", "上传日期"])
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)  # 选择整行
        self.tableWidget.setSelectionMode(QTableWidget.SingleSelection)  # 只能选一行
        self.tableWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 自动填充窗口

        # **让所有列等宽**
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)  # 让所有列宽度均等
        layout.addWidget(self.tableWidget)

        # **确认按钮**
        self.confirmButton = QPushButton("加载选中图像")
        self.confirmButton.clicked.connect(self.select_image)
        layout.addWidget(self.confirmButton)

        # **加载数据**
        self.load_patient_images()

    def load_patient_images(self):
        """加载病人的所有图像记录"""
        try:
            connection = get_connection()
            if not connection:
                QMessageBox.critical(self, "数据库错误", "无法连接到数据库，请检查连接配置！")
                return

            cursor = connection.cursor()

            # 查询病人的所有图像
            cursor.execute("""
                SELECT image_name, modality, image_path, upload_date 
                FROM patient_images 
                WHERE patient_id = %s
                ORDER BY upload_date DESC
            """, (self.patient_id,))

            images = cursor.fetchall()

            # **清空表格**
            self.tableWidget.setRowCount(0)

            # **填充表格**
            if images:
                for row, (image_name, modality, image_path, upload_date) in enumerate(images):
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setItem(row, 0, QTableWidgetItem(image_name))
                    self.tableWidget.setItem(row, 1, QTableWidgetItem(modality))
                    self.tableWidget.setItem(row, 2, QTableWidgetItem(image_path))  # 修正路径列
                    self.tableWidget.setItem(row, 3, QTableWidgetItem(str(upload_date)))
            else:
                QMessageBox.information(self, "提示", "当前病人没有医学图像记录。")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像记录失败：{str(e)}")
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def select_image(self):
        """ 选择表格中的医学图像 """
        selected_row = self.tableWidget.currentRow()
        if selected_row == -1:
            QMessageBox.warning(self, "错误", "请先选择一个医学图像")
            return

        self.selected_image_path = self.tableWidget.item(selected_row, 2).text()  # 获取路径
        if not self.selected_image_path:
            QMessageBox.warning(self, "错误", "选中的图像路径无效")
            return

        self.accept()  # 关闭对话框
