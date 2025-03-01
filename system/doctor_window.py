import datetime
import SimpleITK as sitk
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from xray_viewer import XRayViewer
from ct_viewer import CTViewer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QHeaderView, QListWidgetItem, QFileDialog, \
    QMessageBox, QMenu, QAction, QVBoxLayout, QPushButton
from PyQt5 import uic
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from chat_window import ChatApp
from settings_dialog import SettingsDialog
from db_config import db_config

# 创建数据库连接
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
Session = sessionmaker(bind=engine)
session = Session()

class DoctorUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加载 .ui 文件
        uic.loadUi("ui/doctor_window.ui", self)

        # 初始化属性
        self.patient_data = []
        self.current_page = 1
        self.items_per_page = 10
        self.viewer = None  # Will hold the current image viewer
        self.render_on_open = False

        self.message_list_data = [
            "注意事项：请检查病人病史。",
            "更新提醒：病人列表已更新。",
            "系统提示：分配操作成功。",
        ]

        # 初始化病人列表和消息提示
        self.messageList.addItems(self.message_list_data)

        # 设置分页按钮信号
        self.firstPageButton.clicked.connect(self.first_page)
        self.previousPageButton.clicked.connect(self.previous_page)
        self.nextPageButton.clicked.connect(self.next_page)
        self.lastPageButton.clicked.connect(self.last_page)

        # 表格单击事件
        self.tableWidget.cellClicked.connect(lambda row, column: self.display_details(row))
        self.viewButton.clicked.connect(self.open_image)
        self.searchButton.clicked.connect(self.search_patients)
        self.cancelButton.clicked.connect(self.cancel_search)
        # 设置按钮点击信号
        self.settingsButton.clicked.connect(self.show_settings_menu)
        self.chatCollaButton.clicked.connect(self.open_collaboration_window)  # 点击时打开聊天合作窗口

        # 创建菜单（悬浮按钮）
        self.settings_menu = QMenu(self)

        # 创建菜单项
        self.help_action = QAction("帮助", self)
        self.exit_action = QAction("退出", self)
        self.settings_action = QAction("3D模型", self)
        # 将菜单项添加到菜单中
        self.settings_menu.addAction(self.help_action)
        self.settings_menu.addAction(self.exit_action)
        self.settings_menu.addAction(self.settings_action)

        self.settings_action.triggered.connect(self.open_settings)

        # 调整布局比例
        self.adjust_layout()

        # 初始化表格
        self.load_data_from_database()

        # 应用样式表
        self.apply_stylesheet()

    def adjust_layout(self):
        """
        调整布局比例和组件位置
        """
        # 假设这是主窗口的布局
        layout = self.findChild(QVBoxLayout, "mainLayout")

        # 为布局设置边距
        layout.setContentsMargins(20, 10, 20, 20)  # 左上右下的间距为10像素

        # 调整主分隔器的左右比例
        self.mainSplitter.setSizes([400, 800])  # 左右部分比例

        # 调整右侧分隔器的上下比例
        self.rightSplitter.setSizes([600, 200])  # 上下部分比例

        # 确保顶部导航栏按钮靠左
        self.topLayout.insertStretch(2, 1)  # 添加一个弹性项，将按钮挤到左侧
        # 调整表格列的比例，防止表格占用太多空白
        self.tableWidget.horizontalHeader().setStretchLastSection(True)  # 自动拉伸最后一列
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)

        # 调整分页控件布局
        self.pageControlsLayout.setSpacing(10)  # 调整分页控件间距
        self.pageLabel.setAlignment(Qt.AlignCenter)  # 设置页码居中
        self.pageControlsLayout.setContentsMargins(200, 0, 200, 0)  # 控制分页栏在中间显示

        # 在列表底部添加一个"未命名"项
        add_patient_item = QListWidgetItem("未命名")
        self.patientList.addItem(add_patient_item)
        # 点击"未命名"项时触发相应的操作
        #add_patient_item.clicked.connect(self.add_new_patient)

        # 设置QHBoxLayout的伸缩因子
        self.detailsLayout.setStretch(0, 1)  # 病人信息部分占 1 的比例
        self.detailsLayout.setStretch(1, 0)  # 分隔线不占比例
        self.detailsLayout.setStretch(2, 2)  # 病人病史部分占 2 的比例
        self.reset_details()

    def apply_stylesheet(self):
        dark_theme = """
        QWidget {
            background-color: #20232A;
            color: #FFFFFF;
            font-family: "Arial";
            font-size: 16px;
        }

        QLabel {
            color: #E0E0E0;
        }

        QPushButton {
            background-color: #444;
            color: #FFFFFF;
            border: 1px solid #5C5C5C;
            border-radius: 5px;
            padding: 8px;
        }

        QPushButton:hover {
            background-color: #505357;
        }

        QPushButton:pressed {
            background-color: #606366;
        }

        QLineEdit {
            background-color: #2E3138;
            color: #FFFFFF;
            border: 1px solid #5C5C5C;
            padding: 5px;
            border-radius: 4px;
        }

        QTableWidget {
            background-color: #2E3138;
            color: #FFFFFF;
            border: 1px solid #444;
            gridline-color: #5C5C5C;
            alternate-background-color: #282C34;
        }

        QHeaderView::section {
            background-color: #444;
            color: #E0E0E0;
            border: 1px solid #5C5C5C;
            padding: 4px;
        }

        QListWidget {
            background-color: #2E3138;
            color: #FFFFFF;
            border: 1px solid #444;
            padding: 5px;
        }

        QFrame#detailsFrame {
            background-color: #2E3138;
            border: 2px solid #5C5C5C;
            border-radius: 10px;
            padding: 15px;
        }
        """
        self.setStyleSheet(dark_theme)

    def load_data_from_database(self):
        """从数据库加载数据并更新表格"""
        try:
            query = text("""
                SELECT p.patient_id, p.patient_name, f.fracture_date, f.diagnosis_details
                FROM patients p
                LEFT JOIN fracturehistories f ON p.patient_id = f.patient_id
            """)
            result = session.execute(query).fetchall()
            if result is None:
                result = []

            self.tableWidget.setRowCount(len(result))
            self.tableWidget.setColumnCount(5)  # 5列（增加了"操作"列）
            self.tableWidget.setHorizontalHeaderLabels(["ID", "姓名", "看病日期", "备注信息", "操作"])

            for row_idx, row_data in enumerate(result):
                for col_idx, value in enumerate(row_data):
                    item = QTableWidgetItem(str(value))
                    self.tableWidget.setItem(row_idx, col_idx, item)

                # **添加 "查看" 按钮**
                view_button = QPushButton("查看图像")
                view_button.clicked.connect(lambda _, r=row_data[0]: self.view_patient_details(r))
                self.tableWidget.setCellWidget(row_idx, 4, view_button)  # 第 5 列（索引 4）

        except Exception as e:
            print(f"Error loading data from database: {e}")

    def view_patient_details(self, patient_id):
        """处理查看按钮的点击事件"""
        print(f"查看病人 {patient_id} 的详细信息")
    def open_settings(self):
        dialog = SettingsDialog(self, render_on_open=self.render_on_open)
        if dialog.exec_():
            settings = dialog.get_settings()
            self.render_on_open = settings['render_on_open']
    def update_table(self):
        """更新表格内容并设置分页"""
        self.tableWidget.setRowCount(0)
        start_index = (self.current_page - 1) * self.items_per_page
        end_index = min(start_index + self.items_per_page, len(self.patient_data))

        for row_idx, row_data in enumerate(self.patient_data[start_index:end_index]):
            self.tableWidget.insertRow(row_idx)
            for col_idx, col_data in enumerate(row_data):
                # 将非字符串数据转换为字符串
                if isinstance(col_data, (datetime.date, datetime.datetime)):
                    col_data = col_data.strftime("%Y-%m-%d")  # 转换为 'YYYY-MM-DD' 格式

                self.tableWidget.setItem(row_idx, col_idx, QTableWidgetItem(col_data))

        total_pages = (len(self.patient_data) + self.items_per_page - 1) // self.items_per_page
        self.pageLabel.setText(f"{self.current_page}/{total_pages}")

    def load_patients_to_list(self, row):
        """从数据库加载所有病人信息并显示在patientList中"""
        """
            处理点击表格某一行的操作，获取对应的病人姓名和看病日期并显示在patientList中
            """
        # 获取点击行的病人姓名和看病日期
        patient_name = self.tableWidget.item(row, 1).text()  # 假设病人姓名在第0列
        fracture_date = self.tableWidget.item(row, 2).text()  # 假设看病日期在第1列
        # 格式化看病日期为 MM.DD 格式
        try:
            fracture_date = pd.to_datetime(fracture_date).strftime('%m.%d')  # 使用pandas格式化日期
        except ValueError:
            fracture_date = "未知"  # 如果日期格式不正确，设置为"未知"

        # 格式化病人信息为 '06.25张三' 这种形式
        formatted_patient_info = f"{fracture_date}{patient_name}"

        # 检查patientList中是否已存在相同的病人信息
        existing_items = self.patientList.findItems(formatted_patient_info, QtCore.Qt.MatchExactly)

        # 如果patientList中没有该病人信息，就添加到列表
        if not existing_items:
            self.patientList.insertItem(0, formatted_patient_info)
        else:
            # 如果已经存在，先删除已存在的项
            existing_item = existing_items[0]
            row_index = self.patientList.row(existing_item)  # 获取已存在项的行数
            self.patientList.takeItem(row_index)  # 移除该项
            # 然后再将该项插入到最前面
            self.patientList.insertItem(0, formatted_patient_info)

        # 设置选中对应的项
        selected_item = self.patientList.item(0)  # 选中刚刚插入的项
        self.patientList.setCurrentItem(selected_item)  # 设置选中项
        # 设置鼠标悬停时的背景颜色
        self.patientList.setStyleSheet("""
                QListWidget::item:hover {
                    background-color: #505357;  /* 设置鼠标悬停时的背景颜色 */
                }
            """)

    def show_settings_menu(self):
        """显示设置菜单"""
        # 在按钮位置弹出菜单
        self.settings_menu.exec_(self.settingsButton.mapToGlobal(self.settingsButton.rect().bottomLeft()))

    def open_collaboration_window(self):
        # 打开聊天合作窗口
        self.collab_window = ChatApp()
        self.collab_window.show()
    # 定义点击事件处理函数
    def on_patient_item_clicked(self, item):
        # 获取点击项的文本内容
        patient_name = item.text()
        print(f"点击的病人姓名: {patient_name}")
        # 在这里可以根据病人姓名查询数据库或更新界面

    def first_page(self):
        """跳转到第一页"""
        if self.current_page > 1:
            self.current_page = 1
            self.update_table()

    def previous_page(self):
        """跳转到上一页"""
        if self.current_page > 1:
            self.current_page -= 1
            self.update_table()

    def next_page(self):
        """跳转到下一页"""
        total_pages = (len(self.patient_data) + self.items_per_page - 1) // self.items_per_page
        if self.current_page < total_pages:
            self.current_page += 1
            self.update_table()

    def last_page(self):
        """跳转到最后一页"""
        total_pages = (len(self.patient_data) + self.items_per_page - 1) // self.items_per_page
        if self.current_page < total_pages:
            self.current_page = total_pages
            self.update_table()

    def display_details(self, row):
        """显示病人详情"""
        # 获取选中行的病人 ID
        patient_id = self.tableWidget.item(row, 0).text()
        try:
            print(2)
            # 查询病人信息
            patient_query = text("""
                        SELECT patient_name, gender, age, phone_number, date_of_birth
                        FROM patients WHERE patient_id = :patient_id
                    """)
            patient_result = session.execute(patient_query, {"patient_id": patient_id}).fetchone()
            print("Patient Result:", patient_result)

            print(3)
            # 查询病人的骨折病史信息
            history_query = text("""
                       SELECT fracture_date, fracture_location, severity_level, diagnosis_details
                       FROM fracturehistories WHERE patient_id = :patient_id
                   """)

            history_results = session.execute(history_query, {"patient_id": patient_id}).fetchall()

            print(4)
            # 设置病人基本信息
            if patient_result:
                print(6)
                # 提取字段值，并处理可能为 None 的情况
                patient_name = patient_result[0] if patient_result[0] else "未知"
                gender = patient_result[1] if patient_result[1] else "未知"
                age = patient_result[2] if patient_result[2] is not None else "未知"
                phone_number = patient_result[3] if patient_result[3] else "未知"
                date_of_birth = patient_result[4].strftime('%Y-%m-%d') if patient_result[4] else "未知"
                print(7)
                # 格式化病人信息
                patient_details = (
                    f"病人信息：\n"
                    f"姓名：{patient_name}\n"
                    f"性别：{gender}\n"
                    f"年龄：{age}\n"
                    f"联系电话：{phone_number}\n"
                    f"出生日期：{date_of_birth}\n"
                )
                self.patientInfoLabel.setText(patient_details)
                print(5)  # 调试信息
            else:
                self.patientInfoLabel.setText("病人信息：\n未找到病人信息")
                print("未找到病人信息")
            print(5)
            # 设置病人病史信息
            if history_results:
                history_details = "病人病史：\n"
                for history in history_results:
                    history_details += (
                        f"诊断日期：{history[0]}\n"  # fracture_date
                        f"骨折位置：{history[1]}\n"  # fracture_location
                        f"严重程度：{history[2]}\n"  # severity_level
                        f"诊断详情：{history[3]}\n\n"  # diagnosis_details
                    )
                self.patientHistoryLabel.setText(history_details.strip())
                self.load_patients_to_list(row)
                print(8)
            else:
                self.patientHistoryLabel.setText("病人病史：\n无病史记录")
            print(9)
        except Exception as e:
            print(f"Error fetching patient details: {e}")
            self.reset_details()

    def reset_details(self):
        """重置病人详情"""
        # 假设分隔器divider位于病人信息和病人病史之间，调整它们的比例

        self.patientInfoLabel.setText(
            "病人信息：\n"
            "姓名：\n"
            "性别：\n"
            "年龄：\n"
            "联系电话：\n"
            "出生日期：\n"
        )
        self.patientHistoryLabel.setText(
            "病人病史：\n"
            "诊断日期：\n"
            "骨折位置：\n"
            "严重程度：\n"
            "诊断详情：\n"
        )

    def search_patients(self):
        """根据搜索框内容查询病人数据并更新表格"""
        search_query = self.searchBox.text().strip()

        if not search_query:
            # 如果搜索框为空，加载所有病人数据
            self.load_data_from_database()
            return

        try:
            print(f"搜索关键词: {search_query}")  # 调试信息

            # 构建查询语句，根据搜索框内容过滤病人数据
            query = text("""
                SELECT p.patient_id, p.patient_name, f.fracture_date, f.diagnosis_details
                FROM patients p
                LEFT JOIN fracturehistories f ON p.patient_id = f.patient_id
                WHERE p.patient_name LIKE :search_query OR f.fracture_date LIKE :search_query
            """)
            result = session.execute(query, {"search_query": f"%{search_query}%"}).fetchall()

            if not result:
                print("没有找到匹配的病人数据")  # 调试信息

            self.patient_data = [list(row) for row in result]
            self.update_table()  # 确保表格更新
        except Exception as e:
            print(f"Error during search: {e}")

    def cancel_search(self):
        """取消搜索并还原表格数据"""
        self.searchBox.clear()  # 清空搜索框
        self.load_data_from_database() # 还原表格数据

    def open_image(self):
        # Open file dialog to select image
        options = QFileDialog.Options()
        file_types = "All Files (*);;DICOM Files (*.dcm);;NIfTI Files (*.nii *.nii.gz);;NRRD Files (*.nrrd);;MetaImage Files (*.mha *.mhd)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", file_types, options=options)
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        try:
            self.render_on_open = False
            self.image = sitk.ReadImage(file_path)
            dimension = self.image.GetDimension()
            if dimension == 2:
                # Display 2D image
                image_array = sitk.GetArrayFromImage(self.image)
                self.viewer = XRayViewer(image_array)

            elif dimension == 3:
                # Display 3D image
                self.viewer = CTViewer(self.image, render_model=self.render_on_open)

            else:
                QMessageBox.warning(self, "Unsupported Image", "The selected image has unsupported dimensions.")
                return

            self.setCentralWidget(self.viewer)
            self.statusBar().showMessage(f'Loaded image: {file_path}')
            self.current_file_path = file_path  # Store the current file path
            #self.save_as_action.setEnabled(True)  # Enable "Save As"
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = DoctorUI()
    window.show()
    sys.exit(app.exec_())
