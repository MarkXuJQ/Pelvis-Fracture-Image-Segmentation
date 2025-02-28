import sys
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QHeaderView, QListWidgetItem, QFileDialog, \
    QMessageBox, QMenu, QAction, QVBoxLayout, QPushButton, QLabel, QFrame
from PyQt5 import uic
from PyQt5.QtCore import Qt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db_config import db_config

# 创建数据库连接
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
Session = sessionmaker(bind=engine)
session = Session()

class PatientUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the .ui file
        uic.loadUi("ui/patient_window.ui", self)

        # 初始化控件
        self.current_page = 1
        self.items_per_page = 10

        # 设置按钮点击信号
        self.settingsButton.clicked.connect(self.show_settings_menu)
        self.searchButton.clicked.connect(self.search_patients)
        self.cancelButton.clicked.connect(self.cancel_search)
        self.personalInfoButton.clicked.connect(self.show_personal_info)

        # 设置分页按钮信号
        self.firstPageButton.clicked.connect(self.first_page)
        self.previousPageButton.clicked.connect(self.previous_page)
        self.nextPageButton.clicked.connect(self.next_page)
        self.lastPageButton.clicked.connect(self.last_page)
        # 创建菜单（悬浮按钮）
        self.settings_menu = QMenu(self)

        # 创建菜单项
        self.help_action = QAction("帮助", self)
        self.exit_action = QAction("退出", self)

        # 将菜单项添加到菜单中
        self.settings_menu.addAction(self.help_action)
        self.settings_menu.addAction(self.exit_action)
        # 调整布局比例
        self.adjust_layout()
        # 应用样式表
        self.apply_stylesheet()

        # Set up any additional functionality you need here
        # For example, connecting signals to slots, adjusting UI elements, etc.
    def adjust_layout(self):
        """
        调整布局比例和组件位置
        """
        # 假设这是主窗口的布局
        layout = self.findChild(QVBoxLayout, "mainLayout")

        # 为布局设置边距
        layout.setContentsMargins(20, 10, 20, 20)  # 左上右下的间距为10像素

        # 调整主分隔器的左右比例
        self.mainSplitter.setSizes([200, 600,200])  # 左右部分比例


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

    def show_settings_menu(self):
        """显示设置菜单"""
        # 在按钮位置弹出菜单
        self.settings_menu.exec_(self.settingsButton.mapToGlobal(self.settingsButton.rect().bottomLeft()))

    #####需改
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
            #self.tableWidget.setColumnCount(5)  # 5列（增加了“操作”列）
            #self.tableWidget.setHorizontalHeaderLabels(["ID", "姓名", "看病日期", "备注信息", "操作"])

            for row_idx, row_data in enumerate(result):
                for col_idx, value in enumerate(row_data):
                    item = QTableWidgetItem(str(value))
                    self.tableWidget.setItem(row_idx, col_idx, item)

                # **添加 "查看" 按钮**
                #view_button = QPushButton("查看图像")
                #view_button.clicked.connect(lambda _, r=row_data[0]: self.view_patient_details(r))
                #self.tableWidget.setCellWidget(row_idx, 4, view_button)  # 第 5 列（索引 4）

        except Exception as e:
            print(f"Error loading data from database: {e}")

    def show_personal_info(self):
        """
        显示个人信息
        """
        # 清空搜索框和按钮
        for i in reversed(range(self.searchLayout.count())):
            widget = self.searchLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # 清空表格
        #self.tableWidget.setRowCount(0)
        #self.tableWidget.setColumnCount(0)
        # 清空当前布局
        for i in reversed(range(self.tableLayout.count())):
            widget = self.tableLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        # 清空分页控件
        for i in reversed(range(self.pageControlsLayout.count())):
            widget = self.pageControlsLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # 创建并显示个人信息
        personal_info_label = QLabel("个人信息")
        self.tableLayout.addWidget(personal_info_label)

        # 个人信息框框
        personal_info_frame = QFrame()
        personal_info_frame.setFrameShape(QFrame.Box)
        personal_info_frame.setLineWidth(2)
        personal_info_layout = QVBoxLayout()

        # 添加个人信息内容
        name_label = QLabel("姓名：张三")
        age_label = QLabel("年龄：30")
        gender_label = QLabel("性别：男")
        medical_id_label = QLabel("病历号：P001")

        personal_info_layout.addWidget(name_label)
        personal_info_layout.addWidget(age_label)
        personal_info_layout.addWidget(gender_label)
        personal_info_layout.addWidget(medical_id_label)

        personal_info_frame.setLayout(personal_info_layout)
        self.tableLayout.addWidget(personal_info_frame)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PatientUI()  # Create the DoctorUI instance
    window.show()  # Display the window
    sys.exit(app.exec_())  # Run the application's event loop


