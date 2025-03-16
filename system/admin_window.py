import os
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QTableWidgetItem, QPushButton,
    QVBoxLayout, QHeaderView, QMessageBox, QSplitter, QWidget, QSizePolicy, QListWidgetItem, QListWidget, QHBoxLayout
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.uic import loadUi
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from database.db_manager import doctors, patients, db_config  # 你的数据库表
from admin_manager import AddDoctorDialog, AddPatientDialog, EditDoctorDialog, EditPatientDialog, \
    send_message_to_all_doctors
from stylesheet import apply_stylesheet

# 创建数据库连接
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
Session = sessionmaker(bind=engine)
session = Session()


class AdminUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加载 UI
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(current_dir, "ui", "admin_window.ui")
        loadUi(ui_file, self)

        # **调整 UI 布局比例**
        self.adjust_layout()

        # **初始化导航栏**
        self.setup_navigation()

        # **初始化数据**
        self.current_page = 1
        self.items_per_page = 10
        self.current_mode = "doctor"  # 默认管理医生
        self.load_data()

        # **连接信号**
        self.searchButton.clicked.connect(self.search_records)
        self.resetButton.clicked.connect(self.load_data)
        self.addButton.clicked.connect(self.add_record)
        self.tableWidget.cellClicked.connect(self.display_details)
        self.tableWidget.viewport().installEventFilter(self)

        # **分页按钮**
        self.firstPageButton.clicked.connect(self.first_page)
        self.previousPageButton.clicked.connect(self.previous_page)
        self.nextPageButton.clicked.connect(self.next_page)
        self.lastPageButton.clicked.connect(self.last_page)

        apply_stylesheet(self)

    def reset_details(self):
        """重置详情界面为空"""
        self.detailsLabel.setText("详情信息将在此处显示")

    def mousePressEvent(self, event):
        """监听整个窗口的点击事件，如果点击空白处，则清空详情"""
        if not self.tableWidget.underMouse():  # 如果鼠标不在表格上
            self.reset_details()
        super().mousePressEvent(event)  # 调用默认处理

    def eventFilter(self, obj, event):
        """监听表格的点击事件，如果点击空白处，则清空详情"""
        if obj == self.tableWidget.viewport() and event.type() == QEvent.MouseButtonPress:
            index = self.tableWidget.indexAt(event.pos())
            if not index.isValid():  # 点击的不是有效的表格单元格
                self.reset_details()
        return super().eventFilter(obj, event)

    def adjust_layout(self):
        """调整 UI 布局比例"""
        self.mainSplitter.setSizes([250, 950])  # 左侧小，右侧大
        self.rightSplitter.setSizes([700, 250])  # 上部表格 700px，下部详情 250px
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tableWidget.setSizePolicy(size_policy)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def setup_navigation(self):
        """创建侧边导航栏"""
        self.navList = QListWidget()

        self.leftLayout.addWidget(self.navList)

        # 添加菜单项
        doctor_item = QListWidgetItem("医生管理")
        doctor_item.setTextAlignment(Qt.AlignCenter)
        self.navList.addItem(doctor_item)

        patient_item = QListWidgetItem("病人管理")
        patient_item.setTextAlignment(Qt.AlignCenter)
        self.navList.addItem(patient_item)

        self.navList.itemClicked.connect(self.handle_nav_click)

    def handle_nav_click(self, item):
        """处理导航栏点击事件"""
        if item.text() == "医生管理":
            self.current_mode = "doctor"
        elif item.text() == "病人管理":
            self.current_mode = "patient"
        self.load_data()

    def populate_table(self, data, update_pagination=True):
        """填充表格数据（适用于 load_data 和 search_records），支持分页"""
        if self.current_mode == "doctor":
            headers = ["ID", "姓名", "联系方式", "科室", "操作"]
        else:
            headers = ["ID", "姓名", "联系方式", "邮箱", "操作"]

        # 处理分页
        if update_pagination:
            self.total_pages = (len(data) + self.items_per_page - 1) // self.items_per_page
            self.current_page = min(self.current_page, max(self.total_pages, 1))  # 确保当前页合法
            self.pageLabel.setText(f"{self.current_page}/{self.total_pages}")

        start_index = (self.current_page - 1) * self.items_per_page
        end_index = min(start_index + self.items_per_page, len(data))
        data_to_display = data[start_index:end_index]

        # 如果没有数据
        if not data_to_display:
            self.tableWidget.setRowCount(1)
            self.tableWidget.setColumnCount(1)
            self.tableWidget.setItem(0, 0, QTableWidgetItem("未找到用户"))
            return

        # **插入数据**
        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)
        self.tableWidget.setRowCount(len(data_to_display))

        for row_idx, record in enumerate(data_to_display):
            if self.current_mode == "doctor":
                self.tableWidget.setItem(row_idx, 0, QTableWidgetItem(str(record.doctor_id)))
                self.tableWidget.setItem(row_idx, 1, QTableWidgetItem(record.doctor_name))
                self.tableWidget.setItem(row_idx, 2, QTableWidgetItem(record.phone))
                self.tableWidget.setItem(row_idx, 3, QTableWidgetItem(record.specialty))
            else:
                self.tableWidget.setItem(row_idx, 0, QTableWidgetItem(str(record.patient_id)))
                self.tableWidget.setItem(row_idx, 1, QTableWidgetItem(record.patient_name))
                self.tableWidget.setItem(row_idx, 2, QTableWidgetItem(record.phone_number))
                self.tableWidget.setItem(row_idx, 3, QTableWidgetItem(record.email))

            # **添加操作按钮（编辑 + 删除）**
            action_widget = QWidget()
            layout = QHBoxLayout(action_widget)
            layout.setContentsMargins(0, 0, 0, 0)

            edit_button = QPushButton("编辑")
            delete_button = QPushButton("删除")

            edit_button.clicked.connect(lambda _, r=row_idx: self.edit_record(r))
            delete_button.clicked.connect(lambda _, r=row_idx: self.delete_record(r))

            layout.addWidget(edit_button)
            layout.addWidget(delete_button)
            action_widget.setLayout(layout)

            self.tableWidget.setCellWidget(row_idx, 4, action_widget)

        self.tableWidget.viewport().update()

    def load_data(self):
        """加载医生或病人数据"""
        session = Session()
        if self.current_mode == "doctor":
            data = session.query(doctors).all()
            self.tableWidget.setColumnCount(5)  # 4 列
            self.tableWidget.setHorizontalHeaderLabels(["ID", "姓名", "联系方式", "科室", "操作"])
        else:
            data = session.query(patients).all()
            self.tableWidget.setColumnCount(5)  # 4 列
            self.tableWidget.setHorizontalHeaderLabels(["ID", "姓名", "联系方式", "邮箱", "操作"])

        # self.current_page = 1  # 加载新数据时回到第一页
        # self.populate_table(data)
        # **分页逻辑**
        self.total_pages = max((len(data) + self.items_per_page - 1) // self.items_per_page, 1)

        # **确保当前页不超出范围**
        self.current_page = min(self.current_page, self.total_pages)

        # **填充表格**
        self.populate_table(data)

        # **更新分页按钮**
        self.update_pagination_buttons()

    def update_pagination_buttons(self):
        """更新分页按钮的状态"""
        self.firstPageButton.setEnabled(self.current_page > 1)
        self.previousPageButton.setEnabled(self.current_page > 1)
        self.nextPageButton.setEnabled(self.current_page < self.total_pages)
        self.lastPageButton.setEnabled(self.current_page < self.total_pages)


        # # **分页逻辑**
        # total_pages = (len(data) + self.items_per_page - 1) // self.items_per_page
        # self.pageLabel.setText(f"{self.current_page}/{total_pages}")
        #
        # # **插入数据**
        # start_index = (self.current_page - 1) * self.items_per_page
        # end_index = min(start_index + self.items_per_page, len(data))
        # self.tableWidget.setRowCount(end_index - start_index)
        #
        # for row_idx, record in enumerate(data[start_index:end_index]):
        #     if self.current_mode == "doctor":
        #         self.tableWidget.setItem(row_idx, 0, QTableWidgetItem(str(record.doctor_id)))
        #         self.tableWidget.setItem(row_idx, 1, QTableWidgetItem(record.doctor_name))
        #         self.tableWidget.setItem(row_idx, 2, QTableWidgetItem(record.phone))
        #         self.tableWidget.setItem(row_idx, 3, QTableWidgetItem(record.specialty))
        #     else:
        #         self.tableWidget.setItem(row_idx, 0, QTableWidgetItem(str(record.patient_id)))
        #         self.tableWidget.setItem(row_idx, 1, QTableWidgetItem(record.patient_name))
        #         self.tableWidget.setItem(row_idx, 2, QTableWidgetItem(record.phone_number))
        #         self.tableWidget.setItem(row_idx, 3, QTableWidgetItem(record.email))
        #
        #     # **添加操作按钮（编辑 + 删除）**
        #     action_widget = QWidget()
        #     layout = QHBoxLayout(action_widget)
        #     layout.setContentsMargins(0, 0, 0, 0)
        #
        #     edit_button = QPushButton("编辑")
        #     delete_button = QPushButton("删除")
        #
        #     edit_button.clicked.connect(lambda _, r=row_idx: self.edit_record(r))
        #     delete_button.clicked.connect(lambda _, r=row_idx: self.delete_record(r))
        #
        #     layout.addWidget(edit_button)
        #     layout.addWidget(delete_button)
        #     action_widget.setLayout(layout)
        #
        #     self.tableWidget.setCellWidget(row_idx, 4, action_widget)
        #
        # self.tableWidget.viewport().update()

    def display_details(self, row):
        """点击表格行，显示详情（美化排版 + 两列布局）"""
        session = Session()
        record_id = self.tableWidget.item(row, 0).text()

        if self.current_mode == "doctor":
            record = session.query(doctors).filter_by(doctor_id=record_id).first()
            details = f"""
            <table width="100%">
                <tr><td><b>ID：</b></td>  <td>{record.doctor_id}</td>  <td><b>科室：</b></td> <td>{record.specialty}</td></tr>
                <tr><td><b>姓名：</b></td>  <td>{record.doctor_name}</td>  <td><b>联系方式：</b></td> <td>{record.phone}</td></tr>
            </table>
            """
        else:
            record = session.query(patients).filter_by(patient_id=record_id).first()
            details = f"""
            <table width="100%">
                <tr><td><b>ID：</b></td> <td>{record.patient_id}</td> <td><b>出生日期：</b></td> <td>{record.date_of_birth}</td></tr>
                <tr><td><b>姓名：</b></td> <td>{record.patient_name}</td> <td><b>性别：</b></td> <td>{record.gender}</td></tr>
                <tr><td><b>年龄：</b></td> <td>{record.age}</td> <td><b>身份证号：</b></td> <td>{record.id_card}</td></tr>
                <tr><td><b>联系人：</b></td> <td>{record.contact_person}</td> <td><b>联系人电话：</b></td> <td>{record.contact_phone}</td></tr>
                <tr><td><b>联系电话：</b></td> <td>{record.phone_number}</td> <td><b>邮箱：</b></td> <td>{record.email}</td></tr>
            </table>
            """

        self.detailsLabel.setText(details)


    def search_records(self):
        """搜索医生或病人，支持分页"""
        search_query = self.searchBox.text().strip()
        session = Session()

        if not search_query:
            self.load_data()
            return

        if self.current_mode == "doctor":
            self.filtered_data = session.query(doctors).filter(
                (doctors.doctor_id.like(f"%{search_query}%")) |
                (doctors.doctor_name.like(f"%{search_query}%"))
            ).all()
        else:
            self.filtered_data = session.query(patients).filter(
                (patients.patient_id.like(f"%{search_query}%")) |
                (patients.patient_name.like(f"%{search_query}%"))
            ).all()

        self.current_page = 1  # 搜索时回到第一页
        self.populate_table(self.filtered_data)

    # def load_data_from_list(self, data):
    #     """从查询结果加载数据"""
    #     self.tableWidget.setRowCount(len(data))
    #
    #     for row_idx, record in enumerate(data):
    #         self.tableWidget.setItem(row_idx, 0, QTableWidgetItem(
    #             str(record.doctor_id if self.current_mode == "doctor" else record.patient_id)))
    #         self.tableWidget.setItem(row_idx, 1, QTableWidgetItem(
    #             record.doctor_name if self.current_mode == "doctor" else record.patient_name))
    #         self.tableWidget.setItem(row_idx, 2, QTableWidgetItem(record.phone))
    #         self.tableWidget.setItem(row_idx, 3, QTableWidgetItem(
    #             record.specialty if self.current_mode == "doctor" else str(record.age)))

    def add_record(self):
        """根据当前模式（医生 / 病人）弹出新增窗口"""
        if self.current_mode == "doctor":
            dialog = AddDoctorDialog(self)
        else:
            dialog = AddPatientDialog(self)

        if dialog.exec_():  # 如果成功新增，则刷新表格
            # self.total_pages += 1  # 确保新增后有新页
            # self.current_page = self.total_pages
            self.load_data()

    def edit_record(self, row):
        """编辑医生或病人"""
        session = Session()
        record_id = self.tableWidget.item(row, 0).text()

        if self.current_mode == "doctor":
            record = session.query(doctors).filter_by(doctor_id=record_id).first()
            if not record:
                QMessageBox.warning(self, "错误", "未找到该医生！")
                return
            dialog = EditDoctorDialog(self, record)
        else:
            record = session.query(patients).filter_by(patient_id=record_id).first()
            if not record:
                QMessageBox.warning(self, "错误", "未找到该病人！")
                return
            dialog = EditPatientDialog(self, record)

        if dialog.exec_():
            self.load_data()  # **刷新表格**

    def delete_record(self, row):
        """删除医生/病人"""
        record_id = self.tableWidget.item(row, 0).text()

        reply = QMessageBox.question(
            self, "确认删除", f"确定要删除 ID {record_id} 的记录吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 获取被删除的病人 ID 和姓名
            deleted_id = self.tableWidget.item(row, 0).text()
            deleted_name = self.tableWidget.item(row, 1).text()

            session = Session()

            try:
                if self.current_mode == "doctor":
                    record = session.query(doctors).filter_by(doctor_id=record_id).first()
                else:
                    record = session.query(patients).filter_by(patient_id=record_id).first()

                if record:
                    session.delete(record)
                    session.commit()

                    # **刷新表格**
                    self.load_data()

                    # **如果当前页数据被删光，且不在第一页，则回到上一页**
                    if self.current_page > 1 and self.tableWidget.rowCount() == 0:
                        self.current_page -= 1
                        self.load_data()

                    QMessageBox.information(self, "删除成功", "记录已成功删除！")
                    # 发送通知
                    if self.current_mode == "patient":
                        send_message_to_all_doctors(
                            sender_id="admin",
                            message_type="notification",
                            message_content=f"病人 {deleted_name}（ID: {deleted_id}）已被管理员删除，请更新相关记录。"
                        )
                else:
                    QMessageBox.warning(self, "错误", "未找到该记录，删除失败！")

            except Exception as e:
                session.rollback()
                QMessageBox.critical(self, "错误", f"删除失败：{str(e)}")

            finally:
                session.close()

    def first_page(self):
        self.current_page = 1
        self.load_data()

    def previous_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.load_data()

    def next_page(self):
        """下一页"""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.load_data()

    def last_page(self):
        """最后一页"""
        self.current_page = self.total_pages
        self.load_data()


if __name__ == "__main__":
    app = QApplication([])
    window = AdminUI()
    window.show()
    app.exec_()
