from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QComboBox, QTextEdit, QPushButton, QDateEdit, QHBoxLayout, \
    QListWidget, QListWidgetItem, QMessageBox
from PyQt5.QtCore import QDate, Qt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from database.db_config import db_config

# 创建数据库连接
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
Session = sessionmaker(bind=engine)
session = Session()

class FractureHistoryDialog(QDialog):
    def __init__(self, patient_name, patient_id,fracture_date, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"骨折信息 - {patient_name}")
        self.patient_id = patient_id  # 存储病人 ID
        self.history_id = None  # 记录是否已有历史记录
        self.fracture_date = fracture_date
        self.parent = parent
        self.init_ui()
        self.load_existing_data()  # 预加载已有骨折记录
        self.load_patient_history_list()

    def init_ui(self):
        main_layout = QHBoxLayout()  # 使用水平布局，左侧是选择框，右侧是表单
        # 🔹 左侧骨折日期列表
        list_layout = QVBoxLayout()
        self.history_list = QListWidget()  # 列表显示病人的骨折日期
        self.history_list.itemClicked.connect(self.on_history_selected)  # 绑定点击事件
        list_layout.addWidget(QLabel("历史记录（点击选择日期）:"))
        list_layout.addWidget(self.history_list)
        # 🔹 右侧骨折详情表单
        form_layout = QVBoxLayout()
        # 看病日期选择框
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        if self.fracture_date:
            self.date_edit.setDate(QDate.fromString(self.fracture_date, "yyyy-MM-dd"))
        else:
            self.date_edit.setDate(QDate.currentDate())
        # 骨折部位选择框
        self.location_combo = QComboBox()
        self.location_combo.addItems(["pelvis", "femur", "spine", "other"])
        # 严重程度选择框
        self.severity_combo = QComboBox()
        self.severity_combo.addItems(["mild", "moderate", "severe"])
        # 诊断详情输入框
        self.diagnosis_text = QTextEdit()
        # 提交按钮
        self.submit_button = QPushButton("提交")
        self.submit_button.clicked.connect(self.submit_fracture_history)
        # 将控件添加到表单布局
        form_layout.addWidget(QLabel("看病日期:"))
        form_layout.addWidget(self.date_edit)
        form_layout.addWidget(QLabel("骨折部位:"))
        form_layout.addWidget(self.location_combo)
        form_layout.addWidget(QLabel("严重程度:"))
        form_layout.addWidget(self.severity_combo)
        form_layout.addWidget(QLabel("诊断详情:"))
        form_layout.addWidget(self.diagnosis_text)
        form_layout.addWidget(self.submit_button)

        # 🔹 将左右布局合并
        main_layout.addLayout(list_layout)  # 左侧骨折历史
        main_layout.addLayout(form_layout)  # 右侧骨折表单

        self.setLayout(main_layout)

    def load_existing_data(self):
        """使用 SQLAlchemy 根据 patient_id 和 fracture_date 查询指定的骨折记录"""
        try:
            session = Session()
            # 获取当前 UI 选择的骨折日期
            selected_date = self.date_edit.date().toString("yyyy-MM-dd")

            # 查询该病人在指定日期的骨折记录
            history_query = text("""
                SELECT history_id, fracture_date, fracture_location, severity_level, diagnosis_details
                FROM fracturehistories 
                WHERE patient_id = :patient_id AND fracture_date = :fracture_date
                LIMIT 1
            """)
            history_result = session.execute(history_query,
                                             {"patient_id": self.patient_id, "fracture_date": selected_date}).fetchone()

            if history_result:
                # 解析查询结果
                self.history_id = history_result[0]  # history_id
                fracture_date = history_result[1] if history_result[1] else None
                fracture_location = history_result[2] if history_result[2] else "pelvis"
                severity_level = history_result[3] if history_result[3] else "mild"
                diagnosis_details = history_result[4] if history_result[4] else ""

                # 设置 UI 控件的值
                if fracture_date:
                    self.date_edit.setDate(QDate.fromString(str(fracture_date), "yyyy-MM-dd"))
                self.location_combo.setCurrentText(fracture_location)
                self.severity_combo.setCurrentText(severity_level)
                self.diagnosis_text.setPlainText(diagnosis_details)
            else:
                print(f"病人 {self.patient_id} 在 {selected_date} 没有骨折记录")
                self.history_id = None  # 没有历史记录时，确保 history_id 为空

        except Exception as e:
            print(f"数据库查询错误: {e}")


    def submit_fracture_history(self):
        """提交或更新骨折记录"""
        fracture_info = {
            "patient_id": self.patient_id,
            "fracture_date": self.date_edit.date().toString("yyyy-MM-dd"),
            "fracture_location": self.location_combo.currentText(),
            "severity_level": self.severity_combo.currentText(),
            "diagnosis_details": self.diagnosis_text.toPlainText()
        }
        try:
            # 调用插入/更新方法
            is_new_record,message = self.insert_or_update_fracture_history(fracture_info)
            # 显示消息提示框
            QMessageBox.information(None, "成功", message)
            # 触发信号通知 `DoctorUI` 更新表格
            self.parent.load_data_from_database()
            # 如果是插入新的病历，删除 `patientList` 里旧的项
            if is_new_record:
                self.remove_old_patient_item()

            self.accept()  # 关闭对话框

        except Exception as e:
            print(f"❌ 数据库错误: {e}")
            QMessageBox.critical(None, "错误", f"数据库错误: {e}")

    def insert_or_update_fracture_history(self, fracture_info):
        """检查 fracture_date 是否存在，并执行插入或更新"""
        session = Session()
        try:
            # 检查当前病人是否在该日期已有记录
            check_query = text("""
                SELECT history_id FROM fracturehistories
                WHERE patient_id = :patient_id AND fracture_date = :fracture_date
            """)
            existing_history_id = session.execute(check_query, {
                "patient_id": fracture_info["patient_id"],
                "fracture_date": fracture_info["fracture_date"]
            }).scalar()

            if existing_history_id:  # 如果已存在记录，则更新
                update_query = text("""
                    UPDATE fracturehistories
                    SET fracture_location = :fracture_location,
                        severity_level = :severity_level,
                        diagnosis_details = :diagnosis_details
                    WHERE history_id = :history_id
                """)
                session.execute(update_query, {
                    "fracture_location": fracture_info["fracture_location"],
                    "severity_level": fracture_info["severity_level"],
                    "diagnosis_details": fracture_info["diagnosis_details"],
                    "history_id": existing_history_id
                })
                session.commit()
                return False,f"✅ 已更新病人 {fracture_info['patient_id']} 在 {fracture_info['fracture_date']} 的骨折记录"

            else:  # ✅ 如果不存在，则插入新记录
                last_id_query = text("""
                    SELECT history_id FROM fracturehistories
                    ORDER BY history_id DESC
                    LIMIT 1
                """)
                last_id = session.execute(last_id_query).scalar()

                # 如果没有历史 ID，则从 "F00001" 开始
                if last_id:
                    last_number = int(last_id[1:])
                    new_number = str(last_number + 1).zfill(5)
                    new_history_id = f"F{new_number}"
                else:
                    new_history_id = "F00001"

                insert_query = text("""
                    INSERT INTO fracturehistories (history_id, patient_id, fracture_date, 
                                                  fracture_location, severity_level, diagnosis_details)
                    VALUES (:history_id, :patient_id, :fracture_date, :fracture_location, 
                            :severity_level, :diagnosis_details)
                """)
                session.execute(insert_query, {
                    "history_id": new_history_id,
                    "patient_id": fracture_info["patient_id"],
                    "fracture_date": fracture_info["fracture_date"],
                    "fracture_location": fracture_info["fracture_location"],
                    "severity_level": fracture_info["severity_level"],
                    "diagnosis_details": fracture_info["diagnosis_details"]
                })
                session.commit()
                return True,f"✅ 新骨折记录已添加，ID: {new_history_id}"

        except Exception as e:
            session.rollback()
            print(f"❌ Error inserting or updating fracture history: {e}")
            raise e
        finally:
            session.close()

    def load_patient_history_list(self):
        """加载该病人的所有骨折记录日期到 QListWidget"""
        try:
            session = Session()
            history_query = text("""
                SELECT history_id, fracture_date FROM fracturehistories 
                WHERE patient_id = :patient_id
                ORDER BY fracture_date DESC
            """)
            results = session.execute(history_query, {"patient_id": self.patient_id}).fetchall()

            self.history_list.clear()  # 清空旧数据

            if results:
                for history_id, fracture_date in results:
                    item_text = f"{fracture_date}"  # 只显示日期
                    item = QListWidgetItem(item_text)
                    item.setData(32, history_id)  # 绑定 history_id 到 QListWidgetItem
                    self.history_list.addItem(item)

            else:
                self.history_list.addItem("无历史记录")

        except Exception as e:
            print(f"数据库查询错误: {e}")

    def on_history_selected(self, item):
        """当医生点击病人的某个历史记录时，加载对应的骨折信息"""
        session = Session()
        history_id = item.data(32)  # 获取绑定的 history_id

        try:
            history_query = text("""
                SELECT fracture_date, fracture_location, severity_level, diagnosis_details
                FROM fracturehistories 
                WHERE history_id = :history_id
            """)
            record = session.execute(history_query, {"history_id": history_id}).fetchone()

            if record:
                fracture_date, fracture_location, severity_level, diagnosis_details = record

                self.date_edit.setDate(QDate.fromString(str(fracture_date), "yyyy-MM-dd"))
                self.location_combo.setCurrentText(fracture_location)
                self.severity_combo.setCurrentText(severity_level)
                self.diagnosis_text.setPlainText(diagnosis_details)
                print(f"加载病人 {self.patient_id} 在 {fracture_date} 的骨折记录")

        except Exception as e:
            print(f"数据库查询错误: {e}")

    def remove_old_patient_item(self):
        """从 `patientList` 中删除当前病人的旧项"""
        try:
            existing_items = []
            for i in range(self.parent.patientList.count()):
                item = self.parent.patientList.item(i)
                existing_patient_id, _ = item.data(Qt.UserRole)
                if existing_patient_id == self.patient_id:  # **如果病人 ID 相同，则标记为删除**
                    existing_items.append(item)

            if existing_items:
                for existing_item in existing_items:
                    row_index = self.parent.patientList.row(existing_item)

                    # **直接调用 `remove()` 处理删除逻辑**
                    self.parent.delegate.remove(row_index)

                    # **断开 `itemClicked` 连接，防止重复绑定**
                    self.parent.patientList.itemClicked.disconnect(self.parent.on_patient_item_clicked)

            print(f"✅ 已从 `patientList` 删除病人 {self.patient_id} 的旧项")

        except Exception as e:
            print(f"❌ 删除 `patientList` 旧项时发生错误: {e}")
