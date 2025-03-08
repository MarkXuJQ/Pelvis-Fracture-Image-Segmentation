from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QDateTimeEdit, QComboBox, QFrame
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QDateTime, QTimer


class TaskDetailsWidget:
    def __init__(self,parent,item,rightLayout):
        super().__init__()
        self.chat_app = parent
        self.user_id = self.chat_app.sender_id
        self.task_title = item.text()
        self.task_id = item.data(Qt.UserRole)
        self.sio = self.chat_app.sio
        self.rightLayout = rightLayout
        self.init_ui()
        self.sio.on('task_details', self.on_task_details_received)
        self.sio.on('task_updated', self.on_task_updated)

        self.fetch_task_details()  # 直接请求任务详情

    def init_ui(self):
        """初始化界面布局"""
        # 任务标题（不可修改）
        self.title_label = QLabel("任务：加载中...")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.rightLayout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        # 任务描述（可修改）
        self.description_label = QLabel("任务描述：")
        self.description_edit = QTextEdit()
        self.description_edit.setReadOnly(False)  # **确保可编辑**
        self.rightLayout.addWidget(self.description_label)
        self.rightLayout.addWidget(self.description_edit)

        # 截止日期（可修改）
        self.due_date_label = QLabel("截止日期：")
        self.due_date_edit = QDateTimeEdit()
        self.due_date_edit.setCalendarPopup(True)
        self.rightLayout.addWidget(self.due_date_label)
        self.rightLayout.addWidget(self.due_date_edit)

        # 病人 ID（不可修改）
        self.patient_label = QLabel("病人 ID：加载中...")
        self.rightLayout.addWidget(self.patient_label)

        # 当前医生任务状态（可修改）
        self.status_label = QLabel("您的任务状态：")
        self.status_combo = QComboBox()
        self.status_combo.addItems(["pending", "in_progress", "completed"])
        self.rightLayout.addWidget(self.status_label)
        self.rightLayout.addWidget(self.status_combo)

        # 分隔线
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        self.rightLayout.addWidget(self.separator)

        # 其他医生任务状态
        self.other_doctors_label = QLabel("📌 其他医生任务状态：加载中...")
        self.rightLayout.addWidget(self.other_doctors_label)

        # 保存按钮
        self.save_button = QPushButton("保存修改")
        self.save_button.clicked.connect(self.save_changes)
        self.rightLayout.addWidget(self.save_button, alignment=Qt.AlignCenter)

    def save_changes(self):
        """保存任务修改"""
        updated_data = {
            'task_id': self.task_id,
            'task_description': self.description_edit.toPlainText(),
            'due_date': self.due_date_edit.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            'status': self.status_combo.currentText(),
            'assigned_doctor_id': self.user_id
        }
        # 发送更新请求到服务器
        self.sio.emit('update_task_details', updated_data)

    def on_task_updated(self, data):
        assigned_doctor_id = data.get('assigned_doctor_id')
        task_id = data.get('task_id')

        # 仅在 assigned_doctor_id 匹配当前用户时更新任务列表
        if assigned_doctor_id == self.user_id:
            print(f"🔄 任务 {task_id} 更新，正在刷新任务列表 for 医生 {assigned_doctor_id}")
            self.fetch_task_details()

    def fetch_task_details(self):
        """通过 Socket 请求任务详情并更新 UI"""
        data = {
            'assigned_doctor_id': self.user_id,
            'task_title' : self.task_title
        }
        self.sio.emit('get_task_details', data)  # 发送任务请求
        QTimer.singleShot(1000, self.display)

    def display(self):
        self.description_edit.setPlainText(str(self.saved_description))
        q_due_date = QDateTime.fromString(self.saved_due_date, "yyyy-MM-dd HH:mm:ss")
        # 如果解析失败，使用当前时间
        if not q_due_date.isValid():
            print(f"解析日期失败: {self.saved_due_date}，改用当前时间")
            q_due_date = QDateTime.currentDateTime()
        self.due_date_edit.setDateTime(q_due_date)

    def on_task_details_received(self, data):
        """处理服务器返回的任务详情，并更新 UI"""
        print(f"✅ 收到任务详情: {data}")
        # 1. 检查 data 是否为空
        if not data:
            print("❌ 错误: 服务器返回的任务详情为空")
            return
        # 2. 任务标题
        task_title = data.get('task_title', '未知任务')
        title_text = f"任务：{task_title}"
        self.title_label.clear()
        self.title_label.setText(title_text)
        # 3. 任务描述
        task_description = data.get('task_description', None)
        if not task_description:
            task_description = "(无描述)"
        else:
            task_description = str(task_description)
        # 存储内容到 self 的属性里
        self.saved_description = task_description
        # 获取服务器返回的 due_date
        due_date = data.get('due_date', '无截止日期')
        # 不更新 UI，只存储
        self.saved_due_date = due_date
        # 5. 病人 ID
        patient_id = data.get('patient_id', '未知')
        self.patient_label.clear()
        self.patient_label.setText(f"病人 ID：{patient_id}")
        # 6. 其他医生任务状态
        # 这里把所有医生的状态拼成一大段文字
        doctor_status_list = data.get('tasks', [])
        other_doctors_text = "\n📌 其他医生任务状态：\n"
        for doctor in doctor_status_list:
            doctor_id = doctor.get('assigned_doctor_id', '???')
            doctor_name = doctor.get('doctor_name', '未知医生')
            status = doctor.get('status', '')
            if status == "completed":
                status_str = "✅ 已完成"
            else:
                status_str = "❌ 未完成"
            other_doctors_text += f"- {doctor_name} ({doctor_id}): {status_str}\n"
            # 如果医生 ID 等于当前用户 ID，则设置状态选择框
            if doctor_id == str(self.user_id):
                index = self.status_combo.findText(status)  # 查找状态索引
                if index >= 0:
                    self.status_combo.setCurrentIndex(index)  # 选中当前状态
                    print(f"📌 [DEBUG] 任务状态更新: 医生 {doctor_id} 状态为 {status}")
        # 显示拼接好的结果
        self.other_doctors_label.clear()
        self.other_doctors_label.setText(other_doctors_text)





