from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QTextEdit, QHBoxLayout, \
    QMessageBox
from PyQt5.QtCore import Qt, QTimer
from functools import partial
from system.stylesheet import apply_stylesheet

class NoteDetailsWidget(QWidget):
    def __init__(self, parent, item, rightLayout):
        super().__init__()
        self.parent = parent
        self.task_id = item.data(Qt.UserRole)
        self.task_title = item.text()
        self.user_id = self.parent.sender_id
        self.sio = self.parent.sio
        self.rightLayout = rightLayout
        self.notes_by_doctor = {}  # 存储医生的笔记
        self.selected_doctor_id = None  # 当前查看的医生 ID
        self.selected_doctor_name = None
        self.init_ui()
        self.sio.on("doctor_notes", self.display_note_viewer)
        self.sio.on("task_notes", self.get_notes_data)

    def init_ui(self):
        """初始化右侧任务笔记 UI（只添加组件，不添加布局）"""
        # 滚动区域，用于展示多个医生的笔记
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # 滚动内容容器
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_layout.setAlignment(Qt.AlignTop)

        self.scroll_area.setWidget(self.scroll_content)
        self.rightLayout.addWidget(self.scroll_area)
        # 创建底部固定的按钮区域
        self.bottom_widget = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_widget)
        # 创建“添加笔记”按钮
        self.add_note_button = QPushButton("➕ 添加笔记")
        self.add_note_button.clicked.connect(self.show_note_input)
        self.bottom_layout.addWidget(self.add_note_button, alignment=Qt.AlignCenter)

        self.rightLayout.addWidget(self.bottom_widget)
        self.sio.emit("get_task_notes", {"task_id": self.task_id})
        QTimer.singleShot(1000, self.update_notes_ui)

    def get_notes_data(self, data):
        """解析服务器返回的任务笔记数据，并存储到 `self.notes_by_doctor`"""
        notes_by_doctor = data.get("notes_by_doctor", {})
        if not isinstance(notes_by_doctor, dict):
            self.notes_by_doctor = {}
            return
        # 存储笔记数据
        self.notes_by_doctor = notes_by_doctor

    def update_notes_ui(self):
        """清空旧的 UI 并重新加载医生的笔记"""
        self.scroll_layout.setAlignment(Qt.AlignTop)
        task_title_label = QLabel(f"📌 任务: {self.task_title}")
        task_title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        self.scroll_layout.addWidget(task_title_label)
        # 确保至少有一个医生，否则显示 "暂无笔记"
        if not self.notes_by_doctor:
            empty_label = QLabel("暂无笔记")
            self.scroll_layout.addWidget(empty_label)
            self.scroll_content.update()
            return
        # 遍历医生的笔记，确保notes是列表
        for doctor_id, doctor_data in self.notes_by_doctor.items():
            doctor_name = doctor_data.get("doctor_name", "未知医生")
            notes = doctor_data.get("notes", [])
            if not isinstance(notes, list) or not notes:
                QMessageBox.warning(None, "注意", f"医生 {doctor_name} 没有有效笔记！")
                continue
            if not isinstance(notes[0], dict):
                QMessageBox.warning(None, "注意", f"医生 {doctor_name} 的笔记格式错误！")
                continue
            doctor_frame = QFrame()
            doctor_layout = QVBoxLayout(doctor_frame)
            # 医生名字
            display_name = "👨‍⚕️ Me" if int(doctor_id) == self.user_id else f"👨‍⚕️ 医生 {doctor_name}"
            doctor_label = QLabel(display_name)
            doctor_layout.addWidget(doctor_label)
            # 显示该医生的第一条笔记
            note_preview = QLabel(f"{notes[0].get('content', '无内容')[:30]}...")
            doctor_layout.addWidget(note_preview)
            # "查看" 按钮
            view_button = QPushButton("查看")
            view_button.clicked.connect(partial(self.open_note_viewer, doctor_id,doctor_name))
            doctor_layout.addWidget(view_button)

            doctor_frame.setLayout(doctor_layout)
            self.scroll_layout.addWidget(doctor_frame)
        # 强制更新 UI
        self.scroll_layout.update()
        self.bottom_layout.update()

    def show_note_input(self):
        """点击添加笔记后，清空右侧布局并显示输入框和保存按钮"""
        self.parent.clear_right_layout()
        self.note_input = QTextEdit(self)
        self.note_input.setPlaceholderText("在这里输入笔记...")
        self.note_input.setStyleSheet("border: 1px solid #ccc; padding: 5px; font-size: 14px;")
        #创建保存按钮
        save_button = QPushButton("💾 保存笔记", self)
        save_button.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 5px;")
        save_button.clicked.connect(self.add_note)
        self.rightLayout.addWidget(QLabel("✏️ 新建笔记："), alignment=Qt.AlignLeft)
        self.rightLayout.addWidget(self.note_input)
        self.rightLayout.addWidget(save_button, alignment=Qt.AlignRight)

    def add_note(self):
        """将新笔记发送到服务器"""
        note_content = self.note_input.toPlainText().strip()
        if not note_content:
            QMessageBox.warning(None, "注意", "笔记内容不能为空！")
            return
        # 发送新笔记到服务器
        self.sio.emit("add_task_note", {
            "task_id": self.task_id,
            "doctor_id": self.user_id,
            "content": note_content
        })
        self.parent.clear_right_layout()

    def open_note_viewer(self, doctor_id,doctor_name):
        """使用 Socket.IO 获取医生的笔记并显示在窗口中"""
        self.selected_doctor_id = doctor_id  # 记录当前医生 ID
        self.selected_doctor_name = "Me" if int(doctor_id) == self.user_id else doctor_name
        # 发送请求获取医生的笔记
        self.sio.emit("get_doctor_notes", {"task_id": self.task_id, "doctor_id": doctor_id})
        QTimer.singleShot(1000, self.operate)

    def display_note_viewer(self, data):
        """收到服务器返回的医生笔记数据后，更新 UI"""
        self.one_doctor_notes = data.get("notes", [])

    def operate(self):
        self.note_viewer = NoteViewer(self.one_doctor_notes,self.selected_doctor_name)
        self.note_viewer.show()

class NoteViewer(QWidget):
    def __init__(self, notes,doctor_name):
        super().__init__()
        apply_stylesheet(self)
        self.notes = notes
        self.doctor_name = doctor_name
        self.current_page = 0
        self.init_ui()

    def init_ui(self):
        """初始化医生笔记详情窗口"""
        self.setWindowTitle(f"医生 {self.doctor_name} 的笔记详情")
        self.setGeometry(200, 200, 600, 300)
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        # 医生名字
        self.doctor_label = QLabel(f"👨‍⚕️ 医生 {self.doctor_name}")
        self.doctor_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(self.doctor_label, alignment=Qt.AlignLeft)
        # 笔记内容框
        self.note_content = QTextEdit()
        self.note_content.setReadOnly(True)
        self.note_content.setFont(QFont("Arial", 11))
        self.note_content.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
        self.layout.addWidget(self.note_content)
        # 笔记创建时间
        self.created_at_label = QLabel("")
        self.created_at_label.setFont(QFont("Arial", 10))
        self.created_at_label.setStyleSheet("color: gray;")
        self.created_at_label.setAlignment(Qt.AlignRight)
        # 创建水平布局
        time_layout = QHBoxLayout()
        time_layout.addStretch(1)
        time_layout.addWidget(self.created_at_label)
        self.layout.addLayout(time_layout)
        navigation_layout = QHBoxLayout()
        # 上一条
        self.prev_label = QLabel("⏪")
        self.prev_label.setFont(QFont("Arial", 20))
        self.prev_label.setStyleSheet("color: blue; cursor: pointer;")
        self.prev_label.setToolTip("上一条")
        navigation_layout.addWidget(self.prev_label, alignment=Qt.AlignLeft)
        # 占位符
        navigation_layout.addStretch(1)
        # 下一条
        self.next_label = QLabel("⏩")
        self.next_label.setFont(QFont("Arial", 20))
        self.next_label.setStyleSheet("color: blue; cursor: pointer;")
        self.next_label.setToolTip("下一条")
        navigation_layout.addWidget(self.next_label, alignment=Qt.AlignRight)

        self.next_label.mousePressEvent = lambda event: self.next_page(event)
        self.prev_label.mousePressEvent = lambda event: self.previous_page(event)
        # 添加hover效果
        self.prev_label.enterEvent = lambda event: self.prev_label.setStyleSheet("color: darkblue; cursor: pointer;")
        self.prev_label.leaveEvent = lambda event: self.prev_label.setStyleSheet("color: blue; cursor: pointer;")

        self.next_label.enterEvent = lambda event: self.next_label.setStyleSheet("color: darkblue; cursor: pointer;")
        self.next_label.leaveEvent = lambda event: self.next_label.setStyleSheet("color: blue; cursor: pointer;")

        self.layout.addLayout(navigation_layout)
        self.setLayout(self.layout)
        self.update_note()

    def update_note(self):
        """更新笔记内容"""
        if self.notes and 0 <= self.current_page < len(self.notes):
            note = self.notes[self.current_page]
            self.note_content.setPlainText(note["content"])
            self.created_at_label.setText(f"🕒 {note['created_at']}")

    def previous_page(self,event):
        """跳转到上一条笔记"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_note()

    def next_page(self,event):
        """跳转到下一条笔记"""
        if self.current_page < len(self.notes) - 1:
            self.current_page += 1
            self.update_note()



