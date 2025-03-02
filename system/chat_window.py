import sys
import json

from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
    QApplication, QListView, QWidget, QVBoxLayout, QLabel, QPushButton, QMenu, QStyleOptionViewItem, QStyledItemDelegate
)
from PyQt5.QtCore import Qt, QSize
import socketio
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QListWidget, QTabWidget, \
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QFileDialog, QHBoxLayout, QLabel, QListWidgetItem, QMenu, \
    QAction
from qasync import QEventLoop
import asyncio
from delegate import  TaskItemDelegate


class ChatApp(QMainWindow):
    def __init__(self,user_id):
        super().__init__()
        #self.apply_stylesheet()
        self.user_id = user_id
        self.sio = socketio.Client()
        self.sio.connect('http://localhost:5000')
        self.sender_id = user_id
        self.receiver_id = None
        self.chatlayout()

        # 设置 Socket.IO 事件
        self.sio.on('receive_message', self.on_receive_message)
        self.sio.on('chat_history', self.on_chat_history)
        self.sio.on('task_created', self.on_task_created)
        self.sio.on('task_list',self.on_task_list)

    def chatlayout(self):
        self.setWindowTitle('聊天信息界面')
        self.setGeometry(0, 0, 1200, 800)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.left_widget = QWidget(self)
        self.left_layout = QVBoxLayout(self.left_widget)

        self.tabs_list = QListWidget(self)
        self.tabs_list.addItem("消息")
        self.tabs_list.addItem("任务")
        self.tabs_list.addItem("文档")
        self.tabs_list.addItem("笔记")
        self.tabs_list.itemClicked.connect(self.switch_to_tab)

        self.left_layout.addWidget(self.tabs_list)

        self.center_widget = QWidget(self)
        self.center_layout = QVBoxLayout(self.center_widget)

        self.welcome_label = QLabel("欢迎使用！请选择一个功能或标签。", self)
        self.center_layout.addWidget(self.welcome_label)

        self.right_widget = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_widget)

        self.right_content = QWidget(self)
        self.right_layout.addWidget(self.right_content)

        # 布局设置
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.addWidget(self.left_widget)
        main_layout.setStretchFactor(self.left_widget, 2)
        main_layout.addWidget(self.center_widget)
        main_layout.setStretchFactor(self.center_widget, 4)
        main_layout.addWidget(self.right_widget)
        main_layout.setStretchFactor(self.right_widget, 5)

    def switch_to_tab(self, item):
        selected_tab = item.text()
        if selected_tab == "消息":
            self.load_doctor_list()
        elif selected_tab == "任务":
            self.load_task_list()
        elif selected_tab == "文档":
            self.display_document_page()
        elif selected_tab == "笔记":
            self.display_note_page()


    def display_task_page(self,item):
        # 获取点击任务的标题
        selected_task_title = item.text()

        # 通过任务标题获取任务 ID（或者可以直接通过标题传递任务 ID）
        task_id = selected_task_title.split(":")[0]  # 假设任务列表项的格式是 'task_id: task_title'

        # 发送请求以获取任务详细信息
        data = {
            'task_id': task_id
        }
        print(f"Requesting task details for task_id: {task_id}")
        self.sio.emit('get_task_details', data)  # 向服务器请求任务详情


    def load_doctor_list(self):
        self.welcome_label.setText("医生列表：请选择一个医生")
        self.remove_list_widget()
        self.doctor_list = QListWidget(self)
        self.doctor_list.addItem("Doctor John")
        self.doctor_list.addItem("Doctor Jane")
        self.doctor_list.itemClicked.connect(self.load_chat_history)
        self.center_layout.addWidget(self.doctor_list)

    def on_receive_message(self, data):
        if data.get('sender_id') == self.receiver_id:
            self.chat_area.append(f"Doctor2: {data['message']}")

    def send_message(self):
        message = self.message_input.text()
        if message:
            self.chat_area.append(f"Me: {message}")
            data = {
                'sender_id': self.sender_id,
                'receiver_id': self.receiver_id,
                'message': message
            }
            self.sio.emit('send_message', data)  # 发送消息
            self.message_input.clear()

    def load_chat_history(self):
        self.clear_right_layout()

        self.chat_area = QTextEdit(self)
        self.chat_area.setGeometry(220, 10, 560, 400)
        self.chat_area.setReadOnly(True)

        self.message_input = QLineEdit(self)
        self.message_input.setGeometry(220, 420, 460, 40)

        self.send_button = QPushButton("Send", self)
        self.send_button.setGeometry(690, 420, 90, 40)
        self.send_button.clicked.connect(self.send_message)
        self.right_layout.addWidget(self.chat_area)
        self.right_layout.addWidget(self.message_input)
        self.right_layout.addWidget(self.send_button)
        # 获取聊天记录时，选择的是哪个医生
        selected_doctor = self.doctor_list.selectedItems()[0].text()
        if selected_doctor == "Doctor John":
            self.receiver_id = 2
        elif selected_doctor == "Doctor Jane":
            self.receiver_id = 3
        data = {
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id
        }
        self.sio.emit('get_chat_history', data)

    def on_chat_history(self, data):
        chat_history = data['history']
        for message in chat_history:
            sender = 'Me' if message['sender_id'] == self.sender_id else 'Doctor2'
            self.chat_area.append(f"{sender}: {message['message_content']}")

    def load_task_list(self):
        self.welcome_label.setText("任务列表：请选择一个任务")
        self.remove_list_widget()
        self.clear_right_layout()
        self.task_list = QListWidget(self)
        self.center_layout.addWidget(self.task_list)

        # Set custom delegate for the task list
        delegate = TaskItemDelegate(self.task_list,self.sio)
        self.task_list.setItemDelegate(delegate)

        self.assigned_doctor_id = self.user_id
        data = {
            'assigned_doctor_id': self.assigned_doctor_id
        }
        self.sio.emit('get_task_list', data)

    def on_task_list(self,data):
        task_list_data = data['tasks']
        self.task_list.clear()
        for task in task_list_data:
            task_title = task['task_title']
            task_id = task['task_id']
            # Add task title to the list
            list_item = QListWidgetItem(task_title)
            list_item.setData(Qt.UserRole, task_id)  # Store the task ID as custom data
            self.task_list.addItem(list_item)

    def remove_list_widget(self):
        """删除 center_layout 中的 QListWidget 组件"""
        for i in range(self.center_layout.count()):
            item = self.center_layout.itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, QListWidget):
                widget.deleteLater()

    def clear_right_layout(self):
        """删除 right_layout 中的所有组件，但保留布局本身"""
        for i in range(self.right_layout.count()):
            item = self.right_layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    def create_task(self):
        task_title = "New Task"
        task_description = "Task Description"
        assigned_doctor_id = 2  # 指定任务给某个医生
        data = {
            'task_title': task_title,
            'task_description': task_description,
            'assigned_doctor_id': assigned_doctor_id,
            'sender_id': self.sender_id
        }
        self.sio.emit('create_task', data)

    def on_task_created(self, data):
        # 任务创建成功后更新 UI
        task = data['task']
        row_position = self.tasks_table.rowCount()
        self.tasks_table.insertRow(row_position)
        self.tasks_table.setItem(row_position, 0, QTableWidgetItem(task['task_title']))
        self.tasks_table.setItem(row_position, 1, QTableWidgetItem(task['task_description']))
        self.tasks_table.setItem(row_position, 2, QTableWidgetItem(task['assigned_doctor_id']))

    def display_document_page(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            # 选择文件上传逻辑
            document_data = {'file_paths': file_paths, 'patient_id': self.receiver_id}
            self.sio.emit('upload_document', document_data)

    def add_note(self):
        note_content = self.note_input.text()
        note_data = {'note_content': note_content, 'patient_id': self.receiver_id}
        self.sio.emit('add_note', note_data)

    def display_note_page(self):
        """显示笔记页面"""
        self.welcome_label.setText("笔记管理")
        self.clear_right_layout()
        
        # 创建笔记输入区域
        self.note_input = QTextEdit(self)
        self.note_input.setPlaceholderText("在这里输入笔记...")
        
        # 创建保存按钮
        save_button = QPushButton("保存笔记", self)
        save_button.clicked.connect(self.add_note)
        
        # 添加到布局
        self.right_layout.addWidget(self.note_input)
        self.right_layout.addWidget(save_button)

    def start(self):
        self.sio.connect('http://localhost:5000')
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

if __name__ == "__main__":
    app = QApplication(sys.argv)

    loop = QEventLoop(app)  # Use QEventLoop with PyQt5 to handle asyncio tasks
    asyncio.set_event_loop(loop)  # Set the event loop to be used

    window = ChatApp(1)
    window.show()

    loop.run_forever() # Start the event loop
    sys.exit(app.exec_())


