import sys
import json

from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
    QApplication, QListView, QWidget, QVBoxLayout, QLabel, QPushButton, QMenu, QStyleOptionViewItem,
    QStyledItemDelegate, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize
import os
import socketio
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QListWidget, QTabWidget, \
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QFileDialog, QHBoxLayout, QLabel, QListWidgetItem, QMenu, \
    QAction
from qasync import QEventLoop
import asyncio
from delegate import  TaskItemDelegate
from stylesheet import apply_stylesheet

class ChatApp(QMainWindow):
    def __init__(self,user_id):
        super().__init__()
        apply_stylesheet(self)

        self.sio = socketio.Client()
        self.sio.connect('http://localhost:5000')
        self.sender_id = user_id
        self.receiver_id = None
        self.receiver_name = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(current_dir, "ui", "chat_window.ui")
        uic.loadUi(ui_file, self)
        self.adjustlayout()
        # 设置 Socket.IO 事件
        self.sio.on('receive_message', self.on_receive_message)
        self.sio.on('chat_history', self.on_chat_history)
        self.sio.on('task_created', self.on_task_created)
        self.sio.on('task_list',self.on_task_list)
        self.sio.on('doctors_list',self.on_doctor_list)

    def adjustlayout(self):
        # 假设这是主窗口的布局
        layout = self.findChild(QHBoxLayout, "mainChatLayout")
        layout.setStretchFactor(self.leftWidget, 2)
        layout.setStretchFactor(self.centerWidget, 4)
        layout.setStretchFactor(self.rightWidget, 5)
        # 为布局设置边距
        layout.setContentsMargins(10, 2, 10, 10)  # 左上右下的间距为10像素
        # 获取布局
        topLayout = self.findChild(QHBoxLayout, "topLayout")
        # 为布局设置边距
        topLayout.setContentsMargins(20, 10, 10, 5)  # 左上右下的间距为10像素

        # 创建一个新的 QSpacerItem，并手动添加到布局
        horizontalSpacer = QSpacerItem(30, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        topLayout.addItem(horizontalSpacer)
        self.tabsList = self.findChild(QListWidget, "tabsList")
        self.tabsList.itemClicked.connect(self.switch_to_tab)
        self.welcomeLabel = self.findChild(QLabel, "welcomeLabel")

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
        self.welcomeLabel.setText("消息")
        self.remove_list_widget()
        self.doctor_list = QListWidget(self)
        self.centerLayout.addWidget(self.doctor_list)
        data = {'user_id': self.sender_id}
        self.sio.emit('get_doctors_except_user', data)
        self.doctor_list.itemClicked.connect(self.load_chat_history)

    def on_doctor_list(self,data):
        doctors = data.get('doctors', [])  # 获取医生列表
        for doctor in doctors:
            doctor_id = doctor['doctor_id']
            doctor_name = doctor['doctor_name']
            item = QListWidgetItem(doctor_name)
            item.setData(Qt.UserRole, (doctor_id,doctor_name))  # 绑定 doctor_id
            self.doctor_list.addItem(item)

    def on_receive_message(self, data):
        if data.get('sender_id') == self.receiver_id:
            self.chat_area.append(f"{self.receiver_name}: {data['message']}")

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

    def load_chat_history(self,item):
        self.clear_right_layout()

        self.chat_area = QTextEdit(self)
        self.chat_area.setGeometry(220, 10, 560, 400)
        self.chat_area.setReadOnly(True)

        self.message_input = QLineEdit(self)
        self.message_input.setGeometry(220, 420, 460, 40)

        self.send_button = QPushButton("Send", self)
        self.send_button.setGeometry(690, 420, 90, 40)
        self.send_button.clicked.connect(self.send_message)
        self.rightLayout.addWidget(self.chat_area)
        self.rightLayout.addWidget(self.message_input)
        self.rightLayout.addWidget(self.send_button)
        # 获取聊天记录时，选择的是哪个医生

        self.receiver_id,self.receiver_name = item.data(Qt.UserRole)
        print(self.receiver_name)
        data = {
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id
        }
        self.sio.emit('get_chat_history', data)

    def on_chat_history(self, data):
        chat_history = data['history']
        for message in chat_history:
            sender = 'Me' if message['sender_id'] == self.sender_id else self.receiver_name
            self.chat_area.append(f"{sender}: {message['message_content']}")

    def load_task_list(self):
        self.welcomeLabel.setText("任务")
        self.remove_list_widget()
        self.clear_right_layout()
        self.task_list = QListWidget(self)
        self.centerLayout.addWidget(self.task_list)
        # Set custom delegate for the task list
        delegate = TaskItemDelegate(self.task_list,self.sio)
        self.task_list.setItemDelegate(delegate)
        data = {
            'assigned_doctor_id': self.sender_id
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
        # 创建 "add task" 并设置为固定项
        add_task_item = QListWidgetItem("add task")
        add_task_item.setFlags(add_task_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)  # 禁止选中
        add_task_item.setFlags(add_task_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # 禁止编辑
        add_task_item.setFlags(add_task_item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)  # 禁止拖动
        self.task_list.addItem(add_task_item)  # 确保它在最后

        # 绑定 itemClicked 事件
        self.task_list.itemClicked.connect(self.on_task_clicked)
    def on_task_clicked(self, item):
        if item.text() == "add task":
            print("添加任务")
            #self.add_task()
        else:
            print("任务详情")
            #self.load_task_details(item)

    def remove_list_widget(self):
        """删除 center_layout 中的 QListWidget 组件"""
        for i in range(self.centerLayout.count()):
            item = self.centerLayout.itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, QListWidget):
                widget.deleteLater()

    def clear_right_layout(self):
        """删除 right_layout 中的所有组件，但保留布局本身"""
        for i in range(self.rightLayout.count()):
            item = self.rightLayout.itemAt(i)
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
        self.welcomeLabel.setText("笔记管理")
        self.clear_right_layout()
        
        # 创建笔记输入区域
        self.note_input = QTextEdit(self)
        self.note_input.setPlaceholderText("在这里输入笔记...")
        
        # 创建保存按钮
        save_button = QPushButton("保存笔记", self)
        save_button.clicked.connect(self.add_note)
        
        # 添加到布局
        self.rightLayout.addWidget(self.note_input)
        self.rightLayout.addWidget(save_button)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    loop = QEventLoop(app)  # Use QEventLoop with PyQt5 to handle asyncio tasks
    asyncio.set_event_loop(loop)  # Set the event loop to be used

    window = ChatApp(1)
    window.show()

    loop.run_forever() # Start the event loop
    sys.exit(app.exec_())


