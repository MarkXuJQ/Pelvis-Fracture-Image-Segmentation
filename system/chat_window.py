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
    def __init__(self):
        super().__init__()

        # 应用样式表
        #self.apply_stylesheet()

        # 创建 Socket.IO 客户端
        self.sio = socketio.Client()

        # 连接到服务器
        self.sio.connect('http://localhost:5000')

        self.sender_id = 1  # 示例发送者ID（医生ID）
        self.receiver_id = None  # 将在点击聊天列表时动态设置接收者ID


        self.newlayout()
        # 创建聊天界面
        #self.create_chat_interface()

        # 创建任务管理、文档和笔记标签页
        #self.create_task_document_note_tabs()

        # 设置 Socket.IO 事件
        self.sio.on('receive_message', self.on_receive_message)
        self.sio.on('chat_history', self.on_chat_history)
        self.sio.on('task_created', self.on_task_created)
        #emit('chat_history', {'history': []})
        #emit('task_list', {'tasks': task_list_data}, broadcast=False)
        self.sio.on('task_list',self.on_task_list)

    def newlayout(self):
        self.setWindowTitle('聊天信息界面')
        self.setGeometry(0, 0, 1200, 800)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # 左侧：消息、任务、文档、笔记标签
        self.left_widget = QWidget(self)
        self.left_layout = QVBoxLayout(self.left_widget)

        # 标签列表（消息、任务、文档、笔记）
        self.tabs_list = QListWidget(self)
        self.tabs_list.addItem("消息")
        self.tabs_list.addItem("任务")
        self.tabs_list.addItem("文档")
        self.tabs_list.addItem("笔记")
        self.tabs_list.itemClicked.connect(self.switch_to_tab)

        self.left_layout.addWidget(self.tabs_list)

        # 中间区域：动态内容
        self.center_widget = QWidget(self)
        self.center_layout = QVBoxLayout(self.center_widget)

        # 初始显示内容
        self.welcome_label = QLabel("欢迎使用！请选择一个功能或标签。", self)
        self.center_layout.addWidget(self.welcome_label)


        # 右侧：显示医生列表或任务、文档、笔记等
        self.right_widget = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_widget)

        # 右侧初始内容（可以在后续根据选择切换）
        self.right_content = QWidget(self)
        self.right_layout.addWidget(self.right_content)

        # 布局设置
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.addWidget(self.left_widget)
        main_layout.setStretchFactor(self.left_widget, 2)  # 设置伸缩因子为 2
        main_layout.addWidget(self.center_widget)
        main_layout.setStretchFactor(self.center_widget, 4)  # 设置伸缩因子为 2
        main_layout.addWidget(self.right_widget)
        main_layout.setStretchFactor(self.right_widget, 5)  # 设置伸缩因子为 2

    def switch_to_tab(self, item):
        selected_tab = item.text()

        if selected_tab == "消息":
            # 点击消息时，显示医生列表
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
        # 显示医生列表
        self.welcome_label.setText("医生列表：请选择一个医生")
        self.remove_list_widget()
        # 这里可以加载医生列表，示例显示医生名称
        self.doctor_list = QListWidget(self)
        self.doctor_list.addItem("Doctor John")
        self.doctor_list.addItem("Doctor Jane")
        self.doctor_list.itemClicked.connect(self.load_chat_history)

        self.center_layout.addWidget(self.doctor_list)


    def create_task_document_note_tabs(self):
        # 创建标签页
        self.tabs = QTabWidget(self)
        self.tabs.setGeometry(220, 460, 560, 130)

        # 任务管理标签
        self.task_tab = QWidget()
        self.tasks_layout = QVBoxLayout()
        self.create_task_button = QPushButton("Create Task", self)
        self.create_task_button.clicked.connect(self.create_task)
        self.tasks_layout.addWidget(self.create_task_button)
        self.tasks_table = QTableWidget(self)
        self.tasks_layout.addWidget(self.tasks_table)
        self.task_tab.setLayout(self.tasks_layout)

        # 文档上传标签
        self.document_tab = QWidget()
        self.documents_layout = QVBoxLayout()
        self.upload_document_button = QPushButton("Upload Document", self)
        self.upload_document_button.clicked.connect(self.upload_document)
        self.documents_layout.addWidget(self.upload_document_button)
        self.documents_table = QTableWidget(self)
        self.documents_layout.addWidget(self.documents_table)
        self.document_tab.setLayout(self.documents_layout)

        # 笔记管理标签
        self.note_tab = QWidget()
        self.notes_layout = QVBoxLayout()
        self.note_input = QLineEdit(self)
        self.notes_layout.addWidget(self.note_input)
        self.add_note_button = QPushButton("Add Note", self)
        self.add_note_button.clicked.connect(self.add_note)
        self.notes_layout.addWidget(self.add_note_button)
        self.notes_table = QTableWidget(self)
        self.notes_layout.addWidget(self.notes_table)
        self.note_tab.setLayout(self.notes_layout)

        # 添加标签页
        self.tabs.addTab(self.task_tab, "Tasks")
        self.tabs.addTab(self.document_tab, "Documents")
        self.tabs.addTab(self.note_tab, "Notes")


    def on_receive_message(self, data):
        # 接收到消息后，将消息显示到聊天区域
        if data.get('sender_id') == self.receiver_id:
            self.chat_area.append(f"Patient: {data['message']}")

    def send_message(self):
        message = self.message_input.text()
        if message:
            self.chat_area.append(f"Doctor: {message}")
            data = {
                'sender_id': self.sender_id,
                'receiver_id': self.receiver_id,
                'message': message
            }
            self.sio.emit('send_message', data)  # 发送消息
            self.message_input.clear()

    def load_chat_history(self):
        print(11)
        self.clear_right_layout()
        print(666)
        # 聊天区
        self.chat_area = QTextEdit(self)
        self.chat_area.setGeometry(220, 10, 560, 400)
        self.chat_area.setReadOnly(True)
        print(666666)
        # 消息输入框
        self.message_input = QLineEdit(self)
        self.message_input.setGeometry(220, 420, 460, 40)

        # 发送按钮
        self.send_button = QPushButton("Send", self)
        self.send_button.setGeometry(690, 420, 90, 40)
        self.send_button.clicked.connect(self.send_message)
        print(222)
        self.right_layout.addWidget(self.chat_area)
        self.right_layout.addWidget(self.message_input)
        self.right_layout.addWidget(self.send_button)
        print(333)
        print("Damn")

        # 获取聊天记录时，选择的是哪个医生
        selected_doctor = self.doctor_list.selectedItems()[0].text()
        print(999)
        if selected_doctor == "Doctor John":
            print(00)
            self.receiver_id = 2  # 设置对应医生的接收者ID
        elif selected_doctor == "Doctor Jane":
            self.receiver_id = 3  # 设置对应医生的接收者ID
        print(444)
        # 请求聊天记录
        data = {
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id
        }
        print(222)
        self.sio.emit('get_chat_history', data)
        print(333)

    def on_chat_history(self, data):
        # 获取历史聊天记录并显示
        chat_history = data['history']
        print(12345)
        print(chat_history)
        for message in chat_history:
            print(89)
            sender = 'Doctor' if message['sender_id'] == self.sender_id else 'Patient'
            print(99)
            self.chat_area.append(f"{sender}: {message['message_content']}")
            print(00)

    def load_task_list(self):
        print("去死吧")
        self.welcome_label.setText("任务列表：请选择一个任务")

        self.remove_list_widget()
        self.clear_right_layout()
        self.task_list = QListWidget(self)
        self.center_layout.addWidget(self.task_list)
        print(6)

        # Set custom delegate for the task list
        delegate = TaskItemDelegate(self.task_list)
        self.task_list.setItemDelegate(delegate)

        self.assigned_doctor_id = 1
        data = {
            'assigned_doctor_id': self.assigned_doctor_id
        }
        self.sio.emit('get_task_list', data)
        print("c")
    def on_task_list(self,data):
        print(44)
        task_list_data = data['tasks']
        print(task_list_data)
        # 清空现有的任务项
        self.task_list.clear()

        for task in task_list_data:
            task_title = task['task_title']  # 获取任务标题
            task_id = task['task_id']  # 获取任务ID
            print("cnm")
            print(task_id)
            # 将任务项添加到列表中并将小部件设置为该任务项的控件
            #self.task_list.addItem(task_title)  # 添加任务标题到列表
            # Add task title to the list
            list_item = QListWidgetItem(task_title)
            list_item.setData(Qt.UserRole, task_id)  # Store the task ID as custom data
            #list_item.setFlags(list_item.flags() | Qt.ItemIsEditable)  # 设置为可编辑
            self.task_list.addItem(list_item)
            print(6)

            # Set the custom widget as the item for the QListWidget
            #self.task_list.setItemWidget(list_item, task_item_widget)

            print(f"Added task with ID: {task_id}")
            print("ccc")

    def create_more_button(self, item, task_id):
        """创建悬浮按钮 (显示三个点)"""
        print(99)
        button = QPushButton("...")  # 显示三个点
        button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                color: #007bff;
            }
        """)  # 设置按钮样式（例如字体大小）

        # 当点击按钮时，显示悬浮菜单
        button.clicked.connect(lambda: self.show_context_menu(item, task_id))

        return button

    def show_context_menu(self, item, task_id):
        """显示悬浮菜单（包含删除和编辑按钮）"""
        menu = QMenu(self)
        print(88)
        # 创建删除和编辑按钮
        edit_action = QAction("Edit", self)
        delete_action = QAction("Delete", self)

        # 为按钮设置点击事件
        edit_action.triggered.connect(lambda: self.edit_task(task_id))
        delete_action.triggered.connect(lambda: self.delete_task(task_id))

        # 将按钮添加到菜单
        menu.addAction(edit_action)
        menu.addAction(delete_action)

        # 显示菜单
        menu.exec_(self.mapToGlobal(item.pos()))  # 在任务项的位置显示菜单

    def edit_task(self, task_id):
        """编辑任务"""
        print(f"Editing task with ID: {task_id}")
        # 在这里处理编辑任务的逻辑

    def delete_task(self, task_id):
        """删除任务"""
        print(f"Deleting task with ID: {task_id}")
        # 在这里处理删除任务的逻辑
        self.sio.emit('delete_task', {'task_id': task_id})  # 删除任务的请求发送到后端

    def remove_list_widget(self):
        """删除 center_layout 中的 QListWidget 组件"""
        for i in range(self.center_layout.count()):
            item = self.center_layout.itemAt(i)
            widget = item.widget()

            # 如果是 QListWidget，删除它
            if widget and isinstance(widget, QListWidget):
                widget.deleteLater()  # 安全删除 QListWidget

    def clear_right_layout(self):
        """删除 right_layout 中的所有组件，但保留布局本身"""
        for i in range(self.right_layout.count()):
            item = self.right_layout.itemAt(i)
            widget = item.widget()

            # 如果存在控件，删除它
            if widget:
                widget.deleteLater()  # 删除控件，但保留布局
    def create_task(self):
        task_title = "New Task"  # 获取用户输入
        task_description = "Task Description"  # 获取用户输入
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

    window = ChatApp()
    window.show()

    loop.run_forever() # Start the event loop
    sys.exit(app.exec_())


