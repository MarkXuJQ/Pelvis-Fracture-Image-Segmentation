import sys
import json
import datetime
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont
from PyQt5.QtWidgets import (
    QApplication, QListView, QWidget, QVBoxLayout, QLabel, QPushButton, QMenu, QStyleOptionViewItem,
    QStyledItemDelegate, QSpacerItem, QSizePolicy, QDateTimeEdit, QComboBox, QFrame
)
from PyQt5.QtCore import Qt, QSize, QDateTime, QTimer, QDate, QTime
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
from system.document import DocumentManagerWidget
from taskdetails import TaskDetailsWidget

class ChatApp(QMainWindow):
    def __init__(self,user_id):
        super().__init__()
        apply_stylesheet(self)

        self.sio = socketio.Client()
        self.sio.connect('http://localhost:5000')
        self.sender_id = user_id
        self.receiver_id = None
        self.receiver_name = None
        self.doctors_list = []  # 存储医生列表，避免多次请求导致 UI 冲突
        self.is_listening = True  # 控制是否监听事件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(current_dir, "ui", "chat_window.ui")
        uic.loadUi(ui_file, self)
        self.adjustlayout()
        # 设置 Socket.IO 事件
        self.sio.on('receive_message', self.on_receive_message)
        self.sio.on('chat_history', self.on_chat_history)
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

    def load_doctor_list(self):
        self.welcomeLabel.setText("消息")
        self.remove_list_widget()
        self.clear_right_layout()
        self.doctor_list = QListWidget(self)
        self.centerLayout.addWidget(self.doctor_list)
        data = {'user_id': self.sender_id}
        self.sio.emit('get_doctors_except_user', data)
        self.doctor_list.itemClicked.connect(self.load_chat_history)

    def on_doctor_list(self,data):
        if not self.is_listening:  # 如果 ChatApp 被禁用监听，就不执行
            print("ChatApp 忽略 doctors_list 事件")
            return
        doctors = data.get('doctors', [])  # 获取医生列表
        self.doctors_list = data['doctors']  # 存储医生数据，避免重复请求

        for doctor in doctors:
            doctor_id = doctor['doctor_id']
            doctor_name = doctor['doctor_name']
            item = QListWidgetItem(doctor_name)
            item.setData(Qt.UserRole, (doctor_id,doctor_name))  # 绑定 doctor_id
            self.doctor_list.addItem(item)
        print("ChatApp 更新医生列表:", self.doctors_list)

    def disable_listening(self):
        """禁用 ChatApp 对 doctors_list 事件的监听"""
        self.is_listening = False

    def enable_listening(self):
        """启用 ChatApp 监听"""
        self.is_listening = True
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
        self.receiver_id,self.receiver_name = item.data(Qt.UserRole)
        data = {
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id
        }
        self.sio.emit('get_chat_history', data)

    def on_chat_history(self, data):
        chat_history = data['history']
        for message in chat_history:
            sender = 'Me' if message['sender_id'] == int(self.sender_id) else self.receiver_name
            self.chat_area.append(f"{sender}: {message['message_content']}")

    def load_task_list(self,*args):
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
        self.clear_right_layout()
        if item.text() == "add task":
            # 创建任务创建组件，并添加到 rightLayout
            self.task_creation = TaskCreationWidget(self,self.sender_id, self.rightLayout)
        else:
            self.task_details = TaskDetailsWidget(self,item,self.rightLayout)

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

    def display_document_page(self):
        self.remove_list_widget()
        self.clear_right_layout()
        #self.document = DocumentManagerWidget(self)

        '''file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            # 选择文件上传逻辑
            document_data = {'file_paths': file_paths, 'patient_id': self.receiver_id}
            self.sio.emit('upload_document', document_data)'''

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


class TaskCreationWidget:
    def __init__(self, chat_app,user_id, rightLayout):
        self.chat_app = chat_app  # 直接从 ChatApp 获取医生数据
        self.user_id = user_id
        self.sio = chat_app.sio  # 共享 SocketIO 实例
        self.rightLayout = rightLayout  # 传递已有的布局
        self.init_ui()
        self.sio.on('patients_list', self.on_patients_list)

    def init_ui(self):
        # 任务标题
        self.title_input = QLineEdit()
        self.rightLayout.addWidget(QLabel("任务标题:"))
        self.rightLayout.addWidget(self.title_input)
        # 任务描述
        self.description_input = QTextEdit()
        self.rightLayout.addWidget(QLabel("任务描述:"))
        self.rightLayout.addWidget(self.description_input)
        # 任务截止时间
        self.due_date_input = QDateTimeEdit()
        self.due_date_input.setDateTime(QDateTime.currentDateTime())
        self.rightLayout.addWidget(QLabel("截止时间:"))
        self.rightLayout.addWidget(self.due_date_input)
        # 任务状态
        self.status_input = QComboBox()
        self.status_input.addItems(["pending", "completed", "in_progress"])
        self.rightLayout.addWidget(QLabel("任务状态:"))
        self.rightLayout.addWidget(self.status_input)
        # 医生选择（多选）
        self.doctor_select = QListWidget()
        self.doctor_select.setSelectionMode(QListWidget.MultiSelection)
        self.rightLayout.addWidget(QLabel("分配给医生:"))
        self.rightLayout.addWidget(self.doctor_select)
        # 病人选择
        self.patient_select = QComboBox()
        self.patient_select.addItem("请选择病人", userData=None)
        self.rightLayout.addWidget(QLabel("相关病人:"))
        self.rightLayout.addWidget(self.patient_select)
        # OK 按钮
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.create_task)
        self.rightLayout.addWidget(self.ok_button)
        # 立即尝试获取医生列表
        self.update_doctor_selection()
        self.sio.emit('get_all_patients')  # 发送请求

    def on_patients_list(self, data):
        """处理服务器返回的病人列表"""
        print("TaskCreationWidget 收到病人列表:", data['patients'])
        patients = data.get('patients', [])  # 获取病人列表
        for patient in patients:
            patient_id = patient["patient_id"]
            patient_name = patient["patient_name"]
            # 在 QComboBox 中添加项，并绑定 patient_id
            self.patient_select.addItem(patient_name, userData=patient_id)

    def update_doctor_selection(self):
        """更新医生选择框的内容"""
        doctors = self.chat_app.doctors_list  # 直接从 ChatApp 获取数据

        if not doctors:  # 如果医生列表为空，则请求数据
            print("TaskCreationWidget 发现医生列表为空，向 ChatApp 请求获取医生数据...")
            self.request_doctors_list()
            return  # 避免 UI 崩溃

        self.doctor_select.clear()
        for doctor in doctors:
            doctor_id = doctor["doctor_id"]  # 取 doctor_id
            doctor_name = doctor["doctor_name"]  # 取 doctor_name
            item = QListWidgetItem(doctor_name)  # 创建列表项
            item.setData(Qt.UserRole, doctor_id)  # 绑定 doctor_id
            self.doctor_select.addItem(item)  # 添加到 QListWidget

    def request_doctors_list(self):
        """临时监听 doctors_list 事件，并请求数据"""
        # 让 ChatApp 不再监听事件
        self.chat_app.disable_listening()
        # 让 TaskCreationWidget 监听事件
        self.sio.on('doctors_list', self.on_doctor_list_temp)
        # 发送请求
        print("TaskCreationWidget 正在向服务器请求医生列表...")
        self.sio.emit('get_doctors_except_user', {'user_id': self.chat_app.sender_id})

    def on_doctor_list_temp(self, data):
        """TaskCreationWidget 处理服务器返回的医生列表"""
        print("TaskCreationWidget 收到医生列表:", data['doctors'])
        # 解析数据
        self.chat_app.doctors_list = data['doctors']
        # 更新 UI
        self.update_doctor_selection()
        # 恢复 ChatApp 监听，并移除 TaskCreationWidget 的监听
        self.sio.on('doctors_list', None)  # 取消监听
        self.sio.on('doctors_list', self.chat_app.on_doctor_list)  # 恢复 ChatApp 监听
        self.chat_app.enable_listening()  # 重新启用 ChatApp 监听
    def create_task(self):
        task_title = self.title_input.text()
        task_description = self.description_input.toPlainText()
        due_date = self.due_date_input.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        status = self.status_input.currentText()
        # 获取所有选中的医生 ID
        assigned_doctors = [item.data(Qt.UserRole) for item in self.doctor_select.selectedItems()]  # 获取绑定的 doctor_id
        # 添加当前用户 ID（self.sio.sender_id）
        assigned_doctors.append(self.chat_app.sender_id)
        patient_id = self.patient_select.currentData()  # 获取绑定的 patient_id
        if not assigned_doctors:
            print("错误：未选择医生！")
            return
        if patient_id is None:
            print("错误：未选择病人！")
            return
        data = {
            'task_title': task_title,
            'task_description': task_description,
            'due_date': due_date,
            'status': status,
            'assigned_doctor_ids': assigned_doctors,
            'patient_id': patient_id,
        }
        self.sio.emit('create_task', data)
        # 清空输入框
        self.title_input.clear()
        self.description_input.clear()
        self.due_date_input.setDateTime(QDateTime.currentDateTime())
        self.status_input.setCurrentIndex(0)
        # 清空医生选择（QListWidget）
        self.doctor_select.clearSelection()  # 取消所有选中项
        # 清空病人选择（QComboBox）
        self.patient_select.setCurrentIndex(0)
        # 任务创建后，延迟 1 秒再更新任务列表，确保数据库写入完成
        QTimer.singleShot(1000, self.chat_app.load_task_list)



if __name__ == "__main__":
    app = QApplication(sys.argv)

    loop = QEventLoop(app)  # Use QEventLoop with PyQt5 to handle asyncio tasks
    asyncio.set_event_loop(loop)  # Set the event loop to be used

    window = ChatApp(1)
    window.show()

    loop.run_forever() # Start the event loop
    sys.exit(app.exec_())


