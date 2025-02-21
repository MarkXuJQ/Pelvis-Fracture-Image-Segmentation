import sys
import json
import socketio
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QListWidget
from qasync import QEventLoop
import asyncio

class ChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Doctor-Patient Chat")
        self.setGeometry(100, 100, 800, 600)

        # 添加聊天列表（显示医生聊天列表）
        self.chat_list = QListWidget(self)
        self.chat_list.setGeometry(10, 10, 200, 500)
        self.chat_list.addItem("Doctor John")  # 示例医生
        self.chat_list.addItem("Doctor Jane")  # 示例医生
        self.chat_list.clicked.connect(self.load_chat_history)  # 当点击聊天记录时加载对应聊天记录

        # 聊天区
        self.chat_area = QTextEdit(self)
        self.chat_area.setGeometry(220, 10, 560, 400)
        self.chat_area.setReadOnly(True)

        # 消息输入框
        self.message_input = QLineEdit(self)
        self.message_input.setGeometry(220, 420, 460, 40)

        # 发送按钮
        self.send_button = QPushButton("Send", self)
        self.send_button.setGeometry(690, 420, 90, 40)
        self.send_button.clicked.connect(self.send_message)

        # 应用样式表
        self.apply_stylesheet()

        # 创建 Socket.IO 客户端
        self.sio = socketio.Client()

        # 连接到服务器
        self.sio.connect('http://localhost:5000')

        self.sender_id = 1  # 示例发送者ID（医生ID）
        self.receiver_id = None  # 将在点击聊天列表时动态设置接收者ID

        # 设置 Socket.IO 事件
        self.sio.on('receive_message', self.on_receive_message)
        self.sio.on('chat_history', self.on_chat_history)

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
        # 获取聊天记录时，选择的是哪个医生
        selected_doctor = self.chat_list.selectedItems()[0].text()
        if selected_doctor == "Doctor John":
            self.receiver_id = 2  # 设置对应医生的接收者ID
        elif selected_doctor == "Doctor Jane":
            self.receiver_id = 3  # 设置对应医生的接收者ID

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


