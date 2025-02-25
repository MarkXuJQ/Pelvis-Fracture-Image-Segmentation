from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QLabel, QVBoxLayout, QWidget, QTabWidget, \
    QPushButton, QTableWidget, QHBoxLayout


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('聊天信息界面')
        self.setGeometry(0, 0, 1200, 800)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # 左侧医生列表
        self.left_widget = QWidget(self)
        self.left_layout = QVBoxLayout(self.left_widget)
        self.doctor_list_label = QLabel("医生列表", self)
        self.left_layout.addWidget(self.doctor_list_label)

        self.chat_list = QListWidget(self)
        self.chat_list.addItem("Doctor John")
        self.chat_list.addItem("Doctor Jane")
        self.chat_list.itemClicked.connect(self.display_chat)

        self.left_layout.addWidget(self.chat_list)

        # 中间区域：动态内容
        self.center_widget = QWidget(self)
        self.center_layout = QVBoxLayout(self.center_widget)
        self.welcome_label = QLabel("欢迎使用！请选择医生或右侧的功能。", self)
        self.center_layout.addWidget(self.welcome_label)
        self.dynamic_content_area = QWidget(self)
        self.center_layout.addWidget(self.dynamic_content_area)

        # 右侧：标签页
        self.tabs = QTabWidget(self)
        self.task_tab = QWidget(self)
        self.document_tab = QWidget(self)
        self.note_tab = QWidget(self)

        self.tabs.addTab(self.task_tab, "Task")
        self.tabs.addTab(self.document_tab, "Documents")
        self.tabs.addTab(self.note_tab, "Notes")

        self.tabs.currentChanged.connect(self.display_tab)

        # 设置布局
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.addWidget(self.left_widget)
        main_layout.addWidget(self.center_widget)
        main_layout.addWidget(self.tabs)

    def display_chat(self, item):
        # 根据选择的医生，显示聊天框
        self.welcome_label.setText(f"Chatting with {item.text()}")
        # 这里您可以插入显示聊天内容的代码

    def display_tab(self, index):
        # 根据选择的标签显示不同的内容
        if index == 0:
            self.welcome_label.setText("Task Management")
        elif index == 1:
            self.welcome_label.setText("Document Upload")
        elif index == 2:
            self.welcome_label.setText("Notes Management")

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
