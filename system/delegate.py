from PyQt5.QtWidgets import QStyledItemDelegate, QPushButton, QStyleOptionButton, QStyle, QAction, QMenu, QLineEdit, \
    QMessageBox, QVBoxLayout, QDialog, QLabel, QDialogButtonBox
from PyQt5.QtCore import Qt, QRect


class TaskItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.buttons = []  # 存储所有的按钮实例
        self.current_edit = None

    def paint(self, painter, option, index):
        # Get the task title from the model data
        task_title = index.data(Qt.DisplayRole)
        # Paint the task title as normal
        super().paint(painter, option, index)
        if not self.buttons:
            # Draw the "More" button (three dots)
            rect = QRect(option.rect.right() - 30, option.rect.top(), 30, 30)  # Button position
            # 创建一个 QPushButton 对象
            button = QPushButton("...", self.parent())
            button.setGeometry(rect)  # 设置按钮的位置和大小
            button.setStyleSheet("border: none; font-size: 16px; background: none; padding: 0px;color: black")
            # 设置按钮的字体大小为 14px，去掉边框，设置背景透明
            button.setFlat(True)  # 让按钮看起来像是一个标签（没有边框和背景）
            # 将按钮保存到列表中，避免每次都创建新按钮
            self.buttons.append(button)
            button.show()  # 显示按钮
            # 给按钮绑定点击事件
            button.clicked.connect(lambda: self.on_button_click(index,button))

    def on_button_click(self, index,button):
        # 处理按钮点击事件
        task_id = index.data(Qt.UserRole)  # 获取与该项相关的任务ID
        print(f"按钮被点击，任务ID: {task_id}")
        # 创建一个QMenu，显示悬浮窗
        menu = QMenu(self.parent())
        # 创建编辑和删除动作
        edit_action = QAction("编辑", self.parent())
        delete_action = QAction("删除", self.parent())
        # 连接每个动作的点击事件
        edit_action.triggered.connect(lambda: self.on_edit(index))
        delete_action.triggered.connect(lambda: self.on_delete(index))
        # 添加动作到菜单
        menu.addAction(edit_action)
        menu.addAction(delete_action)
        print(7)
        # 获取按钮的相对位置
        global_pos = button.mapToGlobal(button.rect().topLeft())
        # 显示菜单（传入位置）
        menu.exec_(global_pos)
        print(8)

    def on_edit(self, index):
        task_id = index.data(Qt.UserRole)  # 获取任务ID
        print(f"编辑任务 ID: {task_id}")
        print(index)
        # 这里你可以执行打开编辑界面的操作，传递 task_id
        task_title = index.data(Qt.DisplayRole)  # 获取当前任务标题
        print(f"Current title: {task_title}")
        print(2)
        # 创建并显示 QDialog（弹出窗口）
        dialog = EditTaskDialog(index,task_id)
        dialog.exec_()  # 执行对话框并等待用户操作
        print("编辑完成")
        print(4)



    def on_edit_finished(self, item):
        new_title = self.current_edit.text()  # 获取新的任务标题
        print(f"Editing finished. New title: {new_title}")
        #old_task_title = index.data(Qt.DisplayRole)  # 获取原来的任务标题
        # 更新任务标题
        item.setText(new_title)  # 修改当前任务项的标题
        self.current_edit = None
        print(f"任务标题已更新为: {new_title}")
        # 比较新旧任务标题
        '''if new_task_title != old_task_title:
            # 如果任务标题有变动，则触发更新请求
            self.update_task_title(task_id, new_task_title)

        # 隐藏QLineEdit并清空引用
        self.current_edit.hide()
        self.current_edit = None'''

    def update_task_title(self, task_id, new_task_title):
        # 发送请求更新数据库中的任务标题
        print(f"任务 {task_id} 标题已经修改为: {new_task_title}")

        # 这里可以发送请求到后端，或者通过 socket 等方式更新数据库
        # 举个例子:
        data = {
            'task_id': task_id,
            'new_task_title': new_task_title
        }
        # 使用 socket 发送数据（根据你的实现方式）
        # self.sio.emit('update_task_title', data)

        # 你也可以在此弹出消息框提醒用户成功更新
        QMessageBox.information(self.parent(), "成功", "任务标题已经更新！")

    def on_delete(self, index):
        task_id = index.data(Qt.UserRole)  # 获取任务ID
        print(f"删除任务 ID: {task_id}")
        # 这里你可以执行删除任务的操作，传递 task_id

class EditTaskDialog(QDialog):
    def __init__(self, index,task_id, parent=None):
        super().__init__(parent)
        self.index = index  # 传入要编辑的项
        self.task_id = task_id  # 任务 ID
        self.setWindowTitle("Edit task")
        self.setFixedSize(300, 150)

        # 创建布局和控件
        layout = QVBoxLayout()

        # 创建 QLineEdit 并设置为当前标题
        self.line_edit = QLineEdit(self.index.data(Qt.DisplayRole), self)
        layout.addWidget(self.line_edit)

        # 创建按钮框
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(button_box)

        # 连接信号和槽
        button_box.accepted.connect(lambda: self.accept(self.task_id))

        button_box.rejected.connect(self.reject)

        # 设置布局
        self.setLayout(layout)

    def accept(self,task_id):
        # 获取用户输入的新标题
        new_title = self.line_edit.text()
        print(f"新的任务标题: {new_title}")
        original_title = self.index.data(Qt.DisplayRole)

        # 仅在标题修改时才更新
        if new_title != original_title:
            data = {
                'task_id': task_id,
                'new_task_title': new_title
            }
            # 使用 socket 发送数据（根据你的实现方式）
            # self.sio.emit('update_task_title', data)
            print(f"任务标题已修改，从 '{original_title}' 更新为 '{new_title}'")
            # 发送请求更新任务标题，这里假设你有一个方法来执行任务更新
            self.update_task_title(new_title)


        super().accept()  # 关闭对话框

    def update_task_title(self, new_title):
        # 这里是发送更新请求的地方
        # 假设你有一个方法来处理更新任务标题的请求，例如：
        print(f"发送更新请求：更新任务 ID {self.task_id} 的标题为 '{new_title}'")
        # 你可以在这里实际执行你的请求，比如通过 HTTP 或信号与槽等方式。
        self.index.model().setData(self.index, new_title, Qt.EditRole)  # 更新数据模型

    def reject(self):
        # 用户点击取消时，关闭对话框
        super().reject()
