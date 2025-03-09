from PyQt5.QtWidgets import QStyledItemDelegate, QPushButton, QStyleOptionButton, QStyle, QAction, QMenu, QLineEdit, \
    QMessageBox, QVBoxLayout, QDialog, QLabel, QDialogButtonBox
from PyQt5.QtCore import Qt, QRect


class TaskItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, sio=None):
        super().__init__(parent)
        self.sio = sio
        self.buttons = []  # 存储所有的按钮实例
        self.current_edit = None

    def paint(self, painter, option, index):
        task_text = index.data(Qt.ItemDataRole.DisplayRole)  # 获取任务文本
        if task_text == "add task":
            # 仅绘制文本，不添加按钮
            super().paint(painter, option, index)
            return
        # Paint the task title as normal
        super().paint(painter, option, index)
        # Draw the "More" button (three dots)
        rect = QRect(option.rect.right() - 30, option.rect.top(), 30, 30)  # Button position
        row = index.row()
        for button in self.buttons:
            if button.property("row") == row:
                return  # 如果按钮已经存在，则不重复创建
        button = QPushButton("...", self.parent())
        button.setGeometry(rect)  # 设置按钮的位置和大小
        button.setStyleSheet("border: none; font-size: 16px; background: none; padding: 0px;color: white")
        button.setFlat(True)  # 让按钮看起来像是一个标签
        button.setProperty("row", row)  # 存储 index对应的行
        # 将按钮保存到列表中，避免每次都创建新按钮
        self.buttons.append(button)
        button.show()
        button.clicked.connect(lambda: self.on_button_click(index,button))

    def on_button_click(self, index,button):
        # 处理按钮点击事件
        task_id = index.data(Qt.UserRole)  # 获取与该项相关的任务ID
        print(f"按钮被点击，任务ID: {task_id}")
        menu = QMenu(self.parent())
        edit_action = QAction("编辑", self.parent())
        delete_action = QAction("删除", self.parent())
        edit_action.triggered.connect(lambda: self.on_edit(index))
        delete_action.triggered.connect(lambda: self.on_delete(index))
        menu.addAction(edit_action)
        menu.addAction(delete_action)
        # 获取按钮的相对位置
        global_pos = button.mapToGlobal(button.rect().topLeft())
        # 显示菜单（传入位置）
        menu.exec_(global_pos)

    def on_edit(self, index):
        task_id = index.data(Qt.UserRole)  # 获取任务ID
        print(f"编辑任务 ID: {task_id}")
        # 创建并显示 QDialog（弹出窗口）
        dialog = EditTaskDialog(index,task_id,self.sio)
        dialog.exec_()

    def on_delete(self, index):
        if not index.isValid():
            return
        task_id = index.data(Qt.UserRole)
        data = {'task_id': task_id}
        self.sio.emit('delete_task', data)
        model = index.model()
        if model:
            row = index.row()
            model.removeRow(row)  # 直接从模型中删除对应行
            # 移除按钮
            self.remove_button(row)

    def remove_button(self, row):
        """ 移除指定行的按钮 """
        for button in self.buttons:
            if button.property("row") == row:
                button.deleteLater()
                self.buttons.remove(button)
                self.parent().update()
                break

class EditTaskDialog(QDialog):
    def __init__(self, index,task_id, sio,parent=None):
        super().__init__(parent)
        self.index = index
        self.task_id = task_id
        self.sio = sio
        self.setWindowTitle("Edit task")
        self.setFixedSize(300, 150)
        # 创建布局和控件
        layout = QVBoxLayout()
        self.line_edit = QLineEdit(self.index.data(Qt.DisplayRole), self)
        layout.addWidget(self.line_edit)
        # 创建按钮框
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(button_box)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        # 设置布局
        self.setLayout(layout)

    def accept(self):
        new_title = self.line_edit.text()
        original_title = self.index.data(Qt.DisplayRole)
        if new_title != original_title:
            data = {
                'task_id': self.task_id,
                'new_task_title': new_title
            }
            self.sio.emit('update_task_title', data)
            print(f"任务标题已修改，从 '{original_title}' 更新为 '{new_title}'")
            self.index.model().setData(self.index, new_title, Qt.EditRole)  # 更新数据模型
        super().accept()  # 关闭对话框

    def reject(self):
        # 用户点击取消时，关闭对话框
        super().reject()
