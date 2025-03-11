from PyQt5.QtWidgets import QStyledItemDelegate, QPushButton, QStyleOptionButton, QStyle, QAction, QMenu, QLineEdit, \
    QMessageBox, QVBoxLayout, QDialog, QLabel, QDialogButtonBox
from PyQt5.QtCore import Qt, QRect, QTimer
from stylesheet import apply_stylesheet


class TaskItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, sio=None, doctor_window=None):
        super().__init__(parent)
        #self.parent = parent
        self.doctor_window = doctor_window
        self.sio = sio
        self.buttons = []  # 存储所有的按钮实例
        self.current_edit = None

    def paint(self, painter, option, index):
        task_text = index.data(Qt.ItemDataRole.DisplayRole)  # 获取任务文本
        if task_text in ["add task", "未命名"]:
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
        menu = QMenu(self.parent())
        edit_action = QAction("编辑", self.parent())
        delete_action = QAction("删除", self.parent())
        edit_action.triggered.connect(lambda: self.on_edit(index))
        if self.sio is not None:
            row = button.property("row")
            index = self.parent().indexFromItem(self.parent().item(row))  # 通过 row 获取正确的 index
            delete_action.triggered.connect(lambda: self.on_delete(index,row))
            menu.addAction(edit_action)
        else:
            row = button.property("row")  # 获取当前按钮的行号
            delete_action.triggered.connect(lambda: self.remove(row))
        menu.addAction(delete_action)
        # 获取按钮的相对位置
        global_pos = button.mapToGlobal(button.rect().topLeft())
        # 显示菜单（传入位置）
        menu.exec_(global_pos)

    def on_edit(self, index):
        task_id = index.data(Qt.UserRole)  # 获取任务ID
        # 创建并显示 QDialog（弹出窗口）
        dialog = EditTaskDialog(index,task_id,self.sio)
        dialog.exec_()

    def on_delete(self, index,row):
        if not index.isValid():
            return
        task_id = index.data(Qt.UserRole)
        data = {'task_id': task_id}
        self.sio.emit('delete_task', data)
        self.remove(row)

    def remove(self, row):
        """ 移除指定行的按钮 """
        item = self.parent().takeItem(row)
        if item is None:
            return  # 直接返回，避免后续崩溃
        if item:
            del item
            self.remove_button(row)
            # 确保 UI 强制刷新
            self.parent().repaint()
        print(f"删除后剩余按钮数: {len(self.buttons)}")

    def remove_button(self,row):
        """ 移除指定行的按钮 """
        buttons_to_remove = [button for button in self.buttons if button.property("row") == row]
        for button in buttons_to_remove:
            self.buttons.remove(button)
            button.hide()
            button.setParent(None)
            button.deleteLater()
        for button in self.buttons:
            button_row = button.property("row")
            if button_row >= row:  # 如果行号大于或等于被删除的行号，则调整
                new_row = button_row - 1
                button.setProperty("row", new_row)
                item = self.parent().item(new_row)
                if item:
                    index = self.parent().indexFromItem(item)
                    rect = self.parent().visualRect(index)
                    button.setGeometry(rect.right() - 30, rect.top(), 30, 30)

class EditTaskDialog(QDialog):
    def __init__(self, index,task_id, sio,parent=None):
        super().__init__(parent)
        apply_stylesheet(self)
        self.index = index
        self.task_id = task_id
        self.sio = sio
        self.error = None
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
        self.waiting_for_response = False  # 防止重复调用
        self.sio.on('task_title_updated', self.on_task_title_updated)

    def accept(self):
        """ 处理任务标题更新 """
        if self.waiting_for_response:
            return
        new_title = self.line_edit.text()
        original_title = self.index.data(Qt.DisplayRole)
        if new_title != original_title:
            data = {
                'task_id': self.task_id,
                'new_task_title': new_title
            }
            self.sio.emit('update_task_title', data)
            # 等待服务器返回数据
            self.waiting_for_response = True
            QTimer.singleShot(1000, self.handle)
        else:
            self.close()
            return
    def on_task_title_updated(self, response):
        """ 处理服务器返回的任务标题更新结果 """
        if not self.waiting_for_response:
            return  # 如果不是当前请求的响应，忽略
        self.waiting_for_response = False  # 取消等待状态
        task_id = response.get('task_id')
        self.new_title = response.get('new_task_title')
        updated_rows = response.get('updated_rows', 0)  # 获取被更新的任务数
        error_msg = response.get('error')

        if task_id is None and error_msg:
            # **任务标题更新失败**
            self.message = error_msg
        print(f"✅ 任务标题更新成功: {self.new_title}（更新 {updated_rows} 条任务）")
    def handle(self):
        if self.message is not None:
            QMessageBox.warning(None, "任务更新失败", f"错误：{self.message}")
        else:
            # 更新任务模型
            self.index.model().setData(self.index, self.new_title, Qt.EditRole)
        self.close()
    def reject(self):
        # 用户点击取消时，关闭对话框
        super().reject()
