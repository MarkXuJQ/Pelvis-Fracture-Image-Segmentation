from PyQt5.QtWidgets import QDialog, QProgressBar, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

class UploadProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("上传进度")
        self.setFixedSize(400, 150)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)  # 移除关闭按钮
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 添加状态标签
        self.status_label = QLabel("准备上传...", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 添加进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # 添加详细信息标签
        self.detail_label = QLabel("", self)
        self.detail_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detail_label)
        
        self.setLayout(layout)

    def update_progress(self, progress):
        """更新进度和状态"""
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(f"正在上传: {progress:.1f}%")
        
        # 计算上传速度和剩余时间（这里可以添加）
        if progress > 0:
            self.detail_label.setText("请耐心等待...") 