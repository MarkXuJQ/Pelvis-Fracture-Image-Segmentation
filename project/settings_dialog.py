# settings_dialog.py

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton

class SettingsDialog(QDialog):
    def __init__(self, parent=None, render_on_open=False):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        self.render_on_open = render_on_open

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Checkbox for rendering model on open
        self.render_on_open_checkbox = QCheckBox("Render 3D model when opening a CT scan")
        self.render_on_open_checkbox.setChecked(self.render_on_open)
        layout.addWidget(self.render_on_open_checkbox)

        # OK and Cancel buttons
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_settings(self):
        return {
            'render_on_open': self.render_on_open_checkbox.isChecked()
        }
    
