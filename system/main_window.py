from PyQt5 import uic
from PyQt5.QtCore import Qt
#from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QMessageBox, QToolBar, QTableWidget, QCheckBox, \
    QTableWidgetItem, QHeaderView, QPushButton, QListWidget, QLabel

from xray_viewer import XRayViewer
from ct_viewer import CTViewer
from patient_manage import PatientManageWindow
import SimpleITK as sitk
import os
from settings_dialog import SettingsDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(current_dir, "ui", "main_window.ui")
        
        # 调试输出
        print(f"Current directory: {current_dir}")
        print(f"UI file path: {ui_file}")
        print(f"UI file exists: {os.path.exists(ui_file)}")
        
        uic.loadUi(ui_file, self)
        with open('ui/button_style.qss', 'r', encoding='utf-8') as f:
            self.setStyleSheet(f.read())
        # 获取表格的引用（从 .ui 文件中获取）
        self.tableWidget = self.findChild(QTableWidget, 'tableWidget')
        self.listWidget = self.findChild(QListWidget, 'listWidget_3')
        self.patient_manage_window = PatientManageWindow(self.tableWidget,self.listWidget)
        self.addButton.clicked.connect(self.patient_manage_window.add_patient)

        self.setWindowTitle("Medical Image Viewer")
        self.setGeometry(0, 0, 1900, 1000)
        self.viewer = None  # Will hold the current image viewer
        self.render_on_open = False
        self.initUI()

        print(os.path.abspath("../image/plan/头像测试.jpg"))

    def initUI(self):
        # Create actions
        open_action = QAction('Open Image', self)
        open_action.triggered.connect(self.open_image)

        save_as_action = QAction('Save As', self)
        save_as_action.triggered.connect(self.save_image)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        
        # Create settings action
        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(self.open_settings)

        # Create menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        file_menu.addAction(open_action)
        file_menu.addAction(save_as_action)
        file_menu.addAction(exit_action)
        file_menu.addAction(settings_action)
        self.file_menu = file_menu

        # Save the action for later use
        self.save_as_action = save_as_action
        
        # Create a toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        self.design_table()
        self.open_action.clicked.connect(self.open_image)
        self.exit_action.clicked.connect(self.close)
        self.settings_action.clicked.connect(self.open_settings)
        # Connect Patient Management button
        self.sure.clicked.connect(self.open_patient_manage)
        self.deleteButton.clicked.connect(self.patient_manage_window.on_delete_patient_info)
        # Status bar
        self.statusBar().showMessage('Ready')

    def design_table(self):
        self.tableWidget.setEnabled(True)  # 启用表格交互

        """填充表格并为每一行添加复选框"""
        self.tableWidget.blockSignals(True)  # 暂时禁用信号
        # 调整每一列的宽度
        self.tableWidget.setColumnWidth(0, 50)  # 设置第一列宽度为100
        self.tableWidget.setColumnWidth(1, 200)  # 设置第二列宽度为200
        self.tableWidget.setColumnWidth(2, 200)  # 设置第三列宽度为150
        self.tableWidget.setColumnWidth(3, 200)  # 设置第三列宽度为150

        # 最后一列自动填充剩余宽度
        self.tableWidget.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)

        # 获取当前表格的行数
        rows = self.tableWidget.rowCount()
        for row in range(rows):
            self.tableWidget.setRowHeight(row, 60)  # 设置每一行的高度为40
            checkBoxItem = QTableWidgetItem()  # 创建一个表格项
            checkBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)  # 设置为可选中且启用
            checkBoxItem.setCheckState(Qt.Unchecked)  # 设置复选框初始状态为未选中

            # 将复选框添加到表格的第一列
            self.tableWidget.setItem(row, 0, checkBoxItem)
            # 在control_str列（假设为第3列）下添加"Upload File"按钮
            upload_button = QPushButton('Upload_File')
            upload_button.clicked.connect(self.on_upload_button_clicked)  # 连接按钮点击事件到槽函数

            #self.deleteButton.clicked.connect(self.patient_manage_window.on_delete_patient_info)
            self.tableWidget.setCellWidget(row, 4, upload_button)  # 在第3列插入按钮
            self.tableWidget.blockSignals(False)  # 启用信号
            # 连接itemChanged信号到槽函数
            self.tableWidget.itemChanged.connect(self.on_checkbox_state_changed)
        # 设置焦点
        #self.tableWidget.setFocus()  # 将焦点设置到 QTableWidget

    # 复选框状态改变时的处理函数
    def on_checkbox_state_changed(self, item):
        print(2)
        """复选框状态改变时的处理函数"""
        if item.column() == 0:  # 只处理第一列（复选框列）
            row = item.row()
            check_state = item.checkState()
            if check_state == Qt.Checked:
                print(f"Checkbox in row {row} is checked!")
            elif check_state == Qt.Unchecked:
                print(f"Checkbox in row {row} is unchecked!")

    # 假设on_upload_button_clicked是处理按钮点击事件的槽函数
    def on_upload_button_clicked(self):
        # 获取点击的行
        button = self.sender()
        row = self.tableWidget.indexAt(button.pos()).row()  # 获取按钮所在的行
        print(f"Button clicked at row {row}")

        # 假设文件选择后需要将文件名添加到 listWidget
        self.patient_manage_window.load_patient()

        # 清除表格中的所有选中状态
        self.clear_table_selection()

    def clear_table_selection(self):
        # 清除表格中的所有选中状态
        self.tableWidget.clearSelection()  # 清除选中的单元格、行或列

        # 清除复选框的选中状态
        for row in range(self.tableWidget.rowCount()):
            checkBoxItem = self.tableWidget.item(row, 0)  # 获取复选框所在的单元格项
            if checkBoxItem:
                checkBoxItem.setCheckState(Qt.Unchecked)  # 取消复选框选中状态

        # 如果表格中有按钮，确保按钮的选中状态被清除
        for row in range(self.tableWidget.rowCount()):
            button = self.tableWidget.cellWidget(row, 2)  # 假设按钮在第3列
            if isinstance(button, QPushButton):
                button.setStyleSheet('')  # 清除按钮的选中样式（如果有的话）

    def mousePressEvent(self, event):
        """
                重载 mousePressEvent 来检测点击区域
                """
        # 如果点击的区域不在 listWidget 内部，则清除选中状态
        if not self.listWidget.rect().contains(event.pos()):
            self.listWidget.clearSelection()

        # 如果点击的区域不在 tableWidget 内部，则清除选中状态
        if not self.tableWidget.rect().contains(event.pos()):
            self.tableWidget.clearSelection()

        # 如果点击的区域不在删除按钮内部，则清除选中状态
        if not self.deleteButton.rect().contains(event.pos()):
            self.listWidget.clearSelection()
            self.tableWidget.clearSelection()

        super().mousePressEvent(event)  # 保证其他事件仍然能正常处理

    def open_patient_manage(self):
        # Create and show the Patient Management window
        self.patient_manage_window = PatientManageWindow(self.tableWidget,self.listWidget)
        self.patient_manage_window.show()

    def open_image(self):
        # Open file dialog to select image
        options = QFileDialog.Options()
        file_types = "All Files (*);;DICOM Files (*.dcm);;NIfTI Files (*.nii *.nii.gz);;NRRD Files (*.nrrd);;MetaImage Files (*.mha *.mhd)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", file_types, options=options)
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        try:
            self.image = sitk.ReadImage(file_path)
            dimension = self.image.GetDimension()
            if dimension == 2:
                # Display 2D image
                image_array = sitk.GetArrayFromImage(self.image)
                self.viewer = XRayViewer(image_array)
                # Disable 3D model and crosshair actions
                #self.generate_model_action.setEnabled(False)
                #self.create_crosshairs_action.setEnabled(False)
            elif dimension == 3:
                # Display 3D image
                self.viewer = CTViewer(self.image, render_model=self.render_on_open)
                #self.viewer.setParent(self)
                #self.generate_model_action.setEnabled(True)
                #self.create_crosshairs_action.setEnabled(True)  # Enable crosshair button
            else:
                QMessageBox.warning(self, "Unsupported Image", "The selected image has unsupported dimensions.")
                return

            self.setCentralWidget(self.viewer)
            self.statusBar().showMessage(f'Loaded image: {file_path}')
            self.current_file_path = file_path  # Store the current file path
            self.save_as_action.setEnabled(True)  # Enable "Save As"

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def save_image(self):
        if not hasattr(self, 'image'):
            QMessageBox.warning(self, "No Image", "No image is loaded to save.")
            return

        # Define supported formats
        formats = [
            ("NIfTI (*.nii)", "*.nii"),
            ("NIfTI Compressed (*.nii.gz)", "*.nii.gz"),
            ("NRRD (*.nrrd)", "*.nrrd"),
            ("MetaImage (*.mha *.mhd)", "*.mha *.mhd"),
            ("DICOM (*.dcm)", "*.dcm"),
            ("PNG Image (*.png)", "*.png"),
            ("JPEG Image (*.jpg *.jpeg)", "*.jpg *.jpeg"),
        ]

        # Create file dialog for saving
        options = QFileDialog.Options()
        file_filter = ";;".join([desc for desc, ext in formats])
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Image As", "", file_filter, options=options
        )

        if save_path:
            # Determine the selected format
            for desc, ext in formats:
                if desc == selected_filter:
                    output_extension = ext.replace("*", "").strip().split()[0]
                    break
            else:
                output_extension = os.path.splitext(save_path)[1]

            # Ensure the save path has the correct extension
            if not save_path.lower().endswith(output_extension.lower()):
                save_path += output_extension

            try:
                # Handle DICOM separately if needed
                if output_extension.lower() == ".dcm":
                    self.save_as_dicom(self.image, save_path)
                else:
                    sitk.WriteImage(self.image, save_path)
                QMessageBox.information(self, "Save Successful", f"Image saved to {save_path}")
                self.statusBar().showMessage(f'Image saved to {save_path}')
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")

    def save_as_dicom(self, image, save_path):
        # Check if the image is 3D or 2D
        dimension = image.GetDimension()
        if dimension == 3:
            # For 3D images, save each slice as a separate DICOM file
            size = image.GetSize()
            dir_name = os.path.splitext(save_path)[0]  # Remove extension for directory
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for i in range(size[2]):
                slice_i = image[:, :, i]
                slice_filename = os.path.join(dir_name, f"slice_{i}.dcm")
                sitk.WriteImage(slice_i, slice_filename)
        else:
            # For 2D images
            sitk.WriteImage(image, save_path) 
            
    # def generate_model(self):
    #     if hasattr(self.viewer, 'generate_and_display_model'):
    #         self.viewer.render_model = True
    #         self.viewer.generate_and_display_model()
    #     else:
    #         QMessageBox.warning(self, "Not Available", "Model generation is not available for this image.")

    # def create_crosshairs(self):
    #     if self.viewer is None:
    #         QMessageBox.warning(self, "Not Available", "The image viewer is not initialized.")
    #         return
    #
    #     if hasattr(self.viewer, 'create_crosshairs'):
    #         try:
    #             self.viewer.create_crosshairs()
    #         except Exception as e:
    #             QMessageBox.warning(self, "Error", f"An error occurred while creating crosshairs: {str(e)}")
    #     else:
    #         QMessageBox.warning(self, "Not Available", "Crosshair functionality is not available for this image viewer.")

    def open_settings(self):
        dialog = SettingsDialog(self, render_on_open=self.render_on_open)
        if dialog.exec_():
            settings = dialog.get_settings()
            self.render_on_open = settings['render_on_open']
            
    def closeEvent(self, event):
        # Perform any necessary cleanup
        if self.viewer is not None:
            self.viewer.close()  # Call the viewer's close method
        event.accept()