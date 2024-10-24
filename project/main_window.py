from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QMessageBox, QToolBar
from xray_viewer import XRayViewer
from ct_viewer import CTViewer
from patient_manage import PatientManageWindow
import SimpleITK as sitk
import os
from settings_dialog import SettingsDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/main_window.ui", self)  # Load the UI from XML file
        self.setWindowTitle("Medical Image Viewer")
        self.setGeometry(100, 100, 1200, 800)
        self.viewer = None  # Will hold the current image viewer
        self.render_on_open = False
        self.initUI()

    def initUI(self):
        # Create actions
        open_action = QAction('Open Image', self)
        open_action.triggered.connect(self.open_image)

        save_as_action = QAction('Save As', self)
        save_as_action.triggered.connect(self.save_image)
        save_as_action.setEnabled(False)  # Disabled initially

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

        # Save the action for later use
        self.save_as_action = save_as_action
        
        # Create a toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Add "Generate Model" button to the toolbar
        generate_model_action = QAction('Generate Model', self)
        generate_model_action.triggered.connect(self.generate_model)
        generate_model_action.setEnabled(False)
        toolbar.addAction(generate_model_action)
        self.generate_model_action = generate_model_action

        # Add "Create Crosshair" button to the toolbar
        create_crosshairs_action = QAction('Create Crosshair', self)
        create_crosshairs_action.triggered.connect(self.create_crosshairs)
        create_crosshairs_action.setEnabled(False)  # Initially disabled
        toolbar.addAction(create_crosshairs_action)
        self.create_crosshairs_action = create_crosshairs_action

        # Connect Patient Management button
        self.sure.clicked.connect(self.open_patient_manage)

        # Status bar
        self.statusBar().showMessage('Ready')

    def open_patient_manage(self):
        # Create and show the Patient Management window
        self.patient_manage_window = PatientManageWindow()
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
            elif dimension == 3:
                # Display 3D image
                self.viewer = CTViewer(self.image, render_model=self.render_on_open)
                self.generate_model_action.setEnabled(True)
                self.create_crosshairs_action.setEnabled(True)  # Enable crosshair button
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
            
    def generate_model(self):
        if hasattr(self.viewer, 'generate_and_display_model'):
            self.viewer.render_model = True
            self.viewer.generate_and_display_model()
        else:
            QMessageBox.warning(self, "Not Available", "Model generation is not available for this image.")

    def create_crosshairs(self):
        if self.viewer is None:
            QMessageBox.warning(self, "Not Available", "The image viewer is not initialized.")
            return
        
        if hasattr(self.viewer, 'create_crosshairs'):
            try:
                self.viewer.create_crosshairs()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"An error occurred while creating crosshairs: {str(e)}")
        else:
            QMessageBox.warning(self, "Not Available", "Crosshair functionality is not available for this image viewer.")

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
