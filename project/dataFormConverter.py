import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import SimpleITK as sitk
import os
import logging
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class MedicalImageConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Format Converter")
        self.root.geometry("500x350")

        self.input_files = []
        self.output_format = tk.StringVar(value="nii")
        self.log_file = "conversion_errors.log"

        # Set up logging
        logging.basicConfig(filename=self.log_file, level=logging.ERROR,
                            format='%(asctime)s:%(levelname)s:%(message)s')

        self.create_widgets()

    def create_widgets(self):
        # Input File Selection
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10)

        input_label = tk.Label(input_frame, text="Select Input Files:")
        input_label.pack(side=tk.LEFT)

        input_button = tk.Button(input_frame, text="Browse", command=self.select_input_files)
        input_button.pack(side=tk.LEFT, padx=5)

        # Output Format Selection
        format_frame = tk.Frame(self.root)
        format_frame.pack(pady=10)

        format_label = tk.Label(format_frame, text="Select Output Format:")
        format_label.pack(side=tk.LEFT)

        formats = [("NIfTI (.nii)", "nii"), ("NIfTI Compressed (.nii.gz)", "nii.gz"),
                   ("NRRD (.nrrd)", "nrrd"), ("MetaImage (.mha)", "mha"),
                   ("DICOM (.dcm)", "dcm"), ("PDF (.pdf)", "pdf")]

        for text, mode in formats:
            rb = tk.Radiobutton(format_frame, text=text, variable=self.output_format, value=mode)
            rb.pack(anchor=tk.W)

        # Convert Button
        convert_button = tk.Button(self.root, text="Convert", command=self.convert_files)
        convert_button.pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)

        # Log File Label
        log_label = tk.Label(self.root, text=f"Error log file: {self.log_file}")
        log_label.pack(pady=5)

    def select_input_files(self):
        file_types = [("All Supported Formats", "*.dcm *.nii *.nii.gz *.nrrd *.mha *.mhd"),
                      ("DICOM Files", "*.dcm"),
                      ("NIfTI Files", "*.nii *.nii.gz"),
                      ("NRRD Files", "*.nrrd"),
                      ("MetaImage Files", "*.mha *.mhd")]

        self.input_files = filedialog.askopenfilenames(title="Select Input Files", filetypes=file_types)
        if self.input_files:
            messagebox.showinfo("Selected Files", f"{len(self.input_files)} files selected.")

    def convert_files(self):
        if not self.input_files:
            messagebox.showwarning("No Files Selected", "Please select input files to convert.")
            return

        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return

        self.progress['maximum'] = len(self.input_files)
        self.progress['value'] = 0

        for idx, input_path in enumerate(self.input_files):
            try:
                # Read the input image
                image = sitk.ReadImage(input_path)

                # Generate output file name
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                output_extension = self.output_format.get()

                # Handle compressed NIfTI
                if output_extension == "nii.gz":
                    output_filename = base_name + ".nii.gz"
                elif output_extension == "dcm":
                    output_filename = base_name  # DICOM files may not have extensions per slice
                else:
                    output_filename = base_name + "." + output_extension

                output_path = os.path.join(output_dir, output_filename)

                # Write the image in the desired format
                if output_extension == "dcm":
                    self.write_dicom(image, os.path.join(output_dir, base_name))
                elif output_extension == "pdf":
                    self.save_image_to_pdf(image, output_path)
                else:
                    sitk.WriteImage(image, output_path)

            except Exception as e:
                error_message = f"Error converting {input_path}:\n{str(e)}"
                logging.error(error_message)
                messagebox.showerror("Conversion Error", error_message)
                continue
            finally:
                self.progress['value'] = idx + 1
                self.root.update_idletasks()

        messagebox.showinfo("Conversion Complete", "All files have been converted successfully.")

    def write_dicom(self, image, output_dir):
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dimension = image.GetDimension()

        if dimension == 3:
            # Write each slice as a separate DICOM file
            size = image.GetSize()
            for i in range(size[2]):
                try:
                    slice_i = image[:, :, i]
                    slice_filename = os.path.join(output_dir, f"slice_{i}.dcm")
                    sitk.WriteImage(slice_i, slice_filename)
                except Exception as e:
                    error_message = f"Error writing DICOM slice {i}:\n{str(e)}"
                    logging.error(error_message)
        else:
            # For 2D images
            try:
                sitk.WriteImage(image, output_dir + ".dcm")
            except Exception as e:
                error_message = f"Error writing DICOM file:\n{str(e)}"
                logging.error(error_message)

    def save_image_to_pdf(self, image, output_path):
        try:
            if image.GetDimension() == 3:
                # Convert 3D image to 2D slices
                img_array = sitk.GetArrayFromImage(image)
                num_slices = img_array.shape[0]

                plt.figure(figsize=(8, 8))
                with PdfPages(output_path) as pdf:
                    for i in range(num_slices):
                        plt.imshow(img_array[i, :, :], cmap='gray')
                        plt.axis('off')
                        pdf.savefig()
                        plt.close()
            else:
                # Handle 2D images
                img_array = sitk.GetArrayFromImage(image)
                plt.figure(figsize=(8, 8))
                plt.imshow(img_array, cmap='gray')
                plt.axis('off')
                plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0)
                plt.close()
        except Exception as e:
            error_message = f"Error saving to PDF:\n{str(e)}"
            logging.error(error_message)

if __name__ == "__main__":
    try:
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        messagebox.showerror("Import Error", "matplotlib is required for PDF conversion.")
        exit()

    root = tk.Tk()
    app = MedicalImageConverter(root)
    root.mainloop()
