# CT Segmentation Project

This repository contains the setup for a pelvis CT segmentation project using deep learning models such as U-Net and its variants. The project aims to provide an effective framework for medical image segmentation, utilizing a well-organized folder structure for data, models, and results.

## Folder Structure

```
ct_seg/
│
├── data/             # Contains CT scan data and corresponding labels.
│   ├── PENGWIN_CT_train_images/   # Folder with training images for CT segmentation.combined part 1 2. 
│   ├── PENGWIN_CT_train_labels/   # Folder with labels for the training images.
│
├── models/           # Directory for different trained models, including their outputs.
│   ├── U-net/                  # Folder containing the U-Net model, checkpoints, and outputs.
│   │   ├── checkpoints/        # Checkpoints for the U-Net model to save training progress.
│   │   ├── predictions/        # Predicted segmentation outputs by the U-Net model on test data.
│   │   ├── metrics/            # Evaluation results for the U-Net model, such as accuracy and IoU.
│
├── README.md         # Documentation for the project.
│
├── requirements.txt  # List of Python dependencies for the project.

```

## Getting Started

### Prerequisites

This project requires the following software to be installed:

- Python 3.10
- Anaconda (optional but recommended)
- CUDA 11.8 for GPU acceleration

### Installation

1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd ct_seg
   ```
2. Set up a virtual environment with Anaconda:

   ```bash
   conda create -n ct_seg python=3.10
   conda activate ct_seg
   ```
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Preparing Data

1. Place the CT scan data and corresponding labels into the `data/` folder.
2. Make sure that your data is properly organized for training and evaluation.

### Training a Model

Run the training script to train the U-Net model on the dataset:

```bash
python src/train.py
```

Make sure that the data paths inside the training script point to the correct directories in `ct_seg/data`.

### Evaluating a Model

To evaluate the model, run:

```bash
python src/evaluate.py
```

This will generate evaluation metrics and save outputs to the `outputs/` folder.

## Dependencies

The project requires the following Python packages, which are listed in `requirements.txt`:

- torch
- torchvision
- torchaudio
- SimpleITK
- numpy
- matplotlib

## Usage

- Modify the `train.py` and `evaluate.py` scripts to point to the correct data and model paths.
- Store different versions of models inside the `models/` directory.
- The `outputs/` folder will contain prediction results, logs, and other generated output files.

## Contributing

Feel free to contribute to this project by forking the repository and submitting a pull request. Any improvements, such as new models or enhancements to the training pipeline, are welcome!

## License

This project is licensed under the MIT License.
