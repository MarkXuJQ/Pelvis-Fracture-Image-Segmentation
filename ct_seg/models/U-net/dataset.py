import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import numpy as np

class CTScanDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None, target_shape=(1, 128, 128, 128)):
        # Load paths for images and labels
        self.image_paths = [os.path.join(images_path, fname) for fname in os.listdir(images_path)]
        self.label_paths = [os.path.join(labels_path, fname) for fname in os.listdir(labels_path)]
        
        # Sort the paths to ensure matching order
        self.image_paths.sort()
        self.label_paths.sort()

        # Ensure that the number of images and labels match
        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels must match."

        self.transform = transform
        self.target_shape = target_shape

    def pad_or_resize_to_shape(self, img, target_shape):
        """Pads or crops an image to match the target shape."""
        current_shape = img.shape
        pad = [(0, max(0, target_shape[i] - current_shape[i])) for i in range(len(target_shape))]
        
        # Padding if the dimension is smaller than the target shape
        img = np.pad(img, pad, mode='constant', constant_values=0)

        # Crop if the dimension is larger than the target shape
        slices = tuple(slice(0, min(current_shape[i], target_shape[i])) for i in range(len(target_shape)))
        return img[slices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label using SimpleITK
        image = sitk.ReadImage(self.image_paths[idx])
        label = sitk.ReadImage(self.label_paths[idx])

        # Convert to numpy arrays
        image = sitk.GetArrayFromImage(image).astype(np.float32)
        label = sitk.GetArrayFromImage(label).astype(np.float32)

        # Clip and normalize image values
        image = np.clip(image, -1023, 6153)
        image = (image + 1023) / (6153 + 1023)  # Normalize to [0, 1]

        # Binarize label (0 or 1)
        label = (label > 0).astype(np.float32)

        # Add channel dimension and pad or resize to target shape
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Ensure both image and label are of the target shape
        image = self.pad_or_resize_to_shape(image, self.target_shape)
        label = self.pad_or_resize_to_shape(label, self.target_shape)

        # Convert to PyTorch tensors
        image = torch.tensor(image)
        label = torch.tensor(label)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
