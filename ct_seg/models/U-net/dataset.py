# dataset.py
import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

class CTScanDataset(Dataset):
    def __init__(self, images_path1, images_path2, labels_path, transform=None, target_shape=(1, 512, 512, 512)):
        self.image_paths = (
            [os.path.join(images_path1, fname) for fname in os.listdir(images_path1)] +
            [os.path.join(images_path2, fname) for fname in os.listdir(images_path2)]
        )
        self.label_paths = [os.path.join(labels_path, fname) for fname in os.listdir(labels_path)]
        
        self.image_paths.sort()
        self.label_paths.sort()

        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels must match."

        self.transform = transform
        self.target_shape = target_shape  # Desired shape after padding

    def __len__(self):
        return len(self.image_paths)

    def pad_to_shape(self, img, target_shape):
        # Calculate padding sizes
        padding = [(0, max(0, ts - s)) for s, ts in zip(img.shape, target_shape)]
        # Apply padding
        return np.pad(img, padding, mode='constant', constant_values=0)

    def __getitem__(self, idx):
        # Load image and label using SimpleITK
        image = sitk.ReadImage(self.image_paths[idx])
        label = sitk.ReadImage(self.label_paths[idx])

        # Convert to numpy arrays
        image = sitk.GetArrayFromImage(image).astype(np.float32)
        label = sitk.GetArrayFromImage(label).astype(np.float32)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Pad to target shape
        image = self.pad_to_shape(image, self.target_shape)
        label = self.pad_to_shape(label, self.target_shape)

        # Convert to tensors
        image = torch.tensor(image)
        label = torch.tensor(label)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
