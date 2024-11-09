import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import os

class CTScanDataset(Dataset):
    def __init__(self, images_path1, images_path2, labels_path, transform=None):
        self.images_paths = [
            *[os.path.join(images_path1, fname) for fname in os.listdir(images_path1)],
            *[os.path.join(images_path2, fname) for fname in os.listdir(images_path2)]
        ]
        self.labels_paths = [os.path.join(labels_path, fname) for fname in os.listdir(labels_path)]
        self.transform = transform

        # Sorting paths
        self.images_paths.sort()
        self.labels_paths.sort()

        # Ensure number of images and labels are the same
        assert len(self.images_paths) == len(self.labels_paths), "Mismatch between image and label counts."

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # Load image and label using SimpleITK
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.images_paths[idx]))
        label = sitk.GetArrayFromImage(sitk.ReadImage(self.labels_paths[idx]))

        # Resize depth to a fixed number (e.g., 256)
        target_depth = 256
        image = self.resize_depth(image, target_depth)
        label = self.resize_depth(label, target_depth)

        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32)

        # Add channel dimension (1, D, H, W)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Convert to tensor
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
            label = TF.resize(label, image.shape[1:])  # Resize label to match image

        return image, label

    def resize_depth(self, volume, target_depth):
        """Resize the depth of a 3D volume."""
        current_depth = volume.shape[0]
        if current_depth == target_depth:
            return volume

        # Using SimpleITK to resize along depth
        sitk_volume = sitk.GetImageFromArray(volume)
        original_spacing = sitk_volume.GetSpacing()
        original_size = sitk_volume.GetSize()

        # Compute new spacing
        new_size = list(original_size)
        new_size[2] = target_depth
        new_spacing = (
            original_spacing[0],
            original_spacing[1],
            original_spacing[2] * (original_size[2] / target_depth),
        )

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkLinear)

        resized_volume = resampler.Execute(sitk_volume)
        return sitk.GetArrayFromImage(resized_volume)
