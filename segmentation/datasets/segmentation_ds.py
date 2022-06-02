import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transform_image_mask, color_transformation, is_image=True):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transform_image_mask = transform_image_mask
        self.color_transformation = color_transformation
        self.is_image = is_image

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        image = cv2.imread(self.imagePaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_image:
            mask = cv2.imread(self.maskPaths[idx])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask = np.load(self.maskPaths[idx])

        # check to see if we are applying any transformations
        if self.transform_image_mask is not None:
            transformed = self.transform_image_mask(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if self.color_transformation is not None:
            image = self.color_transformation(image)
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)

        return image, mask