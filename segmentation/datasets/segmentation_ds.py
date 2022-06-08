import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from segmentation.utils import pixel_mask, create_marker_mask


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
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
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


class SegmentationDataset_with_Marker(Dataset):
    def __init__(self, imagePaths, maskPaths, meta_data_path, transform, color_transformation, randomization=True):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.meta_data_path = meta_data_path
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.color_transformation = color_transformation
        self.transform = transform
        self.randomization = randomization

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        colors = torch.load(self.meta_data_path[idx]) * 255

        image = cv2.imread(self.imagePaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = to_tensor(image)
        image = self.transform(image)

        rgb_mask = cv2.imread(self.maskPaths[idx])
        rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
        rgb_mask = to_tensor(rgb_mask)

        color_list = colors

        if not self.randomization:
            random_number = idx % color_list.shape[0]
        else:
            random_number = int(torch.rand(size=[1]).item() * color_list.shape[0])
        color = color_list[random_number]

        binary_mask = pixel_mask(rgb_mask, color)   # on GPU
        binary_mask = self.transform(binary_mask[:][None])

        marker = create_marker_mask(binary_mask).squeeze()[:][None]     # on GPU

        # check to see if we are applying any transformations

        if self.color_transformation is not None:
            image = self.color_transformation(image)
            image = to_tensor(image)

        #image_and_mask = torch.cat((image, marker), dim=0)
        #print("IMAGE:", image.shape)
        #print("MARKER:", marker.shape)
        #to_pil(rgb_mask).show()
        #to_pil(binary_mask*255).show()
        #to_pil(marker * 255).show()

        return image, marker, binary_mask
