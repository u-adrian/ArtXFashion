import os
import time

import numpy as np
import pandas as pd
import torch
from imutils import paths
from torch.utils.data import DataLoader
from torchvision import transforms

from segmentation.datasets.mask_ds import MaskDataset

OLD_MASK_DATASET_PATH = "C:/Dev/Smart_Data/Clothing_Segmentation/archive/labels/pixel_level_labels_colored"
COLOR_LIST_PATH = "C:/Dev/Smart_Data/Clothing_Segmentation/color_lists_reduced"

NEW_DATASET_PATH = ""
INPUT_IMAGE_HEIGHT = 820
INPUT_IMAGE_WIDTH = 550


def load_masks_only():
    maskPaths = sorted(list(paths.list_images(OLD_MASK_DATASET_PATH)))

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])
    # create the train and test datasets
    ds = MaskDataset(maskPaths=maskPaths, transforms=transform)

    print(f"[INFO] found {len(ds)} examples in the test set...")

    # create the training and test data loaders
    data_loader = DataLoader(ds, shuffle=False, batch_size=1,
                             pin_memory=True, num_workers=0)

    return data_loader


def create_color_list():
    data_loader = load_masks_only()

    name = 1 ### Yes name is an int
    for (i, masks) in enumerate(data_loader):
        for single_mask in masks:
            mask_flat = torch.flatten(single_mask,start_dim=1,end_dim=-1)
            colors = torch.unique(mask_flat, dim=1).T

            torch.save(colors, f=os.path.join(COLOR_LIST_PATH, f"{name:04d}"))
            name += 1


def create_color_list_with_overlap(color_list: torch.Tensor):
    data_loader = load_masks_only()

    name = 1 ### Yes name is an int
    for (i, masks) in enumerate(data_loader):
        print("batch:", i)
        for single_mask in masks:
            mask_flat = torch.flatten(single_mask,start_dim=1,end_dim=-1)
            colors = torch.unique(mask_flat, dim=1).T
            intersection = torch.empty((0,3))
            for color in colors:
                eq = torch.eq(color[:][None], color_list)
                all = torch.all(eq, dim=-1)
                any = torch.any(all)
                if any:
                    intersection = torch.cat((intersection, color[:][None]))
            torch.save(intersection, f=os.path.join(COLOR_LIST_PATH, f"{name:04d}"))
            name += 1


if __name__ == "__main__":
    labels_to_drop = [
        "accessories",
        "belt",
        "bracelet",
        "earrings",
        "glasses",
        "hair",
        "ring",
        "skin",
        "socks",
        "sunglasses",
        "wallet",
        "watch"
    ]

    df = pd.read_csv("C:/Dev/Smart_Data/Clothing_Segmentation/archive/class_dict.csv")
    df.iloc[0, 0] = "background"
    df = df.set_index('class_name')
    df = df.drop(labels_to_drop)

    label_colors_to_extract = torch.tensor(df.values)
    colors = label_colors_to_extract / 255
    create_color_list_with_overlap(colors)
