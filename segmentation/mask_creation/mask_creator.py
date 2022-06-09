import os
import time

import numpy as np
import pandas as pd
from imutils import paths
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from segmentation.datasets.mask_ds import MaskDataset

OLD_MASK_DATASET_PATH = "C:/Dev/Smart_Data/Clothing_Segmentation/archive/labels/pixel_level_labels_colored"
INPUT_IMAGE_HEIGHT = 820
INPUT_IMAGE_WIDTH = 550
BATCH_SIZE = 64


def create_new_masks(path):
    encoding = {
        "blazer": 0,
        "blouse": 0,
        "bodysuit": 0,
        "cardigan": 0,
        "coat": 0,
        "dress": 0,
        "hoodie": 0,
        "jacket": 0,
        "jeans": 0,
        "jumper": 0,
        "leggings": 0,
        "pants": 0,
        "shirt": 0,
        "shorts": 0,
        "skirt": 0,
        "suit": 0,
        "sweater": 0,
        "sweatshirt": 0,
        "t-shirt": 0,
        "tights": 0,
        "top": 0,
        "vest": 0
    }

    encoding = {
        "background": 0
    }

    encoding = {
        "dress": 0
    }

    encoding = {
        "t-shirt": 0
    }

    encoding = {
        "t-shirt": 0,
        "jeans": 1,
        "pants": 1,
        "shorts": 1,
        "hat": 2
    }

    # Schuhe: 0
    # Hose: 1
    # Pullover: 2
    encoding = {
        "boots": 0,
        "clogs": 0,
        "flats": 0,
        "heels": 0,
        "loafers": 0,
        "pumps": 0,
        "sandals": 0,
        "shoes": 0,
        "sneakers": 0,
        "wedges": 0,

        "leggings": 1,
        "panties": 1,
        "pants": 1,
        "shorts": 1,
        "tights": 1,

        "sweater": 2,
        "sweatshirt": 2
    }

    encoding = {
        "boots": 0,
        "clogs": 0,
        "flats": 0,
        "heels": 0,
        "loafers": 0,
        "pumps": 0,
        "sandals": 0,
        "shoes": 0,
        "sneakers": 0,
        "wedges": 0,

        "leggings": 1,
        "panties": 1,
        "pants": 1,
        "shorts": 1,
        "tights": 1,

        "sweater": 2,
        "sweatshirt": 2,
        "hoodie": 2,

        "blazer": 3,
        "blouse": 4,
        "bodysuit": 5,
        "cape": 6,
        "cardigan": 7,
        "coat": 8,
        "dress": 9,
        "jacket": 10,
        "jeans": 11,
        "jumper": 12,
        "romper": 13,
        "shirt": 14,
        "skirt": 15,
        "socks": 16,
        "stockings": 17,
        "suit": 18,
        "t-shirt": 19,
        "top": 20,
        "vest": 21
    }

    data_loader = load_masks_only()
    one_hot_encoding_func = create_mapping_function(encoding)
    name = 1 ### Yes name is an int
    for (i, mask) in enumerate(data_loader):
        start_time = time.time()
        print(f"Starting transformation of batch {i} of {len(data_loader)}")
        mask = np.asarray((mask * 255), dtype=int).transpose(0, 2, 3, 1)
        new_masks = one_hot_encoding_func(mask)
        time_for_step = time.time() - start_time
        print(f"The transformation took {time_for_step}")
        print(f"Storing the images at: {path}")
        for single_mask in new_masks:
            np.save(file=os.path.join(path, f"{name:04d}"), arr=single_mask)
            #print(single_mask.squeeze().shape)
            #img = Image.fromarray(np.uint8(single_mask.squeeze()*255))
            #img.save(os.path.join(path, f"{name:04d}.png"))
            name += 1


def create_mapping_function(encoding):
    out_channels = np.array(list(encoding.values())).max() + 1

    df = pd.read_csv("C:/Dev/Smart_Data/Clothing_Segmentation/archive/class_dict.csv")
    df.iloc[0, 0] = "background"
    df = df.set_index('class_name')

    mapping = - np.ones((256, 256, 256), dtype=int)

    for enc in encoding:
        r = df.loc[enc]["r"]
        g = df.loc[enc]["g"]
        b = df.loc[enc]["b"]
        mapping[r][g][b] = encoding.get(enc)
    func = lambda rgb: np.eye(1, out_channels, mapping[rgb[0]][rgb[1]][rgb[2]], dtype=np.byte)
    vec_func = np.vectorize(func, signature='(c)->(d)')

    return vec_func


def load_masks_only():
    maskPaths = sorted(list(paths.list_images(OLD_MASK_DATASET_PATH)))

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
                                    transforms.ToTensor()])
    # create the train and test datasets
    ds = MaskDataset(maskPaths=maskPaths, transforms=transform)

    print(f"[INFO] found {len(ds)} examples in the test set...")

    # create the training and test data loaders
    data_loader = DataLoader(ds, shuffle=False, batch_size=BATCH_SIZE,
                             pin_memory=True, num_workers=0)

    return data_loader

if __name__ == "__main__":
    create_new_masks("C:/Dev/Smart_Data/new_masks/all_test01")