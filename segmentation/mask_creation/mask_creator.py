import os
import time

import numpy as np
import pandas as pd
from imutils import paths
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from segmentation.cloth_segmentation import OLD_MASK_DATASET_PATH, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, BATCH_SIZE, \
    PIN_MEMORY
from segmentation.datasets.mask_ds import MaskDataset


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
            #np.save(file=os.path.join(path, f"{name:04d}"), arr=single_mask)
            #print(single_mask.squeeze().shape)
            img = Image.fromarray(np.uint8(single_mask.squeeze()*255))
            img.save(os.path.join(path, f"{name:04d}.png"))
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
                             pin_memory=PIN_MEMORY, num_workers=0)

    return data_loader