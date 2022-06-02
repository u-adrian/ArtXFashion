import cv2
from torch.utils.data import Dataset


class MaskDataset(Dataset):
    def __init__(self, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.maskPaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        maskPath = self.maskPaths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        mask = cv2.imread(self.maskPaths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            mask = self.transforms(mask)
        # return a tuple of the image and its mask
        return mask