import albumentations as album

from segmentation.datasets.segmentation_ds import SegmentationDataset
from torch.utils.data import DataLoader
from imutils import paths
from sklearn.model_selection import train_test_split
from torchvision import transforms


def load_data(test_split_size, input_image_height, input_image_width, image_dataset_path, mask_dataset_path, batch_size, pin_memory=True):
    imagePaths = sorted(list(paths.list_images(image_dataset_path)))
    maskPaths = sorted(list(paths.list_images(mask_dataset_path)))
    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths, test_size=test_split_size, random_state=42)
    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    transform_mask_image = album.Compose([album.Resize(height=input_image_height, width=input_image_width),
                                          album.RandomResizedCrop(height=input_image_height, width=input_image_width,scale=(0.95,0.95)),
                                          album.HorizontalFlip(p=0.5),
                                          album.Rotate(p=1.0, limit=45),
                                          album.Resize(height=input_image_height, width=input_image_width)])

    color_transformation = transforms.Compose([transforms.ToPILImage(),
                                               transforms.ColorJitter(brightness=.5, hue=.3),
                                               transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))])

    color_transformation2 = transforms.Compose([transforms.ToPILImage(),
                                               transforms.ColorJitter(brightness=.2, hue=.1)])

    transform_mask_image_test = album.Compose([album.Resize(height=input_image_height, width=input_image_width)])

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
                                  transform_image_mask=transform_mask_image, color_transformation=color_transformation2)


    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
                                 transform_image_mask=transform_mask_image_test, color_transformation=None)

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=batch_size,
                             pin_memory=pin_memory, num_workers=0)

    testLoader = DataLoader(testDS, shuffle=False, batch_size=batch_size,
                            pin_memory=pin_memory, num_workers=0)

    return trainLoader, testLoader