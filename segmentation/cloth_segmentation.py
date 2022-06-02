import os
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from imutils import paths
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as album

from segmentation.datasets.segmentation_ds import SegmentationDataset

from segmentation_models_pytorch.losses import DiceLoss

NUM_EPOCHS = 31
OUTPUT_CLASSES = 1
INIT_LR = 0.001
BATCH_SIZE = 32
TEST_SPLIT = 0.2
IMAGE_DATASET_PATH = "C:/Dev/Smart_Data/Clothing_Segmentation/archive/images"
OLD_MASK_DATASET_PATH = "C:/Dev/Smart_Data/Clothing_Segmentation/archive/labels/pixel_level_labels_colored"
MASK_DATASET_PATH = "C:/Dev/Smart_Data/new_masks/test03"
WEIGHTS_PATH = "C:/Dev/Smart_Data/Network_Weights"
STAT_PATH = "C:/Dev/Smart_Data/Network_Weights/Stat"
INPUT_IMAGE_HEIGHT = 256*2
INPUT_IMAGE_WIDTH = 128*2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


# define transformations
def load_data(test_split_size, input_image_height, input_image_width):
    imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))
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
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=BATCH_SIZE,
                             pin_memory=PIN_MEMORY, num_workers=0)

    testLoader = DataLoader(testDS, shuffle=False, batch_size=BATCH_SIZE,
                            pin_memory=PIN_MEMORY, num_workers=0)

    return trainLoader, testLoader


def train(model, lossFunc, optimizer, trainLoader, testLoader):
    H = {"train_loss": [], "test_loss": []}
    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        if e == 60:
            print("ONLY ONCE")

        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set
        trainSteps = None
        testSteps = None
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))

            pred = model(x)
            loss = lossFunc(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
            trainSteps = i
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (i, (x, y)) in enumerate(testLoader):
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalTestLoss += lossFunc(pred, y)
                testSteps = i
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        # print the model training and validation information
        if e % 5 == 0:
            torch.save(model.state_dict(), os.path.join(WEIGHTS_PATH, f"test_E{e:03d}.pt"), )
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    save_stats(model.state_dict(), os.path.join(WEIGHTS_PATH, "test_E45.pt"), H, STAT_PATH)


def save_stats(model_params, weights_path, losses, fig_path):
    torch.save(model_params, weights_path)

    #with open(os.path.join(fig_path, "losses_dict"), 'w') as f:
    #    json.dump(losses, f)

    plt.plot(losses["train_loss"], label='train_loss')
    plt.plot(losses["test_loss"], label='test_loss')
    plt.ylabel('BCEWithLogitsLoss')
    plt.legend()

    plt.savefig(os.path.join(fig_path,"losses.png"))

def test_model():
    # load model
    model = smp.Unet(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=OUTPUT_CLASSES,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    pred_mask_path = "C:/Dev/Smart_Data/pred_mask"
    masks = torch.empty(0).to(DEVICE)
    to_PIL_Image = transforms.ToPILImage()
    checkpoint = torch.load(f"{WEIGHTS_PATH}/test_E020.pt")
    model.load_state_dict(checkpoint)

    # load data
    data_loader, test_loader = load_data(test_split_size=TEST_SPLIT, input_image_height=INPUT_IMAGE_HEIGHT, input_image_width=INPUT_IMAGE_WIDTH)
    # create masks using model
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (i, (x, _)) in enumerate(test_loader):
            # send the input to the device
            print(i)
            x = x.to(DEVICE)
            print("x: ", x.shape)
            # make the predictions and calculate the validation loss
            pred = torch.sigmoid(model(x))
            print("pred: ", pred.shape)
            masks = torch.concat((masks, pred))
            print("masks: ", masks.shape)
            if i > 5:
                break;

    # storing masks
    #for i in range(len(masks)):
    for i in range(5):
        folder = f"img_{i:04d}"

        predMask = np.asarray(masks[i].cpu()).transpose(1, 2, 0)
        predMask = predMask * 255
        predMask = predMask.astype(np.uint8)

        path_folder = os.path.join(pred_mask_path, folder)
        Path(path_folder).mkdir(parents=True, exist_ok=True)
        for single_image in range(predMask.shape[2]):
            #img2 = Image.fromarray(single_image)
            #img2.show()
            img = predMask.T[single_image].T / 255
            #print("Uniques: ", np.unique(img))
            img = (img>0.5)*255
            img = Image.fromarray(np.uint8(img))

            name = f"img_channel{single_image}.png"

            img.save(os.path.join(path_folder, name))


def main():
    trainLoader, testLoader = load_data(TEST_SPLIT, input_image_height=256, input_image_width=128)

    model = smp.Unet(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=OUTPUT_CLASSES,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    lossFunc = nn.BCEWithLogitsLoss()
    #lossFunc = nn.L1Loss()
    dice_loss = DiceLoss("binary",from_logits=True)
    opt = Adam(model.parameters(), lr=INIT_LR)

    train(model, dice_loss, opt, trainLoader, testLoader)


if __name__ == "__main__":
    # create_new_masks("C:/Dev/Smart_Data/new_masks/test_multi")
    #main()
    test_model()
    # test()

    # create_img()
    # test_test()
