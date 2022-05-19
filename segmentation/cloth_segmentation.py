import os
import time

import numpy as np
import pandas
import segmentation_models_pytorch as smp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import pandas as pd
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)
        # return a tuple of the image and its mask
        return image, mask

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



NUM_EPOCHS = 50
INIT_LR = 0.001
BATCH_SIZE = 32
TEST_SPLIT = 0.2
IMAGE_DATASET_PATH = "C:/Dev/Smart_Data/Clothing_Segmentation/archive/images"
MASK_DATASET_PATH = "C:/Dev/Smart_Data/Clothing_Segmentation/archive/labels/pixel_level_labels_colored"
WEIGHTS_PATH = "C:/Dev/Smart_Data/Network_Weights"
STAT_PATH = "C:/Dev/Smart_Data/Network_Weights/Stat"
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


# define transformations
def load_data(test_split_size):
    imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))
    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths, test_size=test_split_size, random_state=42)
    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
                                     transforms.ToTensor()])
    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transform)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transform)

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=BATCH_SIZE,
                             pin_memory=PIN_MEMORY, num_workers=0)

    testLoader = DataLoader(testDS, shuffle=False, batch_size=BATCH_SIZE,
                            pin_memory=PIN_MEMORY, num_workers=0)

    return trainLoader, testLoader

def load_masks_only():
    maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))


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

def train(model, lossFunc, optimizer, trainLoader, testLoader):
    H = {"train_loss": [], "test_loss": []}
    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
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
            # perform a forward pass and calculate the training loss
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
            torch.save(model.state_dict(), os.path.join(WEIGHTS_PATH, f"test_E{e:03d}.pt"),)
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    save_stats(model.state_dict(), os.path.join(WEIGHTS_PATH, "test_E25.pt"), H, os.path.join(STAT_PATH, "loss.png"))


def save_stats(model_params, weights_path, losses, fig_path):
    torch.save(model_params, weights_path)

    with open(os.path.join(fig_path, "losses_dict"), 'w') as f:
        json.dump(losses, f)

    plt.plot(losses["train_loss"], label='train_loss')
    plt.plot(losses["test_loss"], label='test_loss')
    plt.ylabel('BCEWithLogitsLoss')
    plt.legend()

    plt.savefig(fig_path)

def test_model():
    #load model
    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    pred_mask_path = "C:/Dev/Smart_Data/pred_mask"
    masks = torch.empty(0).to(DEVICE)
    to_PIL_Image = transforms.ToPILImage()
    checkpoint = torch.load(f"{WEIGHTS_PATH}/test.pt")
    model.load_state_dict(checkpoint)

    #load data
    data_loader, _ = load_data(test_split_size=TEST_SPLIT)
    #create masks using model
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (i, (x,_)) in enumerate(data_loader):
            # send the input to the device
            print(i)
            x = x.to(DEVICE)
            print("x: ", x.shape)
            # make the predictions and calculate the validation loss
            pred = model(x)
            print("pred: ",pred.shape)
            masks = torch.concat((masks, pred))
            print("masks: ", masks.shape)

    #storing masks
    for i in range(len(masks)):
        name = f"fish_{i}.png"

        predMask = np.asarray(masks[i].cpu()).transpose(1,2,0)
        predMask = (predMask) * 255
        predMask = predMask.astype(np.uint8)
        print("predMask: ", predMask.shape)
        image = to_PIL_Image(predMask.squeeze())
        image.save(os.path.join(pred_mask_path, name))

def main():
    trainLoader, testLoader = load_data(TEST_SPLIT)

    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    lossFunc = BCEWithLogitsLoss()
    opt = Adam(model.parameters(), lr=INIT_LR)

    train(model, lossFunc, opt, trainLoader, testLoader)

def load_class_labels():
    encoding = {
        "boots": 1,
        "hair": 2
    }
    default_encoding = 0
    out_channels = len(encoding) + 1

    df = pd.read_csv("C:/Dev/Smart_Data/Clothing_Segmentation/archive/class_dict.csv")
    df.iloc[0,0] = "background"
    df = df.set_index('class_name')
    df["int_encoding"] = 0
    df["one_hot_encoding"] = 0
    df["rgb"] = 0
    #df["one_hot_encoding"] = df["one_hot_encoding"].astype(object)

    # using dict here since it is much simpler
    df_as_dict = df.transpose().to_dict()

    for x in df_as_dict:
        int_code = default_encoding
        if x in encoding:
            int_code = encoding[x]

        one_hot = np.zeros(out_channels, dtype=int)
        one_hot[int_code] = 1
        df_as_dict[x]["int_encoding"] = int_code
        df_as_dict[x]["one_hot_encoding"] = one_hot

        rgb = np.array([df_as_dict[x]["r"],df_as_dict[x]["g"],df_as_dict[x]["b"]],dtype=int)
        df_as_dict[x]["rgb"] = rgb

    df = pandas.DataFrame.from_dict(df_as_dict).transpose()
    return df_as_dict


def create_new_masks():
    data_loader = load_masks_only()
    df_as_dict = load_class_labels()
    print(df_as_dict)
    for (i, mask) in enumerate(data_loader):
        mask = np.asarray((mask*255),dtype=int).transpose(0,2,3,1)
        new_mask = np.zeros(mask.shape)
        #for label in df_as_dict:
        label = "hair"
        print(label)
        one_hot = np.asarray(df_as_dict[label]["one_hot_encoding"],dtype=int)
        rgb = np.asarray(df_as_dict[label]["rgb"],dtype=int)

        func = lambda x: one_hot if (np.array_equal(x, rgb)) else np.zeros(one_hot.shape)
        vfunc = np.vectorize(func, signature='(c)->(d)')
        new_mask += vfunc(mask)
        print(new_mask.shape)

        image0 = new_mask[0]*255
        image1 = new_mask[1] * 255
        image2 = new_mask[2] * 255
        image3 = new_mask[3] * 255

        print(image0.shape)
        print(type(image0))
        pil_img = Image.fromarray(np.uint8(image0))
        pil_img.show()

        pil_img = Image.fromarray(np.uint8(image1))
        pil_img.show()

        pil_img = Image.fromarray(np.uint8(image2))
        pil_img.show()

        pil_img = Image.fromarray(np.uint8(image3))
        pil_img.show()


def test():
    arr = np.array([[[1,2],[1,2],[1,3],[1,2]],
                    [[1,2],[1,2],[1,3],[1,2]],
                   [[1,2],[1,2],[1,3],[1,2]]])
    new_mask = np.array([0,1,5])
    ref = np.array([1,3])
    func = lambda x: new_mask if (np.array_equal(x, ref)) else np.zeros(new_mask.shape)
    #func = lambda mask: mask*2
    vf = np.vectorize(func,signature='(c)->(d)')
    print(arr.shape)
    print("-------------")
    arr=vf(arr)
    print(arr)

if __name__ == "__main__":
    #main()
    #test_model()
    #load_class_labels()
    create_new_masks()
