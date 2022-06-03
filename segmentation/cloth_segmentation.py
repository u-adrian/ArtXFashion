import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from segmentation_models_pytorch.losses import DiceLoss

from segmentation.data_loading import load_data

CONFIG_PATH = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


def train_loop(args, model, lossFunc, optimizer, trainLoader, testLoader):
    weights_path = args["WEIGHTS_PATH"]
    num_epochs = args["NUM_EPOCHS"]
    stat_path = args["STAT_PATH"]

    H = {"train_loss": [], "test_loss": []}
    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(num_epochs)):
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
            torch.save(model.state_dict(), os.path.join(weights_path, f"test_E{e:03d}.pt"), )
        print("[INFO] EPOCH: {}/{}".format(e + 1, num_epochs))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    save_stats(model.state_dict(), os.path.join(weights_path, "test_E45.pt"), H, stat_path)


def save_stats(model_params, weights_path, losses, fig_path):
    torch.save(model_params, weights_path)

    plt.plot(losses["train_loss"], label='train_loss')
    plt.plot(losses["test_loss"], label='test_loss')
    plt.ylabel('BCEWithLogitsLoss')
    plt.legend()

    plt.savefig(os.path.join(fig_path,"losses.png"))


def test_model(args, epoch):
    weights_path = args["WEIGHTS_PATH"]
    output_classes = args["OUTPUT_CLASSES"]
    test_split = args["TEST_SPLIT"]
    input_image_height = args["INPUT_IMAGE_HEIGHT"]
    input_image_width = args["INPUT_IMAGE_WIDTH"]
    batch_size = args["BATCH_SIZE"]
    image_dataset_path = args["IMAGE_DATASET_PATH"]
    mask_dataset_path = args["MASK_DATASET_PATH"]

    # load model
    model = smp.Unet(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=output_classes,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    to_PIL = transforms.ToPILImage()

    pred_mask_path = "C:/Dev/Smart_Data/pred_mask"
    masks = torch.empty(0).to(DEVICE)
    checkpoint = torch.load(f"{weights_path}/test_E{epoch:03d}.pt")
    model.load_state_dict(checkpoint)

    # load data
    train_loader, test_loader = load_data(test_split, input_image_height=input_image_height,
                                        input_image_width=input_image_width,
                                        batch_size=batch_size, image_dataset_path=image_dataset_path,
                                        mask_dataset_path=mask_dataset_path)
    # create masks using model
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (b, (x, y)) in enumerate(test_loader):
            x = x.to(DEVICE)
            pred = torch.sigmoid(model(x)).cpu()
            for i in range(x.shape[0]):
                folder = f"img_{i:04d}"

                image_folder = os.path.join(pred_mask_path, folder)
                Path(image_folder).mkdir(parents=True, exist_ok=True)

                to_PIL(x[i]).save(os.path.join(image_folder, "ref_image.png"))
                print(pred.shape)
                pred_i = pred[i]

                for channel in range(pred_i.shape[0]):
                    img = pred_i[channel]
                    img = (img > 0.5)*255
                    img_int = img.type(torch.uint8)
                    img_2 = to_PIL(img_int)
                    img_2.show()


                    name = f"img_channel_{channel:03d}.png"

                    img_2.save(os.path.join(image_folder, name))

                    to_PIL(y[i][channel]).save(os.path.join(image_folder, f"ref_mask_{channel:03d}.png"))


            if i > 1:
                break


def train(args):
    print(args)
    output_classes = args["OUTPUT_CLASSES"]
    test_split = args["TEST_SPLIT"]
    input_image_height = args["INPUT_IMAGE_HEIGHT"]
    input_image_width = args["INPUT_IMAGE_WIDTH"]
    init_lr = args["INIT_LR"]
    batch_size = args["BATCH_SIZE"]
    image_dataset_path = args["IMAGE_DATASET_PATH"]
    mask_dataset_path = args["MASK_DATASET_PATH"]

    trainLoader, testLoader = load_data(test_split, input_image_height=input_image_height, input_image_width=input_image_width,
                                        batch_size=batch_size, image_dataset_path=image_dataset_path, mask_dataset_path=mask_dataset_path)

    model = smp.Unet(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=output_classes,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    lossFunc = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss("binary",from_logits=True)
    opt = Adam(model.parameters(), lr=init_lr)

    train_loop(args, model, dice_loss, opt, trainLoader, testLoader)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

        return config


def main(parser: argparse.ArgumentParser):
    parser.add_argument("--config_path", type=str, default="C:/Dev/Smart_Data/ArtXFashion/segmentation/config.json")
    config_path = parser.parse_args()
    args = load_config(config_path=config_path.config_path)

    #train(args)
    test_model(args,20)


if __name__ == "__main__":
    main(argparse.ArgumentParser())
