import math
import os
import time

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn as nn
from torch.optim import Adam
from tqdm import tqdm

from segmentation.globals import DEVICE
from segmentation.utils import save_stats
from segmentation.data_loading import load_data2


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
    meta_data_path = "C:/Dev/Smart_Data/Clothing_Segmentation/color_lists_reduced"

    step_size = args["STEP_SIZE"]
    gamma = args["GAMMA"]

    #trainLoader, testLoader = load_data(test_split, input_image_height=input_image_height, input_image_width=input_image_width,
    #                                    batch_size=batch_size, image_dataset_path=image_dataset_path, mask_dataset_path=mask_dataset_path)

    trainLoader, testLoader = load_data2(test_split, input_image_height=input_image_height, input_image_width=input_image_width,
                                         batch_size=batch_size, image_dataset_path=image_dataset_path,
                                         mask_dataset_path=mask_dataset_path, meta_data_path=meta_data_path)

    model = smp.Unet(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=output_classes,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=output_classes,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)


    lossFunc = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss("binary", from_logits=True)
    opt = Adam(model.parameters(), lr=init_lr)
    #opt = RMSprop(model.parameters(), lr=init_lr)

    def my_loss(pred, label):
        return (dice_loss(pred, label) + lossFunc(pred, label)) / 2

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=step_size, gamma=gamma, verbose=True)

    train_loop_for_marker(args, model, my_loss, opt, trainLoader, testLoader, scheduler, loss_to_track=dice_loss)

def train_loop_for_marker(args, model, loss_func_to_opt, optimizer, trainLoader, testLoader, scheduler, loss_to_track = None):
    weights_path = args["WEIGHTS_PATH"]
    num_epochs = args["NUM_EPOCHS"]
    stat_path = args["STAT_PATH"]

    best_model = model
    best_loss = math.inf
    best_epoch = 0

    history = {"train_loss": [], "test_loss": [], "train_loss_additional": [], "test_loss_additional": []}
    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(num_epochs)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

        totalTrainLoss2 = 0
        totalTestLoss2 = 0

        # loop over the training set
        trainSteps = None
        testSteps = None
        for (i, (image, marker, y)) in enumerate(trainLoader):
            image_and_marker = torch.cat((image, marker), dim=1)
            image_and_marker = image_and_marker.to(DEVICE)
            (image, y) = (image.to(DEVICE), y.to(DEVICE))

            pred = model(image_and_marker)
            loss = loss_func_to_opt(pred, y.type(torch.float))

            with torch.no_grad():
                loss2 = loss_to_track(pred, y.type(torch.float))
                totalTrainLoss2 += loss2
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
            for (i, (image, marker, y)) in enumerate(testLoader):
                image_and_marker = torch.cat((image, marker), dim=1)
                image_and_marker = image_and_marker.to(DEVICE)
                (image, y) = (image.to(DEVICE), y.to(DEVICE))
                # make the predictions and calculate the validation loss
                pred = model(image_and_marker)
                totalTestLoss += loss_func_to_opt(pred, y.type(torch.float))
                with torch.no_grad():
                    loss2 = loss_to_track(pred, y.type(torch.float))
                    totalTestLoss2 += loss2
                testSteps = i
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        avgTrainLoss2 = totalTrainLoss2 / trainSteps
        avgTestLoss2 = totalTestLoss2 / testSteps

        if avgTestLoss < best_loss:
            print("New best model found!")
            best_loss = avgTestLoss
            best_model = model.state_dict()
            best_epoch = e
        # update our training history
        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        history["train_loss_additional"].append(avgTrainLoss2.cpu().detach().numpy())
        history["test_loss_additional"].append(avgTestLoss2.cpu().detach().numpy())
        # print the model training and validation information
        if e % 5 == 0:
            torch.save(model.state_dict(), os.path.join(weights_path, f"test_E{e:03d}.pt"), )
        print("[INFO] EPOCH: {}/{}".format(e + 1, num_epochs))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
        print("Train loss 2: {:.6f}, Test loss 2: {:.4f}".format(
            avgTrainLoss2, avgTestLoss2))

        print(f"Best Epoch: {best_epoch}")
        scheduler.step()
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    save_stats(model.state_dict(), os.path.join(weights_path, "test_E45.pt"), history, stat_path)
    torch.save(best_model, os.path.join(weights_path, f"best_model_E_{best_epoch}.pt"))


def train_loop(args, model, lossFunc, optimizer, trainLoader, testLoader, scheduler):
    weights_path = args["WEIGHTS_PATH"]
    num_epochs = args["NUM_EPOCHS"]
    stat_path = args["STAT_PATH"]

    history = {"train_loss": [], "test_loss": []}
    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(num_epochs)):
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
        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        # print the model training and validation information
        if e % 5 == 0:
            torch.save(model.state_dict(), os.path.join(weights_path, f"test_E{e:03d}.pt"), )
        print("[INFO] EPOCH: {}/{}".format(e + 1, num_epochs))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))

        scheduler.step()
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    save_stats(model.state_dict(), os.path.join(weights_path, "test_E45.pt"), history, stat_path)