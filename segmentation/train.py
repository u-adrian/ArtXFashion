import math
import os
import time

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn as nn
from torch.optim import Adam
from tqdm import tqdm

from globals import DEVICE
from utils import save_stats
from data_loading import load_data2


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

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=output_classes,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    model = smp.DeepLabV3(
        encoder_name="resnet101",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=output_classes,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    class exp_model(nn.Module):
        def __init__(self):
            super().__init__()
            self.m1 = smp.Unet(
                encoder_name="inceptionv4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet+background",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=4,  # model output channels (number of classes in your dataset)
            ).float().to(DEVICE)

            self.m2 = smp.Unet(
                encoder_name="inceptionv4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet+background",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=8,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=output_classes,  # model output channels (number of classes in your dataset)
            ).float().to(DEVICE)

        def forward(self, input):
            m1_out = self.m1(input)
            #print("M1:", m1_out.shape)
            #print("INPUT:", input.shape)
            concat = torch.cat((m1_out, input), dim=1)
            #print("CONCAT:", concat.shape)
            m2_out = self.m2(concat)
            #print("M2:",m2_out.shape)
            return m2_out

    model = smp.Unet(
        encoder_name="inceptionv4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet+background",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=output_classes,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    loss_func = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss("binary", from_logits=True)
    opt = Adam(model.parameters(), lr=init_lr)
    #opt = RMSprop(model.parameters(), lr=init_lr)

    def my_loss(pred, label):
        alpha = 0.5
        return alpha * dice_loss(pred, label) + (1-alpha) * loss_func(pred, label)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=step_size, gamma=gamma, verbose=True)

    train_loop_for_marker(args, model, my_loss, opt, trainLoader, testLoader, scheduler, loss_to_track=dice_loss)


def train_loop_for_marker(args, model, loss_func_to_opt, optimizer, train_loader, test_loader, scheduler, loss_to_track=None):
    weights_path = args["WEIGHTS_PATH"]
    num_epochs = args["NUM_EPOCHS"]
    stat_path = args["STAT_PATH"]

    best_model = model
    best_loss = math.inf
    best_epoch = 0

    history = {"train_loss": [], "test_loss": [], "train_loss_additional": [], "test_loss_additional": []}
    # loop over epochs
    print("[INFO] training the network...")
    start_time = time.time()

    avg_train_loss_se = 0
    avg_test_loss_se = 0

    avg_train_loss2_se = 0
    avg_test_loss2_se = 0

    for e in tqdm(range(num_epochs)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        total_train_loss = 0
        total_test_loss = 0

        total_train_loss2 = 0
        total_test_loss2 = 0

        # loop over the training set
        train_steps = None
        test_steps = None
        for (i, (image, marker, y)) in enumerate(train_loader):
            image_and_marker = torch.cat((image, marker), dim=1)
            image_and_marker = image_and_marker.to(DEVICE)
            (image, y) = (image.to(DEVICE), y.to(DEVICE))

            pred = model(image_and_marker)
            loss = loss_func_to_opt(pred, y.type(torch.float))

            with torch.no_grad():
                loss2 = loss_to_track(pred, y.type(torch.float))
                total_train_loss2 += loss2
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            total_train_loss += loss
            train_steps = i
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set

            for (i, (image, marker, y)) in enumerate(test_loader):
                image_and_marker = torch.cat((image, marker), dim=1)
                image_and_marker = image_and_marker.to(DEVICE)
                (image, y) = (image.to(DEVICE), y.to(DEVICE))
                # make the predictions and calculate the validation loss
                pred = model(image_and_marker)
                total_test_loss += loss_func_to_opt(pred, y.type(torch.float))

                loss2 = loss_to_track(pred, y.type(torch.float))
                total_test_loss2 += loss2

                test_steps = i
        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_test_loss = total_test_loss / test_steps

        avg_train_loss2 = total_train_loss2 / train_steps
        avg_test_loss2 = total_test_loss2 / test_steps

        avg_train_loss_se += avg_train_loss / 10
        avg_test_loss_se += avg_test_loss / 10

        avg_train_loss2_se += avg_train_loss2 / 10
        avg_test_loss2_se += avg_test_loss2 / 10

        if avg_test_loss < best_loss:
            print("New best model found!")
            best_loss = avg_test_loss
            best_model = model.state_dict()
            best_epoch = e
        # update our training history
        #history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        #history["test_loss"].append(avg_test_loss.cpu().detach().numpy())
        #history["train_loss_additional"].append(avg_train_loss2.cpu().detach().numpy())
        #history["test_loss_additional"].append(avg_test_loss2.cpu().detach().numpy())
        # print the model training and validation information
        if e % 5 == 0:
            torch.save(model.state_dict(), os.path.join(weights_path, f"test_E{e:03d}.pt"), )

        if e % 10 == 0 and e > 0:
            # One Super epoch done:
            history["train_loss"].append(avg_train_loss_se.cpu().detach().numpy())
            history["test_loss"].append(avg_test_loss_se.cpu().detach().numpy())
            history["train_loss_additional"].append(avg_train_loss2_se.cpu().detach().numpy())
            history["test_loss_additional"].append(avg_test_loss2_se.cpu().detach().numpy())

            print("[INFO] EPOCH: {}/{}".format(e, num_epochs))
            print("Train loss: {:.6f}, Test loss: {:.4f}".format(
                avg_train_loss_se, avg_test_loss_se))
            print("Train loss 2: {:.6f}, Test loss 2: {:.4f}".format(
                avg_train_loss2_se, avg_test_loss2_se))

            save_stats(history, stat_path)

            avg_train_loss_se = 0
            avg_test_loss_se = 0

            avg_train_loss2_se = 0
            avg_test_loss2_se = 0

        print(f"Best Epoch: {best_epoch}")
        scheduler.step()
    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        end_time - start_time))

    save_stats(history, stat_path)
    torch.save(best_model, os.path.join(weights_path, f"best_model_E_{best_epoch}.pt"))


def train_loop(args, model, loss_func, optimizer, train_loader, test_loader, scheduler):
    weights_path = args["WEIGHTS_PATH"]
    num_epochs = args["NUM_EPOCHS"]
    stat_path = args["STAT_PATH"]

    history = {"train_loss": [], "test_loss": []}
    # loop over epochs
    print("[INFO] training the network...")
    start_time = time.time()
    for e in tqdm(range(num_epochs)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        total_train_loss = 0
        total_test_loss = 0
        # loop over the training set
        train_steps = None
        test_steps = None
        for (i, (x, y)) in enumerate(train_loader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))

            pred = model(x)
            loss = loss_func(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far
            total_train_loss += loss
            train_steps = i
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (i, (x, y)) in enumerate(test_loader):
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))
                # make the predictions and calculate the validation loss
                pred = model(x)
                total_test_loss += loss_func(pred, y)
                test_steps = i
        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_test_loss = total_test_loss / test_steps
        # update our training history
        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        history["test_loss"].append(avg_test_loss.cpu().detach().numpy())
        # print the model training and validation information
        if e % 5 == 0:
            torch.save(model.state_dict(), os.path.join(weights_path, f"test_E{e:03d}.pt"), )
        print("[INFO] EPOCH: {}/{}".format(e + 1, num_epochs))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avg_train_loss, avg_test_loss))

        scheduler.step()
    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        end_time - start_time))

    save_stats(model.state_dict(), os.path.join(weights_path, "test_E45.pt"), history, stat_path)
