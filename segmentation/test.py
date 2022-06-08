import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torchvision import transforms

from segmentation.data_loading import load_data2, load_data
from segmentation.globals import DEVICE


def test_model2(args, epoch):
    weights_path = args["WEIGHTS_PATH"]
    output_classes = args["OUTPUT_CLASSES"]
    test_split = args["TEST_SPLIT"]
    input_image_height = args["INPUT_IMAGE_HEIGHT"]
    input_image_width = args["INPUT_IMAGE_WIDTH"]
    batch_size = args["BATCH_SIZE"]
    image_dataset_path = args["IMAGE_DATASET_PATH"]
    mask_dataset_path = args["MASK_DATASET_PATH"]
    meta_data_path = "C:/Dev/Smart_Data/Clothing_Segmentation/color_lists_reduced"

    # load model
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

    to_PIL = transforms.ToPILImage()

    pred_mask_path = "C:/Dev/Smart_Data/pred_mask"
    masks = torch.empty(0).to(DEVICE)
    #checkpoint = torch.load(f"{weights_path}/test_E{epoch:03d}.pt")
    checkpoint = torch.load(f"{weights_path}/best_model_E_39.pt")
    model.load_state_dict(checkpoint)

    # load data
    train_loader, test_loader = load_data2(test_split, input_image_height=input_image_height,
                                           input_image_width=input_image_width,
                                           batch_size=batch_size, image_dataset_path=image_dataset_path,
                                           mask_dataset_path=mask_dataset_path, meta_data_path=meta_data_path)
    # create masks using model
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (b, (image, marker, label)) in enumerate(test_loader):
            image_and_marker = torch.cat((image, marker), dim=1)
            image_and_marker = image_and_marker.to(DEVICE)
            pred = torch.sigmoid(model(image_and_marker)).cpu()
            for i in range(image_and_marker.shape[0]):
                folder = f"img_{i:04d}"

                image_folder = os.path.join(pred_mask_path, folder)
                Path(image_folder).mkdir(parents=True, exist_ok=True)

                to_PIL(image[i]).save(os.path.join(image_folder, "ref_image.png"))
                to_PIL(marker[i]).save(os.path.join(image_folder, "marker.png"))
                print(pred.shape)
                pred_i = pred[i]
                #to_PIL(pred_i).show()
                for channel in range(pred_i.shape[0]):
                    img = pred_i[channel]
                    img = (img > 0.5)*255
                    img_int = img.type(torch.uint8)
                    img_2 = to_PIL(img_int)

                    name = f"img_channel_{channel:03d}.png"

                    img_2.save(os.path.join(image_folder, name))

                    to_PIL(label[i][channel]*255).save(os.path.join(image_folder, f"ref_mask_{channel:03d}.png"))

            if b > 1:
                break


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
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=output_classes,  # model output channels (number of classes in your dataset)
    ).float().to(DEVICE)

    model = smp.Unet(
        encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
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
                #to_PIL(pred_i).show()
                for channel in range(pred_i.shape[0]):
                    img = pred_i[channel]
                    img = (img > 0.5)*255
                    img_int = img.type(torch.uint8)
                    img_2 = to_PIL(img_int)

                    name = f"img_channel_{channel:03d}.png"

                    img_2.save(os.path.join(image_folder, name))

                    to_PIL(y[i][channel]).save(os.path.join(image_folder, f"ref_mask_{channel:03d}.png"))

            if b > 7:
                break