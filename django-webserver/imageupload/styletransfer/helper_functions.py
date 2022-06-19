from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn


def load_image(img_path, max_size=400, shape=None, for_vgg=True):
    """Laden und transformieren von Bildern und Sicherstellung dass das Bild <= 400 pixels in der x-y dimension."""

    image = Image.open(img_path).convert("RGB")

    # große Bilder beeinträchtigen die Ausführungszeit
    if max(image.size) > max_size:
        size = max_size

    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    compose = [
        transforms.Resize(size),
        transforms.ToTensor(),
    ]

    if for_vgg:
        compose.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
        in_transform = transforms.Compose(compose)

        return in_transform(image)[:3, :, :].unsqueeze(0)

    else:
        in_transform = transforms.Compose(compose)
        image = in_transform(image)[:3, :, :]
        image = image.numpy().transpose(1, 2, 0)

        return image


def im_convert(tensor):
    """Hilfsfunktion um ein Tensor Bild wieder darzustellen (un-normalizing,
    konvertieren des Tensors in ein NumPy Bild), entnommen aus (2)"""

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def vgg_ready(array):
    """Wrapper damit ein numpy rgb image array durch das vgg network propagiert werden kann"""
    in_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    image = in_transform(array)
    return image.unsqueeze(0)


# Methode um die Feature Maps einer spezifizierten Schicht auszugegeben
# basiert auf(2)
# ggf. ein paar Schichten rauslassen um den GPU Speicher nicht auszureizen


def get_features(image, model, selected_layers):
    """Run an image forward through a model and get the features for
    a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    # Convolutional Layer und ihr Name in der VGG Definition
    dict_layers_name_to_num = {
        "conv1_1": "0",
        "conv1_2": "2",
        "conv2_1": "5",
        "conv2_2": "7",
        "conv3_1": "10",
        "conv3_2": "12",
        "conv3_3": "14",
        "conv3_4": "16",
        "conv4_1": "19",
        "conv4_2": "21",
        "conv4_3": "23",
        "conv4_4": "25",
        "conv5_1": "28",
        "conv5_2": "30",
        "conv5_3": "32",
        "conv5_4": "34",
    }
    dict_layers_num_to_name = {v: k for k, v in dict_layers_name_to_num.items()}

    selected_layers_num = [dict_layers_name_to_num[layer] for layer in selected_layers]

    features = {}
    x = image

    # model._modules ist ein dictionary in dem jede Schicht des Models gelistet ist
    for name, layer in model._modules.items():
        x = layer(x)
        if name in selected_layers_num:
            features[dict_layers_num_to_name[name]] = x

    return features


# Funktion zur Berechnung der Gram Matrix aus (2) entnommen
def gram_matrix(tensor):
    """Calculate the Gram Matrix of a given tensor
    Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """

    # get the batch_size, depth, height, and width of the Tensor
    b, d, h, w = tensor.size()

    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(b * d, h * w)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram


def get_mbr_clothing_info(fashion_mask):
    """Get minimum bounding rectangle (mbr) of clothing
    Detecting rgb value of the background, mistakes could be avoided by comparing the rgb values of all corners"""
    x_first_pixel_fashion = None
    x_last_pixel_fashion = None
    y_first_pixel_fashion = None
    y_last_pixel_fashion = None

    for row in range(fashion_mask.shape[0]):
        rgb_values_row = fashion_mask[row, :]
        if (
            rgb_values_row.sum() >= 2.9
        ):  # white pixel has a sum of RGB values of 3, 2.9 is chosen because of some tolerance issues
            y_first_pixel_fashion = row
            break

    for row in range(fashion_mask.shape[0])[y_first_pixel_fashion:]:
        rgb_values_row = fashion_mask[row, :]
        if rgb_values_row.sum() <= 2.9:
            y_last_pixel_fashion = row - 1
            break

    for col in range(fashion_mask.shape[1]):
        rgb_values_col = fashion_mask[:, col]
        if rgb_values_col.sum() >= 2.9:
            x_first_pixel_fashion = col
            break

    for col in range(fashion_mask.shape[1])[x_first_pixel_fashion:]:
        rgb_values_col = fashion_mask[:, col]
        if rgb_values_col.sum() <= 2.9:
            x_last_pixel_fashion = col - 1
            break

    # TODO: incorporate some confirmation method for real world practice
    # eg. sum has to be over 3 iteration over the defined treshhold

    mbr_shape = (
        y_last_pixel_fashion - y_first_pixel_fashion,
        x_last_pixel_fashion - x_first_pixel_fashion,
        3,
    )

    mbr_location = np.s_[
        y_first_pixel_fashion:y_last_pixel_fashion,
        x_first_pixel_fashion:x_last_pixel_fashion,
        :,
    ]

    return (mbr_shape, mbr_location)