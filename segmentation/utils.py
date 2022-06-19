import json
import os

import torch
from matplotlib import pyplot as plt
from torchvision import transforms


def pixel_mask(image_tensor: torch.Tensor, pixel_values: torch.Tensor):
    pixel_values = pixel_values[:][None][None].transpose(0, 2)
    eq = torch.eq(input=image_tensor*255, other=pixel_values)
    eq = torch.ge(torch.sum(eq, dim=0), 3)

    result = eq.type(torch.uint8)
    return result


def create_marker_mask(binary_mask_tensor: torch.Tensor):
    marker_mask = torch.zeros(binary_mask_tensor.shape).squeeze()
    gauss = transforms.GaussianBlur(kernel_size=5, sigma=1)

    blurred = gauss(binary_mask_tensor.type(torch.float32))
    smaller_binary_mask = (blurred > 0.9).type(torch.uint8)

    nonzero = torch.nonzero(smaller_binary_mask.squeeze())
    num_pixels = nonzero.shape[0]

    if num_pixels > 0:
        random_number = int(torch.rand(size=[1]).item() * num_pixels)
        pixel = nonzero[random_number]
        marker_mask[pixel[0]][pixel[1]] = 255
        marker_mask = gauss(marker_mask.unsqueeze(dim=0).type(torch.float32))

    return marker_mask.type(torch.uint8)


def save_stats(losses, fig_path):
    plt.clf()
    plt.plot(losses["train_loss"], label='train_loss')
    plt.plot(losses["test_loss"], label='test_loss')
    plt.plot(losses["train_loss_additional"], label='train_loss_additional')
    plt.plot(losses["test_loss_additional"], label='test_loss_additional')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(os.path.join(fig_path, "losses.png"))
    plt.close()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
