import timeit

import torch
from torchvision import transforms

DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def pixel_mask_for_batch(image_tensor: torch.Tensor, pixel_values: torch.Tensor):
    pixel_values = pixel_values[:][None][None][None].transpose(1, 3)
    eq = torch.eq(input=image_tensor, other=pixel_values)
    eq = torch.ge(torch.sum(eq, dim=1), 3)

    return eq.type(torch.uint8)


def pixel_mask(image_tensor: torch.Tensor, pixel_values: torch.Tensor):
    pixel_values = pixel_values[:][None][None].transpose(0, 2)
    eq = torch.eq(input=image_tensor*255, other=pixel_values)
    eq = torch.ge(torch.sum(eq, dim=0), 3)

    result = eq.type(torch.uint8)
    return result


def create_marker_masks_for_batch(binary_mask_tensor: torch.Tensor, device=DEVICE):
    batch_size = binary_mask_tensor.shape[0]
    image_res = binary_mask_tensor.shape[-2], binary_mask_tensor.shape[-1]

    markers = torch.zeros((batch_size, 1, image_res[0], image_res[1]), dtype=torch.uint8).to(device)

    for i, mask in enumerate(binary_mask_tensor):
        single_marker = create_marker_mask(mask)
        markers[i] = single_marker

    return markers


def create_marker_mask(binary_mask_tensor: torch.Tensor):
    marker_mask = torch.zeros(binary_mask_tensor.shape).squeeze()
    gauss = transforms.GaussianBlur(kernel_size=5, sigma=1)

    blurred = gauss(binary_mask_tensor.type(torch.float32))
    smaller_binary_mask = ((blurred > 0.9)).type(torch.uint8)
    nonzero = torch.nonzero(smaller_binary_mask.squeeze())
    num_pixels = nonzero.shape[0]
    if num_pixels > 0:
        random_number = int(torch.rand(size=[1]).item() * num_pixels)
        pixel = nonzero[random_number]
        marker_mask[pixel[0]][pixel[1]] = 255

        #marker_mask[pixel[0]-1][pixel[1]] = 1
        #marker_mask[pixel[0]][pixel[1]-1] = 1
        #marker_mask[pixel[0]-1][pixel[1]-1] = 1
        #marker_mask[pixel[0]+1][pixel[1]-1] = 1
        #marker_mask[pixel[0]-1][pixel[1]+1] = 1

        #marker_mask[pixel[0]+1][pixel[1]] = 1
        #marker_mask[pixel[0]+1][pixel[1]+1] = 1
        #marker_mask[pixel[0]][pixel[1] + 1] = 1

    return marker_mask.type(torch.uint8)

def extract_colors(single_mask: torch.Tensor):
    mask_flat = torch.flatten(single_mask, start_dim=1, end_dim=-1)
    colors = torch.unique(mask_flat, dim=1).T * 255
    return colors


