import torch
import segmentation_models_pytorch as smp
from torchvision import transforms


class SegmentationModel:
    def __init__(self, device, model_path=""):
        self.model_image_height = 512
        self.model_image_width = 256

        self.device = device
        self.model_path = model_path

        self.model = smp.Unet(
            encoder_name="inceptionv4",
            in_channels=4,
            classes=1,
        ).float().to(self.device)

        self.model.load_state_dict(torch.load(self.model_path))

    def __create_marker_mask(self, x, y):
        marker_mask = torch.zeros((self.model_image_height,self.model_image_width)).squeeze()
        gauss = transforms.GaussianBlur(kernel_size=5, sigma=1)

        marker_mask[y][x] = 255
        marker_mask = gauss(marker_mask.unsqueeze(dim=0).type(torch.float32))

        return marker_mask.type(torch.uint8).unsqueeze(dim=0)

    def do_segmentation(self, image: torch.Tensor, x, y):
        with torch.no_grad:
            # expected image shape: [1,3,512,256]
            input_width = image.shape[3]
            input_height = image.shape[2]

            assert (input_height == self.model_image_height and input_width == self.model_image_width)
            assert (self.model_image_width > x > 0)
            assert (self.model_image_height > y > 0)

            marker_mask_255_max = self.__create_marker_mask(x, y)

            image_and_marker = torch.cat((image, marker_mask_255_max), dim=1)

            self.model.eval()
            segmentation = self.model(image_and_marker)
            segmentation = torch.sigmoid(segmentation)
            segmentation = ((segmentation > 0.5)*255).cpu()

            return segmentation
