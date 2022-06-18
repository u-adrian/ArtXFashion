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
            encoder_name="inceptionv4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet+background",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        ).float().to(self.device)
        
        self.model.load_state_dict(torch.load(self.model_path))

    def __create_marker_mask(self, x, y):
        marker_mask = torch.zeros((self.model_image_height,self.model_image_width)).squeeze()
        gauss = transforms.GaussianBlur(kernel_size=5, sigma=1)

        marker_mask[x][y] = 255
        marker_mask = gauss(marker_mask.unsqueeze(dim=0).type(torch.float32))

        return marker_mask.type(torch.uint8).unsqueeze(dim=0)

    def do_segmentation(self, image: torch.Tensor, x, y):
        with torch.no_grad:
            # expected image shape: [1,3,512,256]
            input_width = image.shape[3]
            input_height = image.shape[2]

            assert (input_height == 512 and input_width == 256)
            assert (256 > x > 0)
            assert (512 > y > 0)

            image_transforms = transforms.Compose([transforms.Resize(size=(self.model_image_height,self.model_image_width))])

            x_rescaled = x
            y_rescaled = y

            marker_mask_255_max = self.__create_marker_mask(x_rescaled, y_rescaled)

            image_and_marker = torch.cat((image, marker_mask_255_max), dim=1)

            self.model.eval()
            segmentation = self.model(image_and_marker)

            return segmentation
