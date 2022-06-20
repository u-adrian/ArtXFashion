import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
#import numpy as np
# Import the required libraries
import torch
import cv2
import torchvision.transforms as transforms
import os

class SegmentationModel:
    def __init__(self, device=None, model_path="./segmentation/weights_E1000.pt"):
        #model_path = 
        self.model_image_height = 512
        self.model_image_width = 256

        #self.device = device
        self.device = torch.device("cpu")
        #self.model_path = f'{os.getcwd()}/weights_E1000.pt'
        #print(os.abspath())
        self.model_path = '/code/imageupload/segmentation/weights_E1000.pt'


        self.model = smp.Unet(
            encoder_name="inceptionv4",
            encoder_weights=None,
            in_channels=4,
            classes=1,
        ).float().to(self.device)

        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))


    def do_image_segmentation(self, image_path, x,y):
        resize_x = 256
        resize_y = 512

        #crappy function since tensor needs to be set
        #in_channels 4?
        #
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape


        x, y = int(x), int(y)

        x = resize_x* x/width
        y = resize_y* y/height

        x, y = int(x), int(y)
        # Define a transform to convert the image to tensor
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize([resize_y, resize_x])])

        # Convert the image to PyTorch tensor
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        # Print the converted image tensor
        result =  self.do_segmentation(tensor, x, y)

        to_pil = transforms.Compose(
        [transforms.Resize([height, width]),
        transforms.ToPILImage()]) 

        return to_pil(result)

    def __create_marker_mask(self, x, y):
        marker_mask = torch.zeros((self.model_image_height,self.model_image_width)).squeeze()
        gauss = transforms.GaussianBlur(kernel_size=5, sigma=1)

        marker_mask[y][x] = 255
        marker_mask = gauss(marker_mask.unsqueeze(dim=0).type(torch.float32))

        return marker_mask.type(torch.uint8).unsqueeze(dim=0)

    def do_segmentation(self, image: torch.Tensor, x, y):
        with torch.no_grad():
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
            #segmentation = ((segmentation > 0.5)*255)
            segmentation[segmentation >= .5] = 1
            segmentation[segmentation < .5] = 0

            return segmentation.squeeze(0)
