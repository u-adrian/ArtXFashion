import argparse

import torch

from test_model import test_model2
from train import train
from utils import load_config

def do_segmentation(image: torch.Tensor, x, y):
    ### expected image shape: [1,3,width,height]
    width = image.shape[2]
    height = image.shape[3]

    # load model
    model_path = ""

    raise NotImplementedError

def main(parser: argparse.ArgumentParser):
    parser.add_argument("--config_path", type=str, default="C:/Dev/Smart_Data/ArtXFashion/segmentation/config.json")
    config_path = parser.parse_args()
    args = load_config(config_path=config_path.config_path)

    #train(args)
    test_model2(args, 1000)


if __name__ == "__main__":
    main(argparse.ArgumentParser())
