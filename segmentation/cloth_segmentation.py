import argparse

from train import train
from utils import load_config


def main(parser: argparse.ArgumentParser):
    parser.add_argument("--config_path", type=str, default="C:/Dev/Smart_Data/ArtXFashion/segmentation/config.json")
    config_path = parser.parse_args()
    args = load_config(config_path=config_path.config_path)

    train(args)
    #test_model2(args, 100)


if __name__ == "__main__":
    main(argparse.ArgumentParser())
