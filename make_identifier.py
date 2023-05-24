import src
import sys
import argparse
from utils import create_classifier, create_embedding

config = src.load_config()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='store_true', help="Whether overwrite existing representations or not")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_embedding(**vars(opt))

    if config['CLASSIFIER']['use_model'] == "similarity":
        sys.exit()
    
    create_classifier()