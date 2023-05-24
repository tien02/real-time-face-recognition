import os
import sys
import yaml
import gdown
import onnxruntime as rt

from termcolor import colored
from insightface.model_zoo import get_model
from facenet_pytorch import MTCNN

def load_config() -> dict:
    with open("./config/config.yaml") as f:
        config = yaml.safe_load(f)
    return config


config = load_config()


def loadDetectorRecognizer(keep_all:bool = True):
    # Init Detector
    print(colored("Detector initialize", 'blue'), end=' - ')
    detector = MTCNN(keep_all=keep_all)
    print(colored("Done", 'green'))

    # Init Recognizer
    print(colored('Recognizer initialize', 'blue'), end=" - ")
    ckpt_path = config['RECOGNIZER']['ckpt_path']
    if not os.path.exists(ckpt_path):
        print(f"Can't find {ckpt_path}, Downloading...")

        file_name = os.path.basename(ckpt_path).split('.')[0]
        download_ckpt(file_name)

    recognizer = get_model(ckpt_path,providers=['CPUExecutionProvider'])
    print(colored("Done", 'green'))

    return detector, recognizer


def loadClassifier():
    ckpt_path = "resource/classifier.onnx"
    print(colored('Classifier initialize', 'blue'), end=" - ")

    if not os.path.exists(ckpt_path):
        print(f"Can't find {ckpt_path}")
        return

    classifier = rt.InferenceSession(ckpt_path, providers=["CPUExecutionProvider"])
    input_name = classifier.get_inputs()[0].name
    # label_name = classifier.get_outputs()[0].name
    print(colored("Done", 'green'))
    return classifier, input_name


def download_ckpt(file_name):
    if file_name == "R100_MS1MV2":
        drive_path = 'https://drive.google.com/file/d/1772DTho9EG047KNUIv2lop2e7EobiCFn/view'
    elif file_name == "R100_MS1MV3":
        drive_path = 'https://drive.google.com/file/d/1fZOfvfnavFYjzfFoKTh5j1YDcS8KCnio/view'
    elif file_name == "R100_Glint":
        drive_path = 'https://drive.google.com/file/d/1Gh8C-bwl2B90RDrvKJkXafvZC3q4_H_z/view'
    elif file_name == "R50_MS1MV3":
        drive_path = "https://drive.google.com/file/d/1FPldzmZ6jHfaC-R-jLkxvQRP-cLgxjCT/view"
    elif file_name == "R50_Glint":
        drive_path = "https://drive.google.com/file/d/1MpRhM76OQ6cTzpr2ZSpHp2_CP19Er4PI/view"
    else:
        print(f"Don't support {file_name}")
        return

    if not os.path.exists('resource'):
        os.makedirs('resource')

    save_path = os.path.join("resource", f"{file_name}.onnx") 
    print(colored(f"Download to {save_path}...", "green"))
    url = 'https://drive.google.com/uc?/export=download&id=' + drive_path.split('/')[-2]
    gdown.download(url, save_path, quiet=False)