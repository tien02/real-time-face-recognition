import sys
import torch
import os
import src
import pickle
import argparse
import cv2 as cv
import numpy as np 

from tqdm import tqdm
from termcolor import colored
from utils import load_data, evaluate
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

config = src.load_config()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='store_true', help="Whether overwrite existing representations or not")
    opt = parser.parse_args()
    return opt


def create_embedding(force:bool):
    print(colored(f"Create Embedding...", "blue"))

    data_dir = config['RECOGNIZER']['data_dir']

    if not os.path.exists(data_dir):
        print(f"Can not find {data_dir}")
        sys.exit()

    _, recognizer = src.loadDetectorRecognizer(keep_all=False)


    for identity in tqdm(next(os.walk(data_dir))[1], desc="Create representation", leave=True):
        save_dir = os.path.join(data_dir, identity, f'{identity}_representation.pkl')

        if os.path.exists(save_dir):
            if force:
                print(colored(f"Found Existing {save_dir}, remove it...", "green"))
                os.remove(save_dir)
            else:
                print(colored(f"Found Existing {save_dir}, go next...", "green"))
                continue

        identity_lst = []
        representation_lst = []

        # Read all image in each identity folders
        for file in os.listdir(os.path.join(data_dir, identity)):
            img_file = os.path.join(data_dir, identity, file)
            image = cv.imread(img_file)
            if image is not None:
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                if image.shape[0] != config['RECOGNIZER']['crop_img_size']:
                    image = cv.resize(image, (config['RECOGNIZER']['crop_img_size'], config['RECOGNIZER']['crop_img_size']))
                image = np.transpose(image, (2,0,1)).astype(np.float32)

                out = recognizer.forward(np.expand_dims(image, axis=0))
                # out = recognizer(faces.unsqueeze(0)).detach().numpy()
                identity_lst.append(identity)
                representation_lst.append(out)
            else:
                print(f"Error reading file {img_file}")

        representation_dict = {
            "name": identity_lst,
            "embedding": representation_lst,
        }

        with open(save_dir, 'wb') as f:
            pickle.dump(representation_dict, f)
        print(colored(f"Save face embedding to {save_dir}", 'green'))
    print("DONE")


if __name__ == "__main__":
    opt = parse_opt()
    create_embedding(**vars(opt))

    if config['CLASSIFIER']['use_model'] == "similarity":
        sys.exit()

    print(colored(f"Create Classifier...", "blue"))
    data = load_data(config['RECOGNIZER']['data_dir'])

    X = np.array(data['embedding']).squeeze(axis=1)
    y = np.array(data['name'])

    # Split train - test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, shuffle=True)

    # Train - Test
    model = MLPClassifier(max_iter=5000, random_state=40)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    evaluate(y_test, pred, model.classes_)

    # Export to onnx
    if not os.path.exists('./resource'):
        os.makedirs('./resource')

    initial_type = [('input', FloatTensorType([1, 512]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open("resource/classifier.onnx", "wb") as f:
        f.write(onx.SerializeToString())