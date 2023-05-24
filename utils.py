import os
import sys
import src
import glob
import pickle
import numpy as np
import cv2 as cv
import numpy as np 

from tqdm import tqdm
from termcolor import colored
from skl2onnx import convert_sklearn
from sklearn.neural_network import MLPClassifier
from skl2onnx.common.data_types import FloatTensorType

config = src.load_config()

class GlobalClassifier:
    classifier = None
    input_name = None
    
    @staticmethod
    def GetClassifier():
        if (GlobalClassifier.classifier == None) or (GlobalClassifier.input_name == None):
            create_classifier()
            GlobalClassifier.classifier, GlobalClassifier.input_name = src.loadClassifier()
        return GlobalClassifier.classifier, GlobalClassifier.input_name

    @staticmethod
    def UpdateClassifier():
        create_classifier()
        GlobalClassifier.classifier, GlobalClassifier.input_name = src.loadClassifier()
  


def create_embedding(force:bool = False):
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


def create_classifier():
    # Create a classifier
    print(colored(f"Create Classifier...", "blue"))
    data = load_data(config['RECOGNIZER']['data_dir'])

    X = np.array(data['embedding']).squeeze(axis=1)
    y = np.array(data['name'])

    # Train - Test
    model = MLPClassifier(max_iter=5000, random_state=40)

    model.fit(X,y)

    # Export to onnx
    if not os.path.exists('./resource'):
        os.makedirs('./resource')

    initial_type = [('input', FloatTensorType([1, 512]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open("resource/classifier.onnx", "wb") as f:
        f.write(onx.SerializeToString())


def load_data(data_dir:str):
    final_data = {'name': [], "embedding": []}
    pkl_lst = glob.glob(f"{data_dir}/*/*.pkl")

    assert len(pkl_lst) > 0, 'No user in database'
    assert len(pkl_lst) == len(next(os.walk(data_dir))[1]), "Expect each subdirectory have .pkl reprensentation"

    for pkl_file in pkl_lst:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        final_data['name'] += data['name']
        final_data['embedding'] += data['embedding']
    
    return final_data


def cosine_similarity(query, values):
    sim = np.dot(query, values.T) / np.multiply(np.linalg.norm(query), np.linalg.norm(values, axis=1))
    return sim