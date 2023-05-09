import sys
import torch
import os
import pickle
import cv2 as cv
import numpy as np 
from tqdm import tqdm
from termcolor import colored
import src
from torchvision.transforms import Resize

if __name__ == "__main__":
    config = src.load_config()
    data_dir = config['RECOGNIZER']['data_dir']

    if not os.path.exists(data_dir):
        print(f"Can not find {data_dir}")
        sys.exit()

    detector, recognizer = src.loadDetectorRecognizer(keep_all=False)

    identity_lst = []
    representation_lst = []

    save_dir = os.path.join(data_dir,'representation.pkl')
    if os.path.exists(save_dir):
        print(f"Found Existing {save_dir}, remove it...")
        os.remove(save_dir)

    for identity in tqdm(next(os.walk(data_dir))[1], desc="Create representation", leave=True):
        if not identity.startswith("."):
            # Read all image in each identity folders
            for file in os.listdir(os.path.join(data_dir, identity)):
                img_file = os.path.join(data_dir, identity, file)
                image = cv.imread(img_file)
                if image is not None:
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    faces, _, _, _ = detector(image)

                    if faces is not None:
                        faces = Resize((112, 112), antialias=True)(faces)
                        if faces.dim() == 3:
                            faces = torch.unsqueeze(faces, dim=0)
                        faces = faces.detach().numpy().astype(np.float32)
                        out = recognizer.forward(faces)
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
    print(colored("Save face embedding to representation.pkl", 'green'))