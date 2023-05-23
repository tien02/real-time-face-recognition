import os
import glob
import pickle
import numpy as np
from sklearn import metrics
from termcolor import colored
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred, labels):
    print(colored("Classification Report"))
    print(metrics.classification_report(y_true, y_pred, zero_division=0, labels=labels))
    print()
    print(colored("Save model to classifier.onnx", "green"))

    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


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