import os
import sys

import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

# from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from termcolor import colored
from src.loader import load_config

def evaluate(y_true, y_pred, labels):
    print(colored("Classification Report"))
    print(metrics.classification_report(y_true, y_pred, zero_division=0, labels=labels))
    print()
    print(colored("Save model to classifier.onnx", "green"))

    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    config = load_config()
    
    representation_path = f"./{config['RECOGNIZER']['data_dir']}/representation.pkl"
    if not os.path.exists(representation_path):
        print(f"Can not find {representation_path}, make sure to run create_representation.py first")
        sys.exit()

    # Get training data
    with open(representation_path, 'rb') as f:
        data = pickle.load(f)

    X = np.array(data['embedding']).squeeze(axis=1)
    y = np.array(data['name'])

    # Split train - test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, shuffle=True)

    # Train - Test
    model = MLPClassifier(max_iter=5000, random_state=40)
    # model = LogisticRegression()
    # model = KNeighborsClassifier()
    # model = GaussianNB()
    # model = RandomForestClassifier()
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