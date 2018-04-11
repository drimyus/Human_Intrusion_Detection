import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import cv2

from script.helmet_det.imgnet_utils import ImgNetUtils
inu = ImgNetUtils()


def train(pos_img_dir, neg_img_dir, model_path):
    data = []  # initialize the data matrix and labels list
    labels = []

    sys.stdout.write("Scanning the dataset.\n")
    for fn in os.listdir(neg_img_dir):
        if not os.path.splitext(fn)[1] == ".jpg":
            continue
        img = cv2.imread(os.path.join(pos_img_dir, fn))
        if img is None:
            continue
        feature = inu.get_feature(img=img)
        data.append(feature)
        labels.append("neg")

    for fn in os.listdir(pos_img_dir):
        if not os.path.splitext(fn)[1] == ".jpg":
            continue
        img = cv2.imread(os.path.join(pos_img_dir, fn))
        if img is None:
            continue
        feature = inu.get_feature(img=img)
        data.append(np.asanyarray(feature))
        labels.append("pos")

    """-----------------------------------------------------------------------------------------"""
    sys.stdout.write("Training the model...\n")
    # Configure the model : linear SVM model with probability capabilities
    model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1,
                decision_function_shape='ovr', random_state=None)
    # model = SVC(probability=True)

    # Train the model
    model.fit(data, labels)

    joblib.dump(model, model_path)
    """-----------------------------------------------------------------------------------------"""
    sys.stdout.write("Finished the training.\n")


if __name__ == '__main__':
    positive_dir = "../../data/train_data/pos"
    negative_dir = "../../data/train_data/neg"
    model_path = "./model/model.pkl"
    train(pos_img_dir=positive_dir, neg_img_dir=negative_dir, model_path=model_path)
