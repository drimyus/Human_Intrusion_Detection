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
        img = cv2.imread(os.path.join(neg_img_dir, fn))
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


def test(model_path, img_path):
    img = cv2.imread(img_path)
    feature = inu.get_feature(img=img)

    model = joblib.load(model_path)

    feature = feature.reshape(1, -1)
    probab = model.predict_proba(feature)

    max_ind = np.argmax(probab)

    # Rearrange by size
    sort_probab = np.sort(probab, axis=None)[::-1]

    if sort_probab[0] / sort_probab[1] > 0.7:
        predlbl = model.classes_[max_ind]
        predlbl = (predlbl == "pos")
    else:
        predlbl = None

    return predlbl, max_ind, probab


if __name__ == '__main__':
    # positive_dir = "../../data/train_data/pos"
    # negative_dir = "../../data/train_data/neg"
    # model_path = "./model/model.pkl"
    # train(pos_img_dir=positive_dir, neg_img_dir=negative_dir, model_path=model_path)
    #
    pos_test = "../../data/train_data/pos/3.jpg"
    neg_test = "../../data/train_data/neg/1.jpg"
    model_path = "./model/model.pkl"
    print test(model_path=model_path, img_path=neg_test)
    print test(model_path=model_path, img_path=pos_test)

