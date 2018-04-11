import os
import sys
import numpy as np
from sklearn.externals import joblib


class ClimbDet:
    def __init__(self):
        root = os.path.dirname(__file__)
        self.model_path = os.path.join(root, "model/model.pkl")
        if os.path.isfile(self.model_path):
            try:
                # loading
                self.model = joblib.load(self.model_path)
            except Exception as ex:
                print(ex)
                sys.exit(0)
        else:
            sys.stderr.write("    No exist Model {}, so it should be trained\n".format(self.model_path))
            sys.exit(0)

    def climb_detect(self, feature):
        feature = feature.reshape(1, -1)

        # Get a prediction from the model including probability:
        probab = self.model.predict_proba(feature)

        max_ind = np.argmax(probab)

        # Rearrange by size
        sort_probab = np.sort(probab, axis=None)[::-1]

        if sort_probab[0] / sort_probab[1] > 0.7:
            predlbl = self.model.classes_[max_ind]
            predlbl = (predlbl == "pos")
        else:
            predlbl = None

        return predlbl, max_ind, probab
