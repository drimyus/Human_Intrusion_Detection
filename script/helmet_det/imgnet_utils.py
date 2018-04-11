import tensorflow as tf
import os
import numpy as np
import sys
import tarfile
import cv2
from six.moves import urllib
from node_lookup import NodeLookup

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for disable the info log of tensorflow


class ImgNetUtils:
    def __init__(self):
        root = os.path.dirname(__file__)
        self.model_path = os.path.join(root, "model")
        self.data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

        self.sess, self.softmax_tensor = None, None

        self.maybe_download_and_extract()
        self.create_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')
        self.node_lookup = NodeLookup(model_dir=self.model_path)
        self.num_top_predictions = 5

        self.keyword = "helmet"

    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(os.path.join(
                self.model_path, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def maybe_download_and_extract(self):
        """Download and extract model tar file."""
        dest_directory = self.model_path
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self.data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                print('    \r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(self.data_url, filepath, _progress)

            statinfo = os.stat(filepath)
            print('    Successfully downloaded' + filename + str(statinfo.st_size) + 'bytes.\n')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def inference(self, img_path):
        # fix error classifier's "Graph is finalized and cannot be modified."
        tf.reset_default_graph()
        tf.Graph().as_default()
        if not tf.gfile.Exists(img_path):
            tf.logging.fatal('File does not exist %s', img_path)

        feature = self.get_feature(img=cv2.imread(img_path))
        result = self.detect(predictions=feature)
        print result

    def get_feature(self, img):
        if img is None:
            sys.exit(1)
        tf.reset_default_graph()
        tf.Graph().as_default()

        image_data = cv2.imencode('.jpg', img)[1].tostring()

        predictions = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        return predictions

    def detect(self, predictions):
        top_k = predictions.argsort()[-self.num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            if human_string.find(self.keyword) != -1:
                print('%s (score = %.5f)' % (human_string, score))
                cv2.imshow(self.keyword, cv2.imread(img_path))
                cv2.waitKey(1)
                return True
        return False


if __name__ == '__main__':
    inu = ImgNetUtils()
    dir = "../../data/train_data/pos"
    fns = os.listdir(dir)
    fns.sort()
    for fn in fns:
        print fn
        img_path = os.path.join(dir, fn)
        inu.inference(img_path=img_path)

        # img = cv2.imread(img_path)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
