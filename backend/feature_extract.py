from abc import abstractmethod

import tensorflow.compat.v1 as tf  # 设置为v1版本
import numpy as np
import os
import base64

tf.disable_v2_behavior()  # 禁用v2版本


class FeatureExtractor(object):
    def __init__(self, img_extensions=None):
        if img_extensions is None:
            img_extensions = ['.jpg', '.png']
        self.img_extensions = img_extensions

    @abstractmethod
    def GetFeature(self, img_path: str):
        pass

    def ReadImages(self, img_path: str) -> [np.numarray, np.numarray]:
        # TODO: Rewrite using tf.data.Dataset
        img_data, img_label = [], []

        def ParseImgs(file: str):
            return tf.gfile.FastGFile(os.path.join(img_path, file), 'rb').read()

        if os.path.isdir(img_path):
            for file in os.listdir(img_path):
                if '.' + file.split('.')[-1] in self.img_extensions:
                    img_data.append(ParseImgs(file))
                    img_label += [file[:-4]]  # doesn't matter since labels are useless for feature extraction
        elif os.path.isfile(img_path):
            if '.' + img_path.split('.')[-1] in self.img_extensions:
                img_data.append(ParseImgs(img_path))
                img_label += [img_path.split('/')[-1][:-4]]
        return np.array(img_data), np.array(img_label)


class InceptionExtractor(FeatureExtractor):
    def __init__(self, model_path='./Inception/tensorflow_inception_graph.pb',
                 img_extensions=['.jpg', '.png']):
        # Official Model from here: http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        super().__init__(img_extensions)
        self.model_path = model_path
        self.feature_size = [1, 1, 2048]
        self.feature_type = None  # TODO: DEBUG Inception to find out
        self.LoadInception()

    @staticmethod
    def bytes2ndarray(base64_str: str) -> np.ndarray:
        pass

    @staticmethod
    def ndarray2bytes(array: np.ndarray) -> str:
        pass

    def LoadInception(self):
        with tf.gfile.FastGFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def=graph_def, name='')

    def GetFeature(self, img_path: str):
        img_data, img_label = self.ReadImages(img_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tensor = sess.graph.get_tensor_by_name(
                'pool_3/_reshape:0')  # only use Inception as a backbone for feature extraction
            v_vector = sess.run(tensor, feed_dict={'DecodeJpeg/contents:0': img_data[0]})  # 1x1x2048
            v_vector = np.array(v_vector)
            return
