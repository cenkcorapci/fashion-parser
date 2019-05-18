from itertools import groupby

import cv2
import numpy as np  # linear algebra

from commons.config import FGVC6_TEST_IMAGES_FOLDER_PATH


class FGVC6SubmissionSetGenerator:
    def __init__(self, df, width, height, num_categories):
        self._df = df
        self._width = width
        self._height = height
        self._num_categories = num_categories

    def __len__(self):
        return len(self._df)

    def test_generator(self):
        img_names = self._df["ImageId"].values
        for img_name in img_names:
            img = cv2.imread(FGVC6_TEST_IMAGES_FOLDER_PATH + img_name)
            img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)
            # HWC -> CHW
            img = img.transpose((2, 0, 1))
            yield img_name, np.asarray([img], dtype=np.float32) / 255

    def run_length(self, label_vec):
        encode_list = self._encode(label_vec)
        index = 1
        class_dict = {}
        for i in encode_list:
            if i[1] != self._num_categories - 1:
                if i[1] not in class_dict.keys():
                    class_dict[i[1]] = []
                class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]
            index += i[0]
        return class_dict

    def _encode(self, input_string):
        return [(len(list(g)), k) for k, g in groupby(input_string)]
