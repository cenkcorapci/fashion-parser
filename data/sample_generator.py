from random import sample

import cv2
import numpy as np
from torch.utils import data

from commons.config import FGVC6_TRAIN_IMAGES_FOLDER_PATH
from utils.image_masker import ImageMasker


class FashionParsingDataSet(data.Dataset):
    def __init__(self, df, category_num, width, height):
        self._images = df.ImageId.values
        self._data_set = df
        self._width = width
        self._height = height
        self._masker = ImageMasker(category_num)

    def shuffle(self):
        self._images = sample(self._images, len(self._images))

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        img_name = self._images[index]
        segment_df = self._data_set.loc[self._data_set.ImageId == img_name].reset_index(drop=True)

        img = cv2.imread(FGVC6_TRAIN_IMAGES_FOLDER_PATH + img_name)
        img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)

        if segment_df["ImageId"].nunique() != 1:
            raise Exception("Index Range Error")
        seg_img = self._masker.make_mask_img(segment_df, self._width, self._height)

        # Height Width Color -> Color Height Width
        img = img.transpose((2, 0, 1))

        return np.array(img, dtype=np.float32) / 255, np.array(seg_img, dtype=np.int32)
