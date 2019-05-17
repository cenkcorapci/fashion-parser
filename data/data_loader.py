import json

import numpy as np
import pandas as pd
from skimage.transform import resize

from commons.config import *


class DataLoader:
    def __init__(self):
        with open(FGVC6_LABEL_DESCRIPTIONS_PATH) as json_file:
            data = json.load(json_file)
            logging.info('Loading label descriptions from: \n {0}'.format(data['info']))

            self._category_definitions = pd.DataFrame(data['categories'])
            self._train_data = pd.read_csv(FGVC6_TRAIN_CSV_PATH)

    def get_category_name(self, category_id):
        return self._category_definitions.loc[self._category_definitions.id == category_id].name.values[0]

    def get_num_of_classes(self):
        return len(self._category_definitions)

    def get_training_data_set(self):
        return self._train_data

    def get_masks(self, image_id, resized_shape=(128, 128)):
        masks = dict()
        temp = self._train_data[self._train_data.ImageId == image_id]
        for i in range(temp.shape[0]):
            width = temp.iloc[i].Width
            height = temp.iloc[i].Height
            class_id = temp.iloc[i].ClassId.split()[0]
            mask_encoded = temp.iloc[i].EncodedPixels.split()
            mask = [0] * (width * height)
            for j in range(0, len(mask_encoded), 2):
                mask[int(mask_encoded[j]): int(mask_encoded[j]) + int(mask_encoded[j + 1])] = [1] * int(
                    mask_encoded[j + 1])
            mask = np.fliplr(np.flip(np.rot90(np.array(mask).reshape((width, height)))))
            mask = resize(mask, resized_shape, anti_aliasing=True)
            masks[int(class_id)] = mask
        masks_classes = []
        for i in range(46):
            if i in masks:
                masks_classes.append(masks[i])
            else:
                masks_classes.append(np.zeros(resized_shape))
        masks_classes = np.array(masks_classes)
        return masks_classes


if __name__ == '__main__':
    loader = DataLoader()
    print(loader.get_category_name(0))
    print(loader.get_training_data_set().head())
