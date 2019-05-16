from torch.utils import data
from random import sample
import cv2
from commons.config import FGVC6_TRAIN_IMAGES_FOLDER_PATH
from utils.image_masker import ImageMasker

class FashionParsingDataSet(data.Dataset):
    def __init__(self, df, width, heigth):
        self._data_set = df.values
        self._width = width
        self._height = heigth

    def shuffle(self):
        self._data_set = sample(self._data_set, len(self._data_set))

    def __len__(self):
        return len(self._data_set)

    def __getitem__(self, index):
        sample = self._data_set[index]
        img_name = sample[0]
        img = cv2.imread(FGVC6_TRAIN_IMAGES_FOLDER_PATH + img_name)
        img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)
        segment_df = (df.loc[index:index + ind_num - 1, :]).reset_index(drop=True)
        index += ind_num
        if segment_df["ImageId"].nunique() != 1:
            raise Exception("Index Range Error")
        seg_img = make_mask_img(segment_df)

        # HWC -> CHW
        img = img.transpose((2, 0, 1))
        # seg_img = seg_img.transpose((2, 0, 1))

        trn_images.append(img)
        seg_images.append(seg_img)
        if ((i + 1) % batch_size == 0):
            yield np.array(trn_images, dtype=np.float32) / 255, np.array(seg_images, dtype=np.int32)
            trn_images = []
            seg_images = []
