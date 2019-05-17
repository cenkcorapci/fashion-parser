import cv2
import numpy as np
from utils.vectorizer import OneHotVectorization


class ImageMasker:
    def __init__(self, category_num):
        self._category_num = category_num
        self._vectorizer = OneHotVectorization(category_num)

    def make_mask_img(self, segment_df, height, width):
        seg_width = segment_df.at[0, "Width"]
        seg_height = segment_df.at[0, "Height"]

        seg_img = np.full(seg_width * seg_height, self._category_num - 1, dtype=np.int32)
        for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
            pixel_list = list(map(int, encoded_pixels.split(" ")))
            for i in range(0, len(pixel_list), 2):
                start_index = pixel_list[i] - 1
                index_len = pixel_list[i + 1] - 1
                seg_img[start_index:start_index + index_len] = int(class_id.split("_")[0])
        seg_img = seg_img.reshape((seg_height, seg_width), order='F')
        seg_img = cv2.resize(seg_img, (width, height), interpolation=cv2.INTER_NEAREST)
        seg_img_onehot = np.zeros((height, width, self._category_num), dtype=np.int32)

        for ind in range(height):
            for col in range(width):
                seg_img_onehot[ind, col] = self._vectorizer.make_one_hot_vec(seg_img[ind, col])
        seg_img_onehot = np.swapaxes(seg_img_onehot, 0, 2)
        return seg_img_onehot
