import os

import torch
from PIL import Image
from torch.utils import data

from data.utils import rle2bbox, rle_decode_string

import numpy as np


class FashionDataset(data.Dataset):
    def __init__(self, df, data_dir, augmentations=None):
        self._augmentations = augmentations
        self._data_dir = data_dir
        self._data = df['ImageId'].unique()
        self._df = df

    def __getitem__(self, idx):
        # load images ad masks
        image_id = self._data[idx]
        image_path = os.path.join(self._data_dir, f'{image_id}.jpg')
        height = self._df.loc[self._df['ImageId'] == '00000663ed1ff0c4e0132b9b9ac53f6e'].head()['Height'].values[0]
        width = self._df.loc[self._df['ImageId'] == '00000663ed1ff0c4e0132b9b9ac53f6e'].head()['Width'].values[0]
        image = Image.open(image_path).convert("RGB")

        # convert the PIL Image into a numpy array
        masks = []
        boxes = []
        labels = []
        for _, row in self._df.loc[self._df['ImageId'] == '00000663ed1ff0c4e0132b9b9ac53f6e'].iterrows():
            class_id = int(row['ClassId'])
            encoded_pixels = row['EncodedPixels']
            mask = rle_decode_string(encoded_pixels, height, width)
            mask = np.expand_dims(mask, axis=2)
            masks.append(mask)
            boxes.append(rle2bbox(encoded_pixels, (height, width)))
            labels.append(class_id)

        if self._augmentations is not None:
            image, target = self._augmentations(image=np.asarray(image), segmentation=masks)
        # {ValueError}Tried to convert an iterable of arrays to a list of SegmentationMapsOnImage. Expected each array to be of shape (H,W,#SegmapsPerImage), i.e. to be 3-dimensional, but got dimensions 2, 2, 2, 2, 2, 2, 2, 2, 2 instead (array shapes: (5214, 3676), (5214, 3676), (5214, 3676), (5214, 3676), (5214, 3676), (5214, 3676), (5214, 3676), (5214, 3676), (5214, 3676)).
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        return image, target

    def __len__(self):
        return len(self._data)


if __name__ == '__main__':
    import pandas as pd
    from config import DATA_TRAIN_CSV
    from config import DATA_TRAIN_FOLDER
    import imgaug.augmenters as iaa

    transforms = iaa.Sequential([
        iaa.Resize(256),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])

    df = pd.read_csv(DATA_TRAIN_CSV)

    ds = FashionDataset(df, DATA_TRAIN_FOLDER, transforms)
    print(ds.__getitem__(0))
