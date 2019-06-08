import cv2
import numpy as np

from commons.config import IMAGE_SIZE, FGVC6_TRAIN_IMAGES_FOLDER_PATH
from data.data_loader import DataLoader
from mrcnn import utils
from utils.image_utils import resize_image


class FashionDataset(utils.Dataset):

    def __init__(self, df, label_names):
        super().__init__(self)
        self._label_names = label_names

        # Add classes
        for index, name in enumerate(self._label_names):
            self.add_class("fashion", index + 1, name)

        # Add images
        for _, row in df.iterrows():
            self.add_image("fashion",
                           image_id=row.name,
                           path='{0}{1}'.format(FGVC6_TRAIN_IMAGES_FOLDER_PATH, row.name),
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'],
                           height=row['Height'], width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [self._label_names[int(x)] for x in info['labels']]

    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []

        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height'] * info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]

            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            mask[:, :, m] = sub_mask
            labels.append(int(label) + 1)

        return mask, np.array(labels)


if __name__ == '__main__':

    import random
    from mrcnn import visualize

    loader = DataLoader()
    dataset = FashionDataset(loader.image_df.sample(1000), loader.label_names)
    dataset.prepare()

    for i in range(6):
        image_id = random.choice(dataset.image_ids)
        print(dataset.image_reference(image_id))

        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)
