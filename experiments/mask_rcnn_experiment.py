# -*- coding: utf-8 -*-
from __future__ import print_function, division

import cv2
import pandas as pd
from imgaug import augmenters as iaa
from keras.callbacks import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import mrcnn.model as modellib
from commons.config import *
from commons.fashion_config import FashionConfig, InferenceConfig
from data.data_loader import DataLoader
from data.fashion_data_set import FashionDataset
from mrcnn import visualize
from utils.image_utils import resize_image, refine_masks, to_rle


class MaskRCNNExperiment:
    def __init__(self,
                 val_split=0.1,
                 nb_epochs=3,
                 learning_rate=0.01):

        self._in_inference_mode = False
        self._nb_epochs = nb_epochs
        self._lr = learning_rate
        self._model_name = 'mask_r_cnn_fashion_resnet_101'

        logging.info("Getting data set")
        loader = DataLoader()
        df = loader.image_df

        logging.info("Splitting {0} samples for validation".format(float(len(df)) * val_split))

        self._class_names = loader.label_names
        self._train_data_set, self._val_data_set = train_test_split(df, random_state=RANDOM_STATE, test_size=val_split)
        train_size = len(self._train_data_set)
        val_size = len(self._val_data_set)

        self._train_data_set = FashionDataset(self._train_data_set, self._class_names)
        self._val_data_set = FashionDataset(self._val_data_set, self._class_names)

        self._train_data_set.prepare()
        self._val_data_set.prepare()
        self._sample_df = pd.read_csv(FGVC6_SAMPLE_SUBMISSION_CSV_PATH)

        self._augmentation = iaa.Sequential([
            iaa.Fliplr(.5),  # horizontal flip
            iaa.Flipud(.5)  # vertical flip
        ])
        self._model_config = FashionConfig()
        self._model_config.STEPS_PER_EPOCH = train_size / self._model_config.IMAGES_PER_GPU
        self._model_config.VALIDATION_STEPS = val_size / self._model_config.IMAGES_PER_GPU
        # load model
        self._model = modellib.MaskRCNN(mode='training',
                                        config=self._model_config,
                                        model_dir=DL_MODELS_PATH)
        if PRE_TRAINED_FASHION_WEIGHTS is not None:
            self._model.load_weights(PRE_TRAINED_FASHION_WEIGHTS,
                                     by_name=True,
                                     exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

        es_cb = EarlyStopping(monitor='val_mrcnn_mask_loss', patience=2)

        self._callbacks = [es_cb]

    def train_model(self):
        self._model.train(self._train_data_set, self._val_data_set,
                          learning_rate=self._lr,
                          epochs=self._nb_epochs,
                          layers='heads',
                          custom_callbacks=self._callbacks,
                          augmentation=self._augmentation)

    def set_to_train_mode(self):
        self._model = modellib.MaskRCNN(mode='training', config=self._model_config, model_dir=DL_MODELS_PATH)
        self._model.load_weights(PRE_TRAINED_FASHION_WEIGHTS,
                                 by_name=True,
                                 exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
        self._in_inference_mode = False

    def set_to_inference_mode(self, model_weights_path=PRE_TRAINED_FASHION_WEIGHTS):
        inf_conf = InferenceConfig()
        self._model = modellib.MaskRCNN(mode='inference',
                                        config=inf_conf,
                                        model_dir=DL_MODELS_PATH)
        self._model.load_weights(model_weights_path,
                                 by_name=True)
        self._in_inference_mode = True

    def predict(self):
        sub_list = []
        missing_count = 0
        for i, row in tqdm(self._sample_df.iterrows(), total=len(self._sample_df)):
            image = resize_image('{0}{1}'.format(FGVC6_TEST_IMAGES_FOLDER_PATH, row['ImageId']))
            result = self._model.detect([image])[0]
            if result['masks'].size > 0:
                masks, _ = refine_masks(result['masks'], result['rois'])
                for m in range(masks.shape[-1]):
                    mask = masks[:, :, m].ravel(order='F')
                    rle = to_rle(mask)
                    label = result['class_ids'][m] - 1
                    sub_list.append([row['ImageId'], ' '.join(list(map(str, rle))), label])
            else:
                # The system does not allow missing ids, this is an easy way to fill them
                sub_list.append([row['ImageId'], '1 1', 23])
                missing_count += 1

        submission_df = pd.DataFrame(sub_list, columns=self._sample_df.columns.values)
        print("Total image results: ", submission_df['ImageId'].nunique())
        print("Missing Images: ", missing_count)
        submission_df.to_csv(FGVC6_SUBMISSION_CSV_PATH, index=False)

        submission_df.head()

    def visualize(self):
        for i in range(9):
            image_id = self._sample_df.sample()['ImageId'].values[0]
            image_path = str('{0}{1}'.format(FGVC6_TEST_IMAGES_FOLDER_PATH, image_id))

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = self._model.detect([resize_image(image_path)])
            r = result[0]

            if r['masks'].size > 0:
                masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
                for m in range(r['masks'].shape[-1]):
                    masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'),
                                                (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                y_scale = img.shape[0] / IMAGE_SIZE
                x_scale = img.shape[1] / IMAGE_SIZE
                rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

                masks, rois = refine_masks(masks, rois)
            else:
                masks, rois = r['masks'], r['rois']

            visualize.display_instances(img, rois, masks, r['class_ids'],
                                        ['bg'] + self._class_names, r['scores'],
                                        title=image_id, figsize=(12, 12))


if __name__ == "__main__":
    experiment = MaskRCNNExperiment(nb_epochs=2, val_split=.1)
    experiment.set_to_inference_mode()
    experiment.predict()
    experiment.visualize()
