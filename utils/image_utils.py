from keras.layers import *
from keras.models import *
from skimage.transform import resize


def resize_masks(masks, shape=(46, 1024, 1024)):
    model_resize = Sequential()
    model_resize.add(MaxPool2D((2, 2), data_format='channels_first', input_shape=shape))
    model_resize.add(MaxPool2D((2, 2), data_format='channels_first'))
    model_resize.add(MaxPool2D((2, 2), data_format='channels_first'))
    return model_resize.predict(np.array([masks]))[0]


def get_mask_images(masks, original_shape, resized_shape=(512, 512)):
    width, height = original_shape
    resized_masks = dict()
    for class_id, mask_encoded in masks.items():
        mask = [0] * (width * height)
        for j in range(0, len(mask_encoded), 2):
            mask[int(mask_encoded[j]): int(mask_encoded[j]) + int(mask_encoded[j + 1])] = [1] * int(mask_encoded[j + 1])
        mask = np.fliplr(np.flip(np.rot90(np.array(mask).reshape((width, height)))))
        mask = resize(mask, resized_shape, anti_aliasing=True)
        resized_masks[int(class_id)] = mask
    return resized_masks
