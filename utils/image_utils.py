from keras.layers import *
from keras.models import *


def resize_masks(masks, shape=(46, 1024, 1024)):
    model_resize = Sequential()
    model_resize.add(MaxPool2D((2, 2), data_format='channels_first', input_shape=shape))
    model_resize.add(MaxPool2D((2, 2), data_format='channels_first'))
    model_resize.add(MaxPool2D((2, 2), data_format='channels_first'))
    return model_resize.predict(np.array([masks]))[0]
