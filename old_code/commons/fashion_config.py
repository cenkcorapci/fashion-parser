from commons.config import IMAGE_SIZE, NUM_CATS
from mrcnn.config import Config


class FashionConfig(Config):
    NAME = "fashion_resnet_101"
    NUM_CLASSES = NUM_CATS + 1  # +1 for the background class

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2  # a memory error occurs when IMAGES_PER_GPU is too high
    STEPS_PER_EPOCH = 10000000  # use all the data
    VALIDATION_STEPS = 100000
    BACKBONE = 'resnet101'

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)


class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
