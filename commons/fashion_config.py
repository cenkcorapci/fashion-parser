from commons.config import IMAGE_SIZE, NUM_CATS
from mrcnn.config import Config


class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1  # +1 for the background class

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4  # a memory error occurs when IMAGES_PER_GPU is too high

    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)


class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
