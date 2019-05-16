import keras
from keras.applications import *
from keras.callbacks import *
from livelossplot import PlotLossesKeras

from utils.ai_utils import mean_iou, iou_bce_loss


class EncoderModelType:
    resnet = "ResNet"
    vgg16 = "vgg16"


class EncoderDecoderModel:
    def __init__(self, encoder_model=EncoderModelType.resnet, nb_epochs=10, image_size=320, early_stopping_patience=3,
                 n_valid_samples=2560,
                 batch_size=16, depth=4, channels=32, n_blocks=2, augment_images=True, debug_sample_size=None):
        self.model_name = 'resnet'
        self.weight_file_path = MODEL_BINARIES_PATH + self.model_name + '.h5'
        self.n_valid_samples = n_valid_samples
        self.nb_epochs = nb_epochs
        self.image_size = image_size
        self.augment_images = augment_images
        self.batch_size = batch_size
        self.depth = depth
        self.channels = channels
        self.n_blocks = n_blocks

        tb_callback = TensorBoard(log_dir=TB_LOGS_PATH, histogram_freq=0, write_graph=True,
                                  write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None)

        self.callbacks = [tb_callback, PlotLossesKeras()]
        self.callbacks.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience))
        self.callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                                patience=2,
                                                verbose=1,
                                                factor=0.1,
                                                min_lr=0.0001))
        self.callbacks.append(ModelCheckpoint(self.weight_file_path, monitor='val_loss', save_best_only=True))

        if debug_sample_size is not None:
            self.debug_sample_size = debug_sample_size
        self.load_data()
        if encoder_model == EncoderModelType.resnet:
            self.model = self.create_resnet_network(input_size=self.image_size,
                                                    channels=self.channels,
                                                    n_blocks=self.n_blocks,
                                                    depth=self.depth)
        elif encoder_model == EncoderModelType.vgg16:
            self.model = self.create_vgg16_network()
        self.model.compile(optimizer='adam',
                           loss=iou_bce_loss,
                           metrics=['accuracy', mean_iou])

    def create_downsample(self, channels, inputs):
        x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
        x = keras.layers.MaxPool2D((2, 2))(x)
        return x

    def create_resblock(self, channels, inputs):
        x = keras.layers.BatchNormalization(momentum=0.9999)(inputs)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization(momentum=0.9999)(x)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
        x = keras.layers.SpatialDropout2D(0.5)(x)
        return keras.layers.add([x, inputs])

    def create_vgg16_network(self, depth=4):
        # create the base pre-trained model
        base_model = VGG16(weights='imagenet', include_top=False)
        x = base_model.output

        # output
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
        outputs = keras.layers.UpSampling2D(2 ** depth)(x)
        model = keras.Model(inputs=base_model.input, outputs=outputs)
        return model

    def create_resnet_network(self, input_size, channels, n_blocks=2, depth=4):
        # input
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
        # residual blocks
        for d in range(depth):
            channels = channels * 2
            x = self.create_downsample(channels, x)
            for b in range(n_blocks):
                x = self.create_resblock(channels, x)
        # output
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
        outputs = keras.layers.UpSampling2D(2 ** depth)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compile_network(self):
        # create network and compiler
        model = self.create_network(input_size=self.image_size, channels=self.channels, n_blocks=self.n_blocks,
                                    depth=self.depth)
        model.compile(optimizer='adam',
                      loss=iou_bce_loss,
                      metrics=['accuracy', mean_iou])
        return model
