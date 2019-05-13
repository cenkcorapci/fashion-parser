import os

from keras.models import model_from_json

from commons.config import DL_MODELS_PATH, logging


class ModelStorage:
    def __init__(self, model_name):
        """
        Stores and loads the model on disk and AWS S3

        :param model_name: Model name to store.
        """
        self.model_name = model_name
        self.weights_file_path = DL_MODELS_PATH + self.model_name + '.h5'
        self.model_architecture_path = DL_MODELS_PATH + self.model_name + '.json'
        self.model = None

    def persist(self, model):
        self.model = model
        # serialize model to json and upload
        with open(self.model_architecture_path, "w") as json_file:
            json_file.write(self.model.to_json())

        print("Saved model architecture to disk")

        # serialize weights to HDF5 and upload
        self.model.save_weights(self.weights_file_path)
        print("Saved model weights to disk")

    def load(self):
        # Load model architecture
        if not os.path.exists(self.model_architecture_path):
            logging.error(
                "Can not model architecture json for  {0} on s3!".format(self.model_name))
            return None
        else:
            json_file = open(self.model_architecture_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)

        # Load model weights
        if not os.path.exists(self.weights_file_path):
            logging.error(
                "Can not find a pre-trained {0} weights on s3!".format(self.model_name))
            return None
        else:
            self.model.load_weights(self.weights_file_path)
            return self.model
