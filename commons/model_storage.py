import torch.onnx

from commons.config import DL_MODELS_PATH


class ModelStorage:
    def __init__(self, model_name):
        self._model_name = model_name
        self._weights_file_path = DL_MODELS_PATH + self._model_name

    def persist(self, model):
        torch.save(model.state_dict(), self._weights_file_path + '.pt')

    def load(self, model):
        model.load_state_dict(torch.load(self._weights_file_path + '.pt'))
        return model
