import torch.onnx
from torch.autograd import Variable

from commons.config import DL_MODELS_PATH


class ModelStorage:
    def __init__(self, model_name, image_size, device):
        self._model_name = model_name
        self._dummy_input = Variable(torch.randn(10, 3, image_size, image_size)).to(device)
        self._weights_file_path = DL_MODELS_PATH + self._model_name

    def persist(self, model):
        torch.onnx.export(model, self._dummy_input, self._weights_file_path + '.onnx')
        torch.save(model.state_dict(), self._weights_file_path + '.pt')

    def load(self, model):
        model.load_state_dict(torch.load(self._weights_file_path + '.pt'))
        return model
