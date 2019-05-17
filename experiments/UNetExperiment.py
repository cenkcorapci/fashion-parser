# -*- coding: utf-8 -*-
from __future__ import print_function, division

import copy
import time

import torch
import torch.nn as nn
import torch.onnx
import torch.optim as torch_optimizer
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

from commons.config import *
from commons.model_storage import *
from data.data_loader import DataLoader
from data.sample_generator import FashionParsingDataSet
from models.unet import UNet
from monitoring.tensorboard import TensorBoardMonitoring


class UNetExperiment:
    def __init__(self,
                 debug_sample_size=None,
                 width=512,
                 height=512,
                 val_split=0.1,
                 early_stopping_at=3,
                 batch_size=16,
                 nb_epochs=3,
                 learning_rate=0.01):
        self._nb_epochs = nb_epochs
        self._batch_size = batch_size
        self._width = width
        self._height = height
        self._model_name = 'unet_parser'
        self._early_stopping_at = early_stopping_at
        self._debug_sample_size = debug_sample_size

        logging.info("Getting data set")
        self._data_loader = DataLoader()
        df = self._data_loader.get_training_data_set()
        if debug_sample_size is not None:
            df = df.sample(debug_sample_size)
        logging.info("Splitting {0} samples for validation".format(float(len(df)) * val_split))

        df_train_data, df_val_data = train_test_split(df, random_state=RANDOM_STATE, test_size=val_split)
        self._num_of_classes = self._data_loader.get_num_of_classes()
        logging.info("Getting data set")
        self._data_loader_train = torch.utils.data.DataLoader(
            dataset=FashionParsingDataSet(df_train_data, self._num_of_classes, self._width, self._height),
            batch_size=self._batch_size,
            shuffle=True)

        self._data_loader_val = torch.utils.data.DataLoader(
            dataset=FashionParsingDataSet(df_val_data, self._num_of_classes, self._width, self._height),
            batch_size=self._batch_size,
            shuffle=False)

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = self.generate_model()

        self._criterion = nn.MSELoss()

        # Observe that all parameters are being optimized
        self._optimizer = torch_optimizer.Adagrad(self._model.parameters(), lr=learning_rate)

        # Decay LR by a factor of 0.1 every 7 epochs
        self._scheduler = lr_scheduler.StepLR(self._optimizer, step_size=10, gamma=0.1)

        # TensorBoard
        self._tensorboard = TensorBoardMonitoring('{0}{1}/'.format(TB_LOGS_PATH, self._model_name))

        # Model storage
        self._model_storage = ModelStorage(self._model_name)

    def generate_model(self):
        return UNet(3, self._num_of_classes).to(self._device)

    def train_model(self):
        since = time.time()

        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_loss = 0.0
        left_for_early_stopping = self._early_stopping_at

        for epoch in tqdm(range(self._nb_epochs)):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self._scheduler.step()
                    self._model.train()  # Set model to training mode
                else:
                    self._model.eval()  # Set model to evaluate mode

                loader = self._data_loader_train if phase == 'train' else self._data_loader_val
                # Iterate over data.

                epoch_progress = tqdm(total=len(loader), desc='Samples from epoch: {0}'.format(epoch))
                epoch_loss = 0.
                inc = 0

                for X_trn, Y_trn in loader:
                    try:
                        X = X_trn.to(self._device, dtype=torch.float32)
                        Y = Y_trn.to(self._device, dtype=torch.float32)
                        inc += 1

                        # forward pass
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self._model(X)
                            loss = self._criterion(outputs if phase != 'train' else outputs[0], Y)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self._optimizer.step()
                            epoch_loss += loss.item()

                        # zero the parameter gradients
                        self._optimizer.zero_grad()

                        epoch_progress.update()
                        logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss / float(inc)))
                    except Exception as exp:
                        logging.error("Can't process sample", exp)

                epoch_val_loss = epoch_loss / float(inc)

                # deep copy the model and persist
                if phase == 'val':
                    if epoch_val_loss > best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self._model.state_dict())
                        self._model_storage.persist(self._model)
                        left_for_early_stopping = self._early_stopping_at
                    else:
                        left_for_early_stopping -= 1
                        logging.info("Accuracy not improved, will stop training after {0} epochs.".format(
                            left_for_early_stopping))
                        if left_for_early_stopping <= 0:
                            time_elapsed = time.time() - since
                            self._model.load_state_dict(best_model_wts)

                            logging.info(
                                'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                            logging.info(
                                'Best val Acc: {0:4f} after {1} / {2} epochs'.format(best_loss, epoch, self._nb_epochs))
                            return

                # log metrics to TensorBoard
                if phase == 'val':

                    self._tensorboard.log_scalar("val_loss", epoch_loss, epoch)

                    for k, v in list(self._model.state_dict().items()):
                        self._tensorboard.log_histogram("Layer {}".format(k), v.to('cpu').numpy(), epoch, bins=10)
                else:
                    self._tensorboard.log_scalar("train_loss", epoch_loss, epoch)

        time_elapsed = time.time() - since
        self._model.load_state_dict(state_dict=best_model_wts)

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_loss))


if __name__ == "__main__":
    model = UNetExperiment(batch_size=4, nb_epochs=10, early_stopping_at=3, debug_sample_size=1024)
    model.train_model()
