# -*- coding: utf-8 -*-
import os
import copy
import time
import pickle
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from IPython import display


class DeepNetTrainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, lr_scheduler=None, callbacks=None, use_gpu='auto'):

        assert (model is not None) and (criterion is not None) and (optimizer is not None)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
        self.last_epoch = 0

        self.callbacks = []
        if callbacks is not None:
            for cb in callbacks:
                self.callbacks.append(cb)
                cb.trainer = self

        self.use_gpu = use_gpu
        if use_gpu == 'auto':
            self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()

    def fit(self, n_epochs, Xin, Yin, valid_data=None, valid_split=None, batch_size=10, shuffle=True):
        if valid_data is not None:
            train_loader = DataLoader(TensorDataset(Xin, Yin), batch_size=batch_size, shuffle=shuffle)
            valid_loader = DataLoader(TensorDataset(*valid_data), batch_size=batch_size, shuffle=shuffle)
        elif valid_split is not None:
            iv = int(valid_split * Xin.shape[0])
            Xval, Yval = Xin[:iv], Yin[:iv]
            Xtra, Ytra = Xin[iv:], Yin[iv:]
            train_loader = DataLoader(TensorDataset(Xtra, Ytra), batch_size=batch_size, shuffle=shuffle)
            valid_loader = DataLoader(TensorDataset(Xval, Yval), batch_size=batch_size, shuffle=shuffle)
        else:
            train_loader = DataLoader(TensorDataset(Xin, Yin), batch_size=batch_size, shuffle=shuffle)
            valid_loader = None
        self.fit_loader(n_epochs, train_loader, valid_data=valid_loader)

    def evaluate(self, Xin, Yin, metrics=None, batch_size=10):
        dloader = DataLoader(TensorDataset(Xin, Yin), batch_size=batch_size, shuffle=False)
        return self.evaluate_loader(dloader, metrics)

    def _do_optimize(self, X, Y):
        self.optimizer.zero_grad()
        Ypred = self.model.forward(X)
        loss = self.criterion(Ypred, Y)
        loss.backward()
        self.optimizer.step()
        return Ypred, loss

    def _do_evaluate(self, X, Y):
        Ypred = self.model.forward(X)
        loss = self.criterion(Ypred, Y)
        return Ypred, loss

    def fit_loader(self, n_epochs, train_data, valid_data=None):
        self.has_validation = valid_data is not None
        try:
            for cb in self.callbacks:
                cb.on_train_begin(n_epochs, self.metrics)

            # for each epoch
            for curr_epoch in range(self.last_epoch + 1, self.last_epoch + n_epochs + 1):

                # training phase
                # ==============
                self.model.train(True)
                for cb in self.callbacks:
                    cb.on_epoch_begin(curr_epoch, self.metrics)

                epo_samples = 0
                epo_batches = 0
                epo_loss = 0

                if self.scheduler is not None:
                    self.scheduler.step()

                # for each minibatch
                for curr_batch, (X, Y) in enumerate(train_data):

                    mb_size = X.size(0)
                    epo_samples += mb_size
                    epo_batches += 1

                    for cb in self.callbacks:
                        cb.on_batch_begin(curr_epoch, curr_batch, mb_size)

                    if self.use_gpu:
                        X, Y = Variable(X.cuda()), Variable(Y.cuda())
                    else:
                        X, Y = Variable(X), Variable(Y)

                    Ypred, loss = self._do_optimize(X, Y)

                    vloss = loss.data.cpu()[0]
                    if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                        epo_loss += mb_size * vloss
                    else:
                        epo_loss += vloss

                    for cb in self.callbacks:
                        cb.on_batch_end(curr_epoch, curr_batch, X, Y, Ypred, loss)

                # end of training minibatches
                eloss = float(epo_loss / epo_samples)
                self.metrics['train']['losses'].append(eloss)

                # validation phase
                # ================
                self.model.train(False)
                if self.has_validation:
                    epo_samples = 0
                    epo_batches = 0
                    epo_loss = 0

                    # for each minibatch
                    for curr_batch, (X, Y) in enumerate(valid_data):
                        mb_size = X.size(0)
                        epo_samples += mb_size
                        epo_batches += 1

                        for cb in self.callbacks:
                            cb.on_vbatch_begin(curr_epoch, curr_batch, mb_size)

                        if self.use_gpu:
                            X, Y = Variable(X.cuda(), volatile=True), Variable(Y.cuda(), volatile=True)
                        else:
                            X, Y = Variable(X, volatile=True), Variable(Y, volatile=True)

                        Ypred, loss = self._do_evaluate(X, Y)

                        vloss = loss.data.cpu()[0]
                        if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                            epo_loss += vloss * mb_size
                        else:
                            epo_loss += vloss

                        for cb in self.callbacks:
                            cb.on_vbatch_end(curr_epoch, curr_batch, X, Y, Ypred, loss)

                    # end minibatches
                    eloss = float(epo_loss / epo_samples)
                    self.metrics['valid']['losses'].append(eloss)

                else:
                    self.metrics['valid']['losses'].append(None)

                for cb in self.callbacks:
                    cb.on_epoch_end(curr_epoch, self.metrics)

        except KeyboardInterrupt:
            pass

        for cb in self.callbacks:
            cb.on_train_end(n_epochs, self.metrics)

    def evaluate_loader(self, data_loader, metrics=None):
        metrics = metrics or []
        my_metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
        for cb in metrics:
            cb.on_train_begin(1, my_metrics)
            cb.on_epoch_begin(1, my_metrics)

        epo_samples = 0
        epo_batches = 0
        epo_loss = 0

        try:
            self.model.train(False)
            ii_n = len(data_loader)

            for curr_batch, (X, Y) in enumerate(data_loader):
                mb_size = X.size(0)
                epo_samples += mb_size
                epo_batches += 1

                if self.use_gpu:
                    X, Y = Variable(X.cuda()), Variable(Y.cuda())
                else:
                    X, Y = Variable(X), Variable(Y)

                Ypred, loss = self._do_evaluate(X, Y)

                vloss = loss.data.cpu()[0]
                if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                    epo_loss += vloss * mb_size
                else:
                    epo_loss += vloss

                for cb in metrics:
                    cb.on_batch_end(1, curr_batch, X, Y, Ypred, loss) # RAL
                             
                print('\revaluate: {}/{}'.format(curr_batch, ii_n - 1), end='')
            print(' ok')

        except KeyboardInterrupt:
            print(' interrupted!')

        if epo_batches > 0:
            epo_loss /= epo_samples
            my_metrics['train']['losses'].append(epo_loss)
            for cb in metrics:
                cb.on_epoch_end(1, my_metrics)

        return dict([(k, v[0]) for k, v in my_metrics['train'].items()])

    def load_state(self, file_basename):
        if self.use_gpu:
            self.model.cpu()
        load_trainer_state(file_basename, self.model, self.metrics)
        self.last_epoch = len(self.metrics['train']['losses'])
        if self.use_gpu:
            self.model.cuda()

    def save_state(self, file_basename):
        if self.use_gpu:
            cpu_model = copy.deepcopy(self.model)
            cpu_model.cpu()
        else:
            cpu_model = self.model
        save_trainer_state(file_basename, cpu_model, self.metrics)

    def predict_loader(self, data_loader):
        self.model.train()
        predictions = []
        for X, _ in data_loader:
            if self.use_gpu:
                X = Variable(X.cuda())
            else:
                X = Variable(X)

            Ypred = self.model(X)
            Ypred = Ypred.cpu().data
            predictions.append(Ypred)
        return torch.cat(predictions, 0)

    def predict(self, Xin):
        if self.use_gpu:
            Xin = Xin.cuda()
        return predict(self.model, Xin)

    def predict_classes_loader(self, data_loader):
        y_pred = self.predict_loader(data_loader)
        _, pred = torch.max(y_pred, 1)
        return pred

    def predict_classes(self, Xin):
        if self.use_gpu:
            Xin = Xin.cuda()
        return predict_classes(self.model, Xin)

    def predict_probas_loader(self, data_loader):
        y_pred = self.predict_loader(data_loader)
        probas = F.softmax(y_pred)
        return probas

    def predict_probas(self, Xin):
        if self.use_gpu:
            Xin = Xin.cuda()
        return predict_probas(self.model, Xin)

    def summary(self):
        pass


def load_trainer_state(file_basename, model, metrics):
    model.load_state_dict(torch.load(file_basename + '.model', map_location=lambda storage, loc: storage))
    if os.path.isfile(file_basename + '.histo'):
        metrics.update(pickle.load(open(file_basename + '.histo', 'rb')))


def save_trainer_state(file_basename, model, metrics):
    torch.save(model.state_dict(), file_basename + '.model')
    pickle.dump(metrics, open(file_basename + '.histo', 'wb'))


def predict(model, Xin):
    y_pred = model.forward(Variable(Xin))
    return y_pred.data


def predict_classes(model, Xin):
    y_pred = predict(model, Xin)
    _, pred = torch.max(y_pred, 1)
    return pred


def predict_probas(model, Xin):
    y_pred = predict(model, Xin)
    probas = F.softmax(y_pred)
    return probas


class Callback(object):
    def __init__(self):
        pass

    def on_train_begin(self, n_epochs, metrics):
        pass

    def on_train_end(self, n_epochs, metrics):
        pass

    def on_epoch_begin(self, epoch, metrics):
        pass

    def on_epoch_end(self, epoch, metrics):
        pass

    def on_batch_begin(self, epoch, batch, mb_size):
        pass

    def on_batch_end(self, epoch, batch, x, y, y_pred, loss):
        pass

    def on_vbatch_begin(self, epoch, batch, mb_size):
        pass

    def on_vbatch_end(self, epoch, batch, x, y, y_pred, loss):
        pass


class AccuracyMetric(Callback):
    def __init__(self):
        super().__init__()
        self.name = 'acc'

    def on_batch_end(self, epoch_num, batch_num, x, y_true, y_pred, loss):
        _, preds = torch.max(y_pred.data, 1)
        ok = (preds == y_true.data).sum()
        self.train_accum += ok
        self.n_train_samples += y_pred.size(0)

    def on_vbatch_end(self, epoch_num, batch_num, x, y_true, y_pred, loss):
        _, preds = torch.max(y_pred.data, 1)
        ok = (preds == y_true.data).sum()
        self.valid_accum += ok
        self.n_valid_samples += y_pred.size(0)

    def on_epoch_begin(self, epoch_num, metrics):
        self.train_accum = 0
        self.valid_accum = 0
        self.n_train_samples = 0
        self.n_valid_samples = 0

    def on_epoch_end(self, epoch_num, metrics):
        if self.n_train_samples > 0:
            metrics['train'][self.name].append(1.0 * self.train_accum / self.n_train_samples)
        if self.n_valid_samples > 0:
            metrics['valid'][self.name].append(1.0 * self.valid_accum / self.n_valid_samples)

    def on_train_begin(self, n_epochs, metrics):
        metrics['train'][self.name] = []
        metrics['valid'][self.name] = []


class ModelCheckpoint(Callback):

    def __init__(self, model_basename, reset=False, verbose=0):
        super().__init__()
        os.makedirs(os.path.dirname(model_basename), exist_ok=True)
        self.basename = model_basename
        self.reset = reset
        self.verbose = verbose

    def on_train_begin(self, n_epochs, metrics):
        if (self.basename is not None) and (not self.reset) and (os.path.isfile(self.basename + '.model')):
            load_trainer_state(self.basename, self.trainer.model, self.trainer.metrics)
            if self.verbose > 0:
                print('Model loaded from', self.basename + '.model')

        self.trainer.last_epoch = len(self.trainer.metrics['train']['losses'])
        if self.trainer.scheduler is not None:
            self.trainer.scheduler.last_epoch = self.trainer.last_epoch

        self.best_model = copy.deepcopy(self.trainer.model)
        self.best_epoch = self.trainer.last_epoch
        self.best_loss = 1e10
        if self.trainer.last_epoch > 0:
            self.best_loss = self.trainer.metrics['valid']['losses'][-1] or self.trainer.metrics['train']['losses'][-1]

    def on_train_end(self, n_epochs, metrics):
        if self.verbose > 0:
            print('Best model was saved at epoch {} with loss {:.5f}: {}'
                  .format(self.best_epoch, self.best_loss, self.basename))

    def on_epoch_end(self, epoch, metrics):
        eloss = metrics['valid']['losses'][-1] or metrics['train']['losses'][-1]
        if eloss < self.best_loss:
            self.best_loss = eloss
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(self.trainer.model)
            if self.basename is not None:
                save_trainer_state(self.basename, self.trainer.model, self.trainer.metrics)
                if self.verbose > 1:
                    print('Model saved to', self.basename + '.model')


class PrintCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_train_begin(self, n_epochs, metrics):
        print('Start training for {} epochs'.format(n_epochs))

    def on_train_end(self, n_epochs, metrics):
        n_train = len(metrics['train']['losses'])
        print('Stop training at epoch: {}/{}'.format(n_train, self.trainer.last_epoch + n_epochs))

    def on_epoch_begin(self, epoch, metrics):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, metrics):
            is_best = ''
            has_valid = len(metrics['valid']['losses']) > 0 and metrics['valid']['losses'][0] is not None
            has_metrics = len(metrics['train'].keys()) > 1
            etime = time.time() - self.t0

            if has_valid:
                if epoch == int(np.argmin(metrics['valid']['losses'])) + 1:
                    is_best = 'best'
                if has_metrics:
                    # validation and metrics
                    metric_name = [mn for mn in metrics['valid'].keys() if mn != 'losses'][0]
                    # metric_name = list(self.trainer.compute_metric.keys())[0]
                    print('{:3d}: {:5.1f}s   T: {:.5f} {:.5f}   V: {:.5f} {:.5f} {}'
                          .format(epoch, etime,
                                  metrics['train']['losses'][-1],
                                  metrics['train'][metric_name][-1],
                                  metrics['valid']['losses'][-1],
                                  metrics['valid'][metric_name][-1], is_best))
                else:
                    # validation and no metrics
                    print('{:3d}: {:5.1f}s   T: {:.5f}   V: {:.5f} {}'
                          .format(epoch, etime,
                                  metrics['train']['losses'][-1],
                                  metrics['valid']['losses'][-1], is_best))
            else:
                if epoch == int(np.argmin(metrics['train']['losses'])) + 1:
                    is_best = 'best'
                if has_metrics:
                    # no validation and metrics
                    metric_name = list(self.trainer.compute_metric.keys())[0]
                    print('{:3d}: {:5.1f}s   T: {:.5f} {:.5f} {}'
                          .format(epoch, etime,
                                  metrics['train']['losses'][-1],
                                  metrics['train'][metric_name][-1], is_best))
                else:
                    # no validation and no metrics
                    print('{:3d}: {:5.1f}s   T: {:.5f} {}'
                          .format(epoch, etime,
                                  metrics['train']['losses'][-1], is_best))


class PlotCallback(Callback):
    def __init__(self, interval=1, max_loss=None):
        super().__init__()
        self.interval = interval
        self.max_loss = max_loss

    def on_train_begin(self, n_epochs, metrics):
        self.line_train = None
        self.line_valid = None
        self.dot_train = None
        self.dot_valid = None

        self.fig = plt.figure(figsize=(15, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.grid(True)

        self.plot_losses(self.trainer.metrics['train']['losses'],
                         self.trainer.metrics['valid']['losses'])

    def on_epoch_end(self, epoch, metrics):
        if epoch % self.interval == 0:
            display.clear_output(wait=True)
            self.plot_losses(self.trainer.metrics['train']['losses'],
                             self.trainer.metrics['valid']['losses'])

    def plot_losses(self, htrain, hvalid):
        epoch = len(htrain)
        if epoch == 0:
            return

        x = np.arange(1, epoch + 1)
        if self.line_train:
            self.line_train.remove()
        if self.dot_train:
            self.dot_train.remove()
        self.line_train, = self.ax.plot(x, htrain, color='#1f77b4', linewidth=2, label='training loss')
        best_epoch = int(np.argmin(htrain)) + 1
        best_loss = htrain[best_epoch - 1]
        self.dot_train = self.ax.scatter(best_epoch, best_loss, c='#1f77b4', marker='o')

        if hvalid[0] is not None:
            if self.line_valid:
                self.line_valid.remove()
            if self.dot_valid:
                self.dot_valid.remove()
            self.line_valid, = self.ax.plot(x, hvalid, color='#ff7f0e', linewidth=2, label='validation loss')
            best_epoch = int(np.argmin(hvalid)) + 1
            best_loss = hvalid[best_epoch - 1]
            self.dot_valid = self.ax.scatter(best_epoch, best_loss, c='#ff7f0e', marker='o')

        self.ax.legend()
        # self.ax.vlines(best_epoch, *self.ax.get_ylim(), colors='#EBDDE2', linestyles='dashed')
        self.ax.set_title('Best epoch: {}, Current epoch: {}'.format(best_epoch, epoch))

        display.display(self.fig)
        time.sleep(0.1)


def plot_losses(htrain, hvalid):
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True)

    epoch = len(htrain)
    x = np.arange(1, epoch + 1)

    best_epoch = int(np.argmin(htrain)) + 1
    best_loss = htrain[best_epoch - 1]
    ax.plot(x, htrain, color='#1f77b4', linewidth=2, label='training loss')
    ax.scatter(best_epoch, best_loss, c='#1f77b4', marker='o')

    if hvalid[0] is not None:
        best_epoch = int(np.argmin(hvalid)) + 1
        best_loss = hvalid[best_epoch - 1]
        ax.plot(x, hvalid, color='#ff7f0e', linewidth=2, label='validation loss')
        ax.scatter(best_epoch, best_loss, c='#ff7f0e', marker='o')

    ax.legend()
    ax.set_title('Best epoch: {}, Current epoch: {}'.format(best_epoch, epoch))
