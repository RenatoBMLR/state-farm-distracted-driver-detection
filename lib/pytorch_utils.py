import os
import copy
import time
import pickle
import torch
import numpy as np
from torch.autograd import Variable


class DeepNetTrainer:

    def __init__(self, file_basename=None, model=None, criterion=None, metrics=None,
                 optimizer=None, lr_scheduler=None, callbacks=[], reset=False):
        assert (model is not None) and (criterion is not None) and (optimizer is not None)
        self.basename = file_basename
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
        self.compute_metric = dict()
        self.callbacks = callbacks

        if metrics is not None:
            for name, funct in metrics.items():
                self.metrics['train'][name] = []
                self.metrics['valid'][name] = []
                self.compute_metric[name] = funct

        if (self.basename is not None) and (not reset) and (os.path.isfile(self.basename + '.model')):
            self.load_trainer_state(self.basename, self.model, self.optimizer, self.metrics)
            print('Model loaded from', self.basename + '.model')

        self.last_epoch = len(self.metrics['train']['losses'])
        if self.scheduler is not None:
            self.scheduler.last_epoch = self.last_epoch

    def fit(self, n_epochs, train_data, valid_data=None, use_gpu='auto'):
        data = dict(train=train_data, valid=valid_data)
        if valid_data is None:
            phases = [('train', True)]
        else:
            phases = [('train', True), ('valid', False)]

        if use_gpu == 'auto':
            use_gpu = torch.cuda.is_available()
        assert not use_gpu or use_gpu

        try:
            print('Starting training for {} epochs\n'.format(n_epochs))

            best_epoch = self.last_epoch
            best_loss = 1e10
            if self.last_epoch > 0:
                best_loss = self.metrics['valid']['losses'][-1] or self.metrics['train']['losses'][-1]
                self.print_losses(self.last_epoch)

            for cb in self.callbacks:
                cb.on_train_begin(self, has_validation=(valid_data is not None))

            # for each epoch
            for i in range(self.last_epoch + 1, self.last_epoch + n_epochs + 1):
                t0 = time.time()

                # for training and evaluating
                for phase, is_train in phases:

                    epo_samp = 0
                    epo_loss = 0
                    epo_metrics = dict([(n, 0) for n in self.compute_metric.keys()])

                    self.model.train(is_train)
                    if is_train and self.scheduler is not None:
                        self.scheduler.step()

                    # for each minibatch
                    for ii, (X, Y) in enumerate(data[phase]):
                        if use_gpu:
                            X, Y = Variable(X.cuda()), Variable(Y.cuda())
                        else:
                            X, Y = Variable(X), Variable(Y)

                        Ypred = self.model.forward(X)
                        loss = self.criterion(Ypred, Y)
                        if is_train:
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        epo_loss += loss.data.cpu().numpy()
                        epo_samp += 1

                        for name, fun in self.compute_metric.items():
                            metric = fun(Ypred, Y)
                            epo_metrics[name] += metric

                    # end minibatch
                    eloss = float(epo_loss / epo_samp)
                    self.metrics[phase]['losses'].append(eloss)

                    for name, fun in self.compute_metric.items():
                        metric = float(epo_metrics[name] / epo_samp)
                        self.metrics[phase][name].append(metric)

                # end phase
                if valid_data is None:
                    self.metrics['valid']['losses'].append(None)
                    for name, fun in self.compute_metric.items():
                        self.metrics['valid'][name].append(None)

                is_best = ''
                if eloss < best_loss:
                    is_best = 'best'
                    best_loss = eloss
                    best_epoch = i
                    if self.basename is not None:
                        self.save_trainer_state(self.basename, self.model, self.optimizer, self.metrics)

                self.print_losses(i, t0, is_best)

                for cb in self.callbacks:
                    cb.on_epoch_end(self, i, best_epoch, t0)

                # t0 = time.time()

        except KeyboardInterrupt:
            print('Interrupted!!')

        print('\nModel from epoch {} saved as "{}.*", loss = {:.5f}'.format(best_epoch, self.basename, best_loss))

    def print_losses(self, i, t0=0, is_best='best'):
        has_valid = self.metrics['valid']['losses'][-1] is not None
        has_metrics = len(self.compute_metric) > 0
        etime = 0
        if t0 > 0:
            etime = time.time() - t0
        if has_valid and has_metrics:
            # validation and metrics
            mtrc = list(self.compute_metric.keys())[0]
            print('{:3d}: {:5.1f}s   T: {:.5f} {:.5f}   V: {:.5f} {:.5f} {}'
                  .format(i, etime,
                          self.metrics['train']['losses'][-1], self.metrics['train'][mtrc][-1],
                          self.metrics['valid']['losses'][-1], self.metrics['valid'][mtrc][-1], is_best))
        elif has_valid:
            # validation and no metrics
            print('{:3d}: {:5.1f}s   T: {:.5f}   V: {:.5f} {}'
                  .format(i, etime,
                          self.metrics['train']['losses'][-1],
                          self.metrics['valid']['losses'][-1], is_best))
        elif not has_valid and has_metrics:
            # no validation and metrics
            mtrc = list(self.compute_metric.keys())[0]
            print('{:3d}: {:5.1f}s   T: {:.5f} {:.5f} {}'
                  .format(i, etime,
                          self.metrics['train']['losses'][-1],
                          self.metrics['train'][mtrc][-1], is_best))
        else:
            # no validation and no metrics
            print('{:3d}: {:5.1f}s   T: {:.5f} {}'
                  .format(i, etime,
                          self.metrics['train']['losses'][-1], is_best))

    def predict(self, data_loader, use_gpu='auto'):
        if use_gpu == 'auto':
            use_gpu = torch.cuda.is_available()
        predictions = []
        try:
            self.model.train(False)  # Set model to evaluate mode
            ii_n = len(data_loader)
            for ii, (image, labels) in enumerate(data_loader):
                if use_gpu:
                    image = Variable(image.cuda())
                else:
                    image = Variable(image)
                outputs = self.model.forward(image)
                predictions.append(outputs.data.cpu())
                print('\rpredict: {}/{}'.format(ii, ii_n - 1), end='')
            print(' ok')
        except KeyboardInterrupt:
            print(' interrupted!')
        finally:
            if len(predictions) > 0:
                return torch.cat(predictions, 0)

    def evaluate(self, data_loader, metrics=None, use_gpu='auto'):
        if use_gpu == 'auto':
            use_gpu = torch.cuda.is_available()
        n_batches = 0
        try:
            if metrics is None:
                metric_dict = self.compute_metric
            else:
                metric_dict = metrics
            epo_metrics = {}
            for name in metric_dict.keys():  # zero all metrics
                epo_metrics[name] = 0
            self.model.train(False)  # Set model to evaluate mode
            ii_n = len(data_loader)
            for ii, (X, Y) in enumerate(data_loader):
                if use_gpu:
                    X, Y = Variable(X.cuda()), Variable(Y.cuda())
                else:
                    X, Y = Variable(X), Variable(Y)
                Ypred = self.model.forward(X)
                for name, fun in metric_dict.items():
                    vmetric = fun(Ypred, Y)
                    epo_metrics[name] += vmetric
                n_batches += 1
                print('\revaluate: {}/{}'.format(ii, ii_n - 1), end='')
            print(' ok')
        except KeyboardInterrupt:
            print(' interrupted!')
        finally:
            if n_batches > 0:
                for name in epo_metrics.keys():
                    epo_metrics[name] /= n_batches
                return epo_metrics

    @staticmethod
    def load_trainer_state(file_basename, model, optimizer, metrics):
        model.load_state_dict(torch.load(file_basename + '.model'))
        # if os.path.isfile(file_basename + '.optim'):
        #     optimizer.load_state_dict(torch.load(file_basename + '.optim'))
        if os.path.isfile(file_basename + '.histo'):
            metrics.update(pickle.load(open(file_basename + '.histo', 'rb')))

    @staticmethod
    def save_trainer_state(file_basename, model, optimizer, metrics):
        torch.save(model.state_dict(), file_basename + '.model')
        # torch.save(optimizer.state_dict(), file_basename + '.optim')
        pickle.dump(metrics, open(file_basename + '.histo', 'wb'))


def test_network(model, dataset, criterion, batch_size=32, use_gpu='auto'):
    temp_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

    # desliga o treinamento para nao executar o dropout
    model.train(False)

    if use_gpu == 'auto':
        use_gpu = torch.cuda.is_available()
    assert not use_gpu or use_gpu

    loss_sum = 0.0
    hit_sum = 0.0
    all_preds = np.zeros(len(dataset)).astype(int)
    all_probs = np.zeros(len(dataset)).astype(float)
    for i, data in enumerate(temp_dataloader):
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        loss_sum += criterion(outputs, labels)

        outputs = torch.nn.functional.softmax(outputs)
        probs, preds = torch.max(outputs, 1)
        curr_img_index = i * temp_dataloader.batch_size
        all_preds[curr_img_index:curr_img_index + labels.size(0)] = preds.data.cpu().numpy()
        all_probs[curr_img_index:curr_img_index + labels.size(0)] = probs.data.cpu().numpy()
        hit_sum += torch.sum(preds.data == labels.data)

    loss = loss_sum.data.cpu()[0] / len(temp_dataloader)
    accuracy = hit_sum / len(dataset)

    print("\nAccuracy on the test data set: {:.2f}% [{:.5f}]".format(accuracy * 100, loss))
    return (all_preds, all_probs)
