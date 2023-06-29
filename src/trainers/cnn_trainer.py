import torch
import torch.nn as nn

from tqdm import tqdm

from torchvision.utils import make_grid
from .base import BaseTrainer
from utils.io import MetricTracker

import models.cnn.model as module_arch
import models.cnn.loss as module_loss
import models.cnn.metric as module_metric


class CNNTrainer(BaseTrainer):

    def __init__(self, config, train_loader, eval_loader=None):
        """
        Create the model, loss criterion, optimizer, and dataloaders
        And anything else that might be needed during training. (e.g. device type)
        """
        super().__init__(config)
    
        # build model architecture, then print to console
        self.model = config.init_obj('arch', module_arch)
        self.model.to(self._device)
        if len(self._device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self._device_ids)

        # Initialize the model weights based on weights_init logic
        self.model.apply(self.weights_init)

        # Simply Log the model
        self.logger.info(self.model)

        # Prepare Losses
        self.criterion = getattr(module_loss, config['loss'])

        # Prepare Optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer) 

        # Set DataLoaders
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        
        self.log_step = 100 # arbitrary

        # Prepare Metrics
        self.metric_ftns = [getattr(module_metric, met['type'])(**met['args']) for met in config['metrics']]
        self.epoch_metrics = MetricTracker(keys=['loss'] + [str(m) for m in self.metric_ftns], writer=self.writer)
        self.eval_metrics = MetricTracker(keys=['loss'] + [str(m) for m in self.metric_ftns], writer=self.writer)

    def weights_init(self, m):
        """
        Initializes the model weights! Must be used with .apply of an nn.Module so that it works recursively!
        """
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 1e-2)
            nn.init.normal_(m.bias, 0.0, 1e-2)

    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        #######
        # Set model to train mode
        ######
        self.model.train()
        self.epoch_metrics.reset()

        self.logger.debug(f"==> Start Training Epoch {self.current_epoch}/{self.epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ")

        pbar = tqdm(total=len(self._train_loader) * self._train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (images, labels) in enumerate(self._train_loader):

            images = images.to(self._device)
            labels = labels.to(self._device)

            self.optimizer.zero_grad()

            output = self.model(images)
            loss = self.criterion(output, labels)

            loss.backward()
            self.optimizer.step()

            if self.writer is not None: self.writer.set_step((self.current_epoch - 1) * len(self._train_loader) + batch_idx)
            self.epoch_metrics.update('loss', loss.item())
            for metric in self.metric_ftns:
                self.epoch_metrics.update(str(metric), metric.compute(output, labels))

            pbar.set_description(f"Train Epoch: {self.current_epoch} Loss: {loss.item():.6f}")

            if batch_idx % self.log_step == 0:
                # self.logger.debug('Train Epoch: {} Loss: {:.6f}'.format(self.current_epoch, loss.item()))
                if self.writer is not None: self.writer.add_image('input_train', make_grid(images.cpu(), nrow=8, normalize=True))

            pbar.update(self._train_loader.batch_size)

        log_dict = self.epoch_metrics.result()
        pbar.close()
        self.lr_scheduler.step()

        self.logger.debug(f"==> Finished Epoch {self.current_epoch}/{self.epochs}.")
        
        return log_dict
    
    @torch.no_grad()
    def evaluate(self, loader=None):
        """
        Evaluate the model on the val_loader given at initialization

        :param loader: A Dataloader to be used for evaluatation. If not given, it will use the 
        self._eval_loader that's set during initialization..
        :return: A dict that contains metric(s) information for validation set
        """
        if loader is None:
            assert self._eval_loader is not None, 'loader was not given and self._eval_loader not set either!'
            loader = self._eval_loader

        self.model.eval()
        self.eval_metrics.reset()

        self.logger.debug(f"++> Evaluate at epoch {self.current_epoch} ...")

        pbar = tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx, (images, labels) in enumerate(loader):
            
            images = images.to(self._device)
            labels = labels.to(self._device)

            output = self.model(images)
            loss = self.criterion(output, labels)

            if self.writer is not None: self.writer.set_step((self.current_epoch - 1) * len(loader) + batch_idx, 'valid')
            self.eval_metrics.update('loss', loss.item())
            for metric in self.metric_ftns:
                self.eval_metrics.update(str(metric), metric.compute(output, labels))

            pbar.set_description(f"Eval Loss: {loss.item():.6f}")
            if self.writer is not None: self.writer.add_image('input_valid', make_grid(images.cpu(), nrow=8, normalize=True))

            pbar.update(loader.batch_size)

        # add histogram of model parameters to the tensorboard
        # if self.writer is not None:
        #     for name, p in self.model.named_parameters():
        #         self.writer.add_histogram(name, p, bins='auto')

        pbar.close()
        self.logger.debug(f"++> Evaluate epoch {self.current_epoch} Finished.")
        
        return self.eval_metrics.result()