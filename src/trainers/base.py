import torch
import os

from abc import abstractmethod
from numpy import inf
from utils.io import prepare_device
from logger import TensorboardWriter


import wandb



class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config):

        self.config = config
        self.logger = self.config.get_logger('trainer', config['trainer']['verbosity'])

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.eval_period = cfg_trainer['eval_period']

        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = inf if self.monitor_mode == 'min' else -inf

            # Only enable early stopping if given and above 0
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf # training proceeds till the very last epoch

        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = inf if self.monitor_mode == 'min' else -inf

            # Only enable early stopping if given and above 0
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf  # training proceeds till the very last epoch

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = None
        if config['tensorboard']:
            self.writer = TensorboardWriter(config.log_dir, self.logger)


        self.start_epoch = 1
        self.best_epoch = 1
        self.best_epoch = 1
        self.current_epoch = 1
        self.best_top1 = 0

        # prepare for (multi-device) GPU training
        # This part doesn't do anything if you don't have a GPU.
        self._device, self._device_ids = prepare_device(config['n_gpu'])
        self.wandb_enabled = config['wandb']
        if self.wandb_enabled:
            wandb.login()
            wandb.init(project="human-pose-prediction-in-the-wild")

    

    @abstractmethod
    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        self.not_improved_count = 0
        prev_metric_value = 999999999999999 if self.monitor_mode == "min" else 0        
        if self.wandb_enabled: wandb.watch(self.model, self.criterion, log='all')
        states = {"loss": {"train": [], "val": []}}
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            train_result = self._train_epoch()
            states["loss"]["train"].append(train_result["loss"])
            # save logged informations into log dict
            log = {'epoch': self.current_epoch}
            log.update(train_result)

            if self._do_evaluate():
                eval_result = self.evaluate()
                states["loss"]["val"].append(eval_result["loss"])
                # save eval information to the log dict as well
                log.update({f'eval_{key}': value for key, value in eval_result.items()})
                if self.wandb_enabled:
                    wandb.log(log)



            if self.monitor_mode != 'off' : # Then there is a metric to monitor
                if self.monitor_metric in log: # Then we have measured it in this epoch

                    
                    metric_value = log[self.monitor_metric]
                    if self.monitor_mode == "min" and metric_value < prev_metric_value:
                        self.not_improved_count = 0
                        path = os.path.join(self.checkpoint_dir, f'best_model.pth')
                        self.save_model(path=path)
                        self.logger.info(f"Saving model with best metric at {path}")                        
                        prev_metric_value = metric_value

                    elif self.monitor_mode == "max" and metric_value > prev_metric_value:
                        self.not_improved_count = 0
                        path = os.path.join(self.checkpoint_dir, f'best_model.pth')
                        self.save_model(path=path)      
                        self.logger.info(f"Saving model with best metric at {path}")                                          
                        prev_metric_value = metric_value
                    
                    else:
                        if self.early_stop:
                            self.not_improved_count += 1
                            if self.not_improved_count == self.early_stop:
                                self.logger.info(f"No improvement so far... breaking...goodby")
                                break
                            self.logger.info(f"Patience running out... {self.not_improved_count}")           


                else:
                    ## The metric wasn't measured in this epoch. Don't change not_impoved_count or similar things here!!!
                    self.logger.warning(f"Warning: At epoch {self.current_epoch} Metric '{self.monitor_metric}' wasn't measured. Not monitoring it for this epoch.")
            
            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if self.wandb_enabled: wandb.log(log)

            if self.current_epoch % self.save_period == 0:
                # Just to regularly save the model every save_period epochs
                path = os.path.join(self.checkpoint_dir, f'per_epoch_model.pth')
                self.save_model(path=path)
        
        # Always save the last model
        path = os.path.join(self.checkpoint_dir, f'last_model.pth')
        self.save_model(path=path)
        return states

    def _do_evaluate(self):
        """
        Based on the self.current_epoch and self.eval_interval, determine if we should evaluate.
        You can take hint from saving logic implemented in BaseTrainer.train() method

        returns a Boolean
        """
        if self.current_epoch % self.eval_period == 0:
            return True
        else:
            return False
    
    @abstractmethod
    def evaluate(self, loader=None):
        """
        Evaluate the model on the val_loader given at initialization

        :param loader: A Dataloader to be used for evaluation. If not given, it will use the 
        self._eval_loader that's set during initialization..
        :return: A dict that contains metric(s) information for validation set
        """
        raise NotImplementedError
    
    def save_model(self, path=None):
        """
        Saves only the model parameters.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Saving checkpoint: {} ...".format(path))
        torch.save(self.model.state_dict(), path)
        self.logger.info("Checkpoint saved.")
    
    def load_model(self, path=None):
        self.logger.info("Saving checkpoint: {} ...".format(path))
        torch.save(self.model.state_dict(), path)
        self.logger.info("Checkpoint saved.")

    def load_model(self, path=None):
        """
        Loads model params from the given path.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Loading checkpoint: {} ...".format(path))
        self.model.load_state_dict(torch.load(path))
        self.logger.info("Checkpoint loaded.")


    def save_checkpoint(self, path=None):
        """
        Saving TRAINING checkpoint. Including the model params and other training stats
        (optimizer, current epoch, etc.)

        :param path: if True, rename the saved checkpoint to 'model_best.ckpt'
        :param path: if True, rename the saved checkpoint to 'model_best.ckpt'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        torch.save(state, path)
        self.logger.info("Saving checkpoint: {} ...".format(path))


    def resume_checkpoint(self, resume_path=None):
        """
        Loads TRAINING checkpoint. Including the model params and other training stats
        (optimizer, current epoch, etc.)

        :param path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model'], strict=False)
        if len(missing_keys) > 0:
            self.logger.warning(f"[WARNING] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.logger.warning(f"[WARNING] unexpected keys: {unexpected_keys}")

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # load lr_scheduler state from checkpoint only when lr_scheduler type is not changed.
        if checkpoint['config']['lr_scheduler']['type'] != self.config['lr_scheduler']['type']:
            self.logger.warning("Warning: lr_scheduler type given in config file is different from that of checkpoint. "
                                "lr_scheduler parameters not being resumed.")
        else:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
