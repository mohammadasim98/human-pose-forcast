import os
import torch
import torch.nn as nn

from tqdm import tqdm

from torchvision.utils import make_grid
from trainers.base3d import BaseTrainer
from utils.io import MetricTracker

import models.hppw as module_arch
import models.hppw as module_metric
import models.hppw as module_loss
from models.hppw.transforms import cvt_relative_pose
from models.projection.model import LinearProjection



class HPPW3DTrainer(BaseTrainer):

    def __init__(self, config, train_loader, eval_loader=None):
        """
        Create the model, loss criterion, optimizer, and dataloaders
        And anything else that might be needed during training. (e.g. device type)
        """
        super().__init__(config)    
        # build model architecture, then print to console
        self.model = config.init_obj('arch', module_arch)
        self.model_proj = LinearProjection(**config["arch3d"])
        self.model.to(self._device)
        if len(self._device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self._device_ids)

        # Simply Log the model (enable if you want to see the model architecture)
        # self.logger.info(self.model)
        self.use_root_relative = config["use_root_relative"]
        self.use_pose_norm = config["use_pose_norm"]
        self.use_projection = config["use_projection"]
        # Prepare Losses
        # self.criterion = getattr(module_loss, config['loss'])
        self.criterion = getattr(module_loss, config['loss']["type"])
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
        self.epoch_metrics = MetricTracker(keys=['loss2d', 'loss3d'] + [str(m) for m in self.metric_ftns], writer=self.writer)
        self.eval_metrics = MetricTracker(keys=['loss2d', 'loss3d'] + [str(m) for m in self.metric_ftns], writer=self.writer)

        
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
        loss3d = None

        pbar = tqdm(total=len(self._train_loader) * self._train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for batch_idx, (history, future) in enumerate(self._train_loader):

            
            img_seq = history[0].float().to(self._device)
            history_pose2d_seq = history[1].float().to(self._device)
            history_root_seq = history[2].float().to(self._device)
            history_mask = history[3].float().to(self._device)
            history_pose3d_seq = history[4].to(self._device)
            history_trans_seq = history[5].to(self._device)
            
            future_pose2d_seq = future[0].float().to(self._device)
            future_root_seq = future[1].float().to(self._device)
            future_pose3d_seq = future[2].to(self._device)
            future_trans_seq = future[3].to(self._device)
            
            if self.use_root_relative:
                history_pose2d_seq = cvt_relative_pose(history_root_seq, history_pose2d_seq)
                history_pose3d_seq = cvt_relative_pose(history_trans_seq, history_pose3d_seq)
                
                future_pose2d_seq = cvt_relative_pose(future_root_seq, future_pose2d_seq)
                future_pose3d_seq = cvt_relative_pose(future_trans_seq, future_pose3d_seq)
        
            if self.use_pose_norm:
                history_pose2d_seq /= img_seq.shape[3]
                history_root_seq /= img_seq.shape[3]
                
                future_pose2d_seq /= img_seq.shape[3]
                future_root_seq /= img_seq.shape[3]
            
            self.optimizer.zero_grad()

            output2d, _ = self.model(img_seq, history_pose2d_seq, history_root_seq, history_mask)
            
            if self.use_root_relative:
                future_poses2d = torch.cat([future_root_seq.unsqueeze(2), future_pose2d_seq], dim=2)
                history_poses2d = torch.cat([history_root_seq.unsqueeze(2), history_pose2d_seq], dim=2)
                
                history_poses3d = torch.cat([history_trans_seq.unsqueeze(2), history_pose3d_seq], dim=2)  
                future_poses3d = torch.cat([future_trans_seq.unsqueeze(2), future_pose3d_seq], dim=2)  
                
            else:
                future_poses2d = future_pose2d_seq
                history_poses2d = history_pose2d_seq   
                history_poses3d = history_pose3d_seq  
                future_poses3d = future_poses3d  
                
            
            
            poses2d = torch.cat([history_poses2d, future_poses2d], dim=1) 
            loss_2d = self.criterion(output2d, future_poses2d)
            loss = loss_2d

            if self.use_projection:
                poses3d = torch.cat([history_poses3d, future_poses3d], dim=1) 
                
                output3d = self.model_proj(poses2d)
                loss_3d = self.criterion(output3d, poses3d)
                loss += loss_3d
                
                output3d = self.model_proj(output2d)
                self.epoch_metrics.update('loss3d', loss_3d.item())
            
            loss.backward()
            self.optimizer.step()
                            

            met2d = None
            met3d = None
            if self.writer is not None: self.writer.set_step((self.current_epoch - 1) * len(self._train_loader) + batch_idx)
            self.epoch_metrics.update('loss2d', loss_2d.item())
            for metric in self.metric_ftns:
                if str(metric) == "vim2d":
                    met2d = metric.compute(output2d, future_poses2d, self.use_pose_norm, self.use_root_relative)
                    self.epoch_metrics.update(str(metric), met2d.item())

                if str(metric) == "vim3d" and self.use_projection:
                    met3d = metric.compute(output3d, future_poses3d, False, self.use_root_relative) * 100 # convert to centimeters
                    self.epoch_metrics.update(str(metric), met3d.item())
            
            if self.use_projection:
                
                pbar.set_description(f"Train epoch: {self.current_epoch} loss2d: {loss_2d.item():.6f} loss3d: {loss_3d.item():.6f} vim2d: {met2d.item() if met2d is not None else None:.5f} vim3d: {met3d.item() if met3d is not None else None:.4f}")
            else:
                
                pbar.set_description(f"Train epoch: {self.current_epoch} loss2d: {loss_2d.item():.6f} vim2d: {met2d.item() if met2d is not None else None:.5f}")

            # if batch_idx % self.log_step == 0:
            #     # self.logger.debug('Train Epoch: {} Loss: {:.6f}'.format(self.current_epoch, loss.item()))

            #     ## Log to Tensorboard
            #     if self.writer is not None:
            #         self.writer.add_image('input_train', make_grid(history.cpu(), nrow=8, normalize=True))

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
        loss3d = None
        self.logger.debug(f"++> Evaluate at epoch {self.current_epoch} ...")

        pbar = tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx, (history, future) in enumerate(loader): 
            
            img_seq = history[0].float().to(self._device)
            history_pose2d_seq = history[1].float().to(self._device)
            history_root_seq = history[2].float().to(self._device)
            history_mask = history[3].float().to(self._device)
            history_pose3d_seq = history[4].to(self._device)
            history_trans_seq = history[5].to(self._device)
            
            future_pose2d_seq = future[0].float().to(self._device)
            future_root_seq = future[1].float().to(self._device)
            future_pose3d_seq = future[2].to(self._device)
            future_trans_seq = future[3].to(self._device)
            
            if self.use_root_relative:
                history_pose2d_seq = cvt_relative_pose(history_root_seq, history_pose2d_seq)
                history_pose3d_seq = cvt_relative_pose(history_trans_seq, history_pose3d_seq)
                
                future_pose2d_seq = cvt_relative_pose(future_root_seq, future_pose2d_seq)
                future_pose3d_seq = cvt_relative_pose(future_trans_seq, future_pose3d_seq)

            if self.use_pose_norm:
                history_pose2d_seq /= img_seq.shape[3]
                history_root_seq /= img_seq.shape[3]
                
                future_pose2d_seq /= img_seq.shape[3]
                future_root_seq /= img_seq.shape[3]
            
            output2d, _ = self.model(img_seq, history_pose2d_seq, history_root_seq, history_mask)
            
            if self.use_root_relative:
                future_poses2d = torch.cat([future_root_seq.unsqueeze(2), future_pose2d_seq], dim=2)
                history_poses2d = torch.cat([history_root_seq.unsqueeze(2), history_pose2d_seq], dim=2)
                history_poses3d = torch.cat([history_trans_seq.unsqueeze(2), history_pose3d_seq], dim=2)  
                future_poses3d = torch.cat([future_trans_seq.unsqueeze(2), future_pose3d_seq], dim=2)  
                
            else:
                future_poses2d = future_pose2d_seq
                history_poses2d = history_pose2d_seq   
                history_poses3d = history_pose3d_seq  
                future_poses3d = future_poses3d   
            
            poses2d = torch.cat([history_poses2d, future_poses2d], dim=1) 
            loss_2d = self.criterion(output2d, future_poses2d)

            
            if self.use_projection:
                poses3d = torch.cat([history_poses3d, future_poses3d], dim=1) 
                
                output3d = self.model_proj(poses2d)
                loss_3d = self.criterion(output3d, poses3d)
                
                output3d = self.model_proj(output2d)
                self.eval_metrics.update('loss3d', loss_3d.item())            

            met2d = None
            met3d = None
            if self.writer is not None: self.writer.set_step((self.current_epoch - 1) * len(self._train_loader) + batch_idx)
            self.eval_metrics.update('loss2d', loss_2d.item())
            for metric in self.metric_ftns:
                if str(metric) == "vim2d":
                    met2d = metric.compute(output2d, future_poses2d, self.use_pose_norm, self.use_root_relative)
                    self.eval_metrics.update(str(metric), met2d.item())

                if str(metric) == "vim3d" and self.use_projection:
                    met3d = metric.compute(output3d, future_poses3d, False, self.use_root_relative) * 100 # convert to centimeters
                    self.eval_metrics.update(str(metric), met3d.item())
            
            if self.use_projection:
                pbar.set_description(f"Eval epoch: {self.current_epoch} loss2d: {loss_2d.item():.6f} loss3d: {loss_3d.item():.6f if loss3d is not None else ''} vim2d: {met2d.item() if met2d is not None else None:.5f} vim3d: {met3d.item() if met3d is not None else None:.4f}")
            
            else:
                pbar.set_description(f"Eval epoch: {self.current_epoch} loss2d: {loss_2d.item():.6f} vim2d: {met2d.item() if met2d is not None else None:.5f}")
            # if self.writer is not None: self.writer.add_image('input_valid', make_grid(history.cpu(), nrow=8, normalize=True))
            pbar.update(loader.batch_size)
                
        # add histogram of model parameters to the tensorboard
        '''
        Uncommenting the next 3 lines with "tensorboard" set to true in vgg_cifar10_pretrained.json will make the tensorboard file large and training very slow. 
        Also, tensorboard is likely to crash because of VGG's large number of parameters.
        Better to use wandb.watch() for this purpose.
        '''
        # if self.writer is not None:
        #     for name, p in self.model.named_parameters():
        #         self.writer.add_histogram(name, p, bins='auto')

        pbar.close()
        self.logger.debug(f"++> Evaluate epoch {self.current_epoch} Finished.")
        
        return self.eval_metrics.result()