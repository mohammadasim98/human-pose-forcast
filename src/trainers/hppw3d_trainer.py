import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm

from torchvision.utils import make_grid


import models.hppw as module_arch
import models.hppw as module_metric
import models.hppw as module_loss
from models.hppw.transforms import cvt_relative_pose, cvt_absolute_pose
from models.projection.model import LinearProjection
from models.embedding.dct import get_dct_matrix

from trainers.base3d import BaseTrainer

from utils.io import MetricTracker
from utils.viz import annotate_pose_2d, annotate_root_2d



def _generate_key_padding_mask(poses: torch.Tensor) -> torch.Tensor:
    mask = torch.where(poses==0.0, 1.0, 0.0)

    return torch.sum(mask, dim=-1).bool()

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
        self.use_dct = config["use_dct"]
        self.curriculum = config["curriculum"]
        self.qsample = config["qsample"]
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
        self.epoch_metrics = MetricTracker(keys=['loss2d'] + [str(m) for m in self.metric_ftns], writer=self.writer)
        self.eval_metrics = MetricTracker(keys=['loss2d'] + [str(m) for m in self.metric_ftns], writer=self.writer)
        
        self.history_window = self.curriculum["history_window"]
        self.max_history_window = self.curriculum["max_history_window"]
        self.future_window = self.curriculum["future_window"]
        self.max_future_window = self.curriculum["max_future_window"]
        
        self.qbatch_index = self.qsample["batch_index"]
        self.qsequence_index = self.qsample["sequence_index"]
        self.qperiod = self.qsample["period"]
        
    def send_imgs(self, history, future, output2d, qsequence_index, batch_idx):


        imgs = history[0][qsequence_index, -self.history_window:, ...].cpu().numpy()

        # (1, S, N, 2)
        hist_pose2d = history[1][qsequence_index, -self.history_window:, ...].unsqueeze(0).cpu().numpy()
        hist_root2d = history[2][qsequence_index, -self.history_window:, ...].unsqueeze(0).cpu().numpy()
        gt_pose2d = future[0][qsequence_index, :self.future_window, ...].unsqueeze(0).cpu().numpy()
        gt_root2d = future[1][qsequence_index, :self.future_window, ...].unsqueeze(0).cpu().numpy()

        root2d = None
        # (1, S, N, 2) or (S, N+1, 2)
        pred2d = output2d[qsequence_index, ...].unsqueeze(0)

        if self.use_pose_norm:
            pred2d *= imgs.shape[2]

        if self.use_root_relative:
            # (1, S, 2)
            root2d = pred2d[..., 0, :]
            # (1, S, N+1, 2)
            pred2d = cvt_absolute_pose(root2d, pred2d[..., 1:, :])


        if root2d is not None:
            root2d = root2d.cpu().numpy()
        pred2d = pred2d.cpu().numpy()
        # abs_pose = cvt_absolute_pose(root_joint=np.expand_dims(root_joint, 0), norm_pose=np.expand_dims(norm_pose, 0))

        # cv2.imshow("History Image", img)
        # cv2.imshow("History Mask", mask*255)

        imgs_list = []

        for j in range(hist_pose2d.shape[1]-1):
            annotated_img = annotate_pose_2d(img=imgs[j, ...], pose=hist_pose2d[:, j, ...], color=(255, 0, 0), radius=2, thickness=2, text=False)
            annotated_img = annotate_root_2d(img=annotated_img, root=hist_root2d[:, j, ...], color=(0, 0, 255), thickness=3)

            imgs_list.append(self.wandb.Image(annotated_img.astype(np.uint8)[..., ::-1]))



        for j in range(pred2d.shape[1]):
            curr_img = deepcopy(imgs[-1, ...])

            gt_abs_pose = gt_pose2d[:, j, ...]
            gt_root_joint = gt_root2d[:, j, ...]

            pred_abs = pred2d[:, j, ...].astype(np.uint8)

            # abs_pose = cvt_absolute_pose(root_joint=np.expand_dims(root_joint, 0), norm_pose=np.expand_dims(norm_pose, 0))
            annotated_img = annotate_pose_2d(img=curr_img, pose=pred_abs, color=(225, 225, 0), radius=2, thickness=2, text=False)
            if root2d is not None:
                pred_root = root2d[:, j, ...].astype(np.uint8)
                annotated_img = annotate_root_2d(img=annotated_img, root=pred_root, color=(255, 0, 255), thickness=3)

            annotated_img = annotate_pose_2d(img=annotated_img, pose=gt_abs_pose, color=(0, 255, 0), radius=2, thickness=2, text=False)
            annotated_img = annotate_root_2d(img=annotated_img,root=gt_root_joint, color=(0, 255, 255), thickness=3)
            # cv2.imwrite(ospj(folder_path, str(i)+".jpg"), annoted_img)

            imgs_list.append(self.wandb.Image(annotated_img.astype(np.uint8)[..., ::-1]))


        self.wandb.log({f"images_{qsequence_index}_{batch_idx}": imgs_list})
        
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


            img_seq = history[0][:, -self.history_window:, ...].float().to(self._device)
            history_pose2d_seq = history[1][:, -self.history_window:, ...].float().to(self._device)
            history_root_seq = history[2][:, -self.history_window:, ...].float().to(self._device)
            history_mask = history[3].float().to(self._device)
            history_pose3d_seq = history[4][:, -self.history_window:, ...].to(self._device)
            history_trans_seq = history[5][:, -self.history_window:, ...].to(self._device)
            
            history_pose2d_mask = _generate_key_padding_mask(history_pose2d_seq)
            history_root_mask = _generate_key_padding_mask(history_root_seq)
            
            future_pose2d_seq = future[0][:, :self.future_window, ...].float().to(self._device)
            future_root_seq = future[1][:, :self.future_window, ...].float().to(self._device)
            future_pose3d_seq = future[2][:, :self.future_window, ...].to(self._device)
            future_trans_seq = future[3][:, :self.future_window, ...].to(self._device)
            
            future_pose2d_mask = _generate_key_padding_mask(future_pose2d_seq)
            future_root_mask = _generate_key_padding_mask(future_root_seq)
            
            if self.use_pose_norm:
                history_pose2d_seq /= img_seq.shape[3]
                history_root_seq /= img_seq.shape[3]
                
                future_pose2d_seq /= img_seq.shape[3]
                future_root_seq /= img_seq.shape[3]
            
            if self.use_root_relative:
                history_pose2d_seq = cvt_relative_pose(history_root_seq, history_pose2d_seq)
                history_pose3d_seq = cvt_relative_pose(history_trans_seq, history_pose3d_seq)
                
                future_pose2d_seq = cvt_relative_pose(future_root_seq, future_pose2d_seq)
                future_pose3d_seq = cvt_relative_pose(future_trans_seq, future_pose3d_seq)
                
                future_poses2d = torch.cat([future_root_seq.unsqueeze(2), future_pose2d_seq], dim=2)
                history_poses2d = torch.cat([history_root_seq.unsqueeze(2), history_pose2d_seq], dim=2)
                
                history_poses3d = torch.cat([history_trans_seq.unsqueeze(2), history_pose3d_seq], dim=2)  
                future_poses3d = torch.cat([future_trans_seq.unsqueeze(2), future_pose3d_seq], dim=2)  
                
                history_pose_mask =  torch.cat([history_root_mask.unsqueeze(-1), history_pose2d_mask], dim=-1)
                future_pose_mask =  torch.cat([future_root_mask.unsqueeze(-1), future_pose2d_mask], dim=-1)

                
            else:
                future_poses2d = future_pose2d_seq
                history_poses2d = history_pose2d_seq   
                history_poses3d = history_pose3d_seq  
                future_poses3d = future_poses3d  
                history_pose_mask = history_pose2d_mask
                future_pose_mask = future_pose2d_mask
                
            self.optimizer.zero_grad()
            
            
#             if self.use_dct:
#                 dct_m, idct_m = get_dct_matrix(history_poses2d.shape[1])
                
#                 history_poses2d = torch.matmul(dct_m[:dct_n, :], history_poses2d.permute())
            output2d, _ = self.model(img_seq, history_pose2d_seq, history_root_seq, history_mask, history_pose_mask, self.future_window)                

            poses2d = torch.cat([history_poses2d, future_poses2d], dim=1) 
            loss_2d = self.criterion(output2d, future_poses2d, future_pose_mask)
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
    def evaluate(self, loader=None, history_window=None, future_window=None):
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
        
        if history_window is not None:
            self.history_window = history_window
            
        if future_window is not None:
            self.future_window = future_window
            
        for batch_idx, (history, future) in enumerate(loader): 


            img_seq = history[0][:, -self.history_window:, ...].float().to(self._device)
            history_pose2d_seq = history[1][:, -self.history_window:, ...].float().to(self._device)
            history_root_seq = history[2][:, -self.history_window:, ...].float().to(self._device)
            history_mask = history[3].float().to(self._device)
            history_pose3d_seq = history[4][:, -self.history_window:, ...].to(self._device)
            history_trans_seq = history[5][:, -self.history_window:, ...].to(self._device)
            
            history_pose2d_mask = _generate_key_padding_mask(history_pose2d_seq)
            history_root_mask = _generate_key_padding_mask(history_root_seq)
            
            future_pose2d_seq = future[0][:, :self.future_window, ...].float().to(self._device)
            future_root_seq = future[1][:, :self.future_window, ...].float().to(self._device)
            future_pose3d_seq = future[2][:, :self.future_window, ...].to(self._device)
            future_trans_seq = future[3][:, :self.future_window, ...].to(self._device)
            
            future_pose2d_mask = _generate_key_padding_mask(future_pose2d_seq)
            future_root_mask = _generate_key_padding_mask(future_root_seq)
            
            if self.use_pose_norm:
                history_pose2d_seq /= img_seq.shape[3]
                history_root_seq /= img_seq.shape[3]
                
                future_pose2d_seq /= img_seq.shape[3]
                future_root_seq /= img_seq.shape[3]
            
            if self.use_root_relative:
                history_pose2d_seq = cvt_relative_pose(history_root_seq, history_pose2d_seq)
                history_pose3d_seq = cvt_relative_pose(history_trans_seq, history_pose3d_seq)
                
                future_pose2d_seq = cvt_relative_pose(future_root_seq, future_pose2d_seq)
                future_pose3d_seq = cvt_relative_pose(future_trans_seq, future_pose3d_seq)
                
                future_poses2d = torch.cat([future_root_seq.unsqueeze(2), future_pose2d_seq], dim=2)
                history_poses2d = torch.cat([history_root_seq.unsqueeze(2), history_pose2d_seq], dim=2)
                
                history_poses3d = torch.cat([history_trans_seq.unsqueeze(2), history_pose3d_seq], dim=2)  
                future_poses3d = torch.cat([future_trans_seq.unsqueeze(2), future_pose3d_seq], dim=2)  
                
                history_pose_mask =  torch.cat([history_root_mask.unsqueeze(-1), history_pose2d_mask], dim=-1)
                future_pose_mask =  torch.cat([future_root_mask.unsqueeze(-1), future_pose2d_mask], dim=-1)

                
            else:
                future_poses2d = future_pose2d_seq
                history_poses2d = history_pose2d_seq   
                history_poses3d = history_pose3d_seq  
                future_poses3d = future_poses3d  
                history_pose_mask = history_pose2d_mask
                future_pose_mask = future_pose2d_mask
            
            output2d, _ = self.model(img_seq, history_pose2d_seq, history_root_seq, history_mask, history_pose_mask, self.future_window)                
            poses2d = torch.cat([history_poses2d, future_poses2d], dim=1) 
            loss_2d = self.criterion(output2d, future_poses2d, future_pose_mask)

            
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
            
            if self.wandb_enabled:

                if self.current_epoch % self.qperiod == 0:
                    for qsequence_index in self.qsequence_index:
                        for index in self.qbatch_index:
                            if index == batch_idx:
                                self.send_imgs(history, future, output2d, qsequence_index, index)
                
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