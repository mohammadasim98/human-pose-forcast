import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm


import matplotlib.pyplot as plt

from models.hppw.transforms import cvt_relative_pose, cvt_absolute_pose
from models.embedding.dct import get_dct_matrix

from trainers.basev2 import BaseTrainer

from utils.viz import annotate_pose_2d, annotate_root_2d



def _generate_key_padding_mask(poses: torch.Tensor) -> torch.Tensor:
    mask = torch.where(poses==0.0, 1.0, 0.0)

    return torch.sum(mask, dim=-1).bool()

class HPPW3DTrainerV2(BaseTrainer):

    def __init__(self, config, train_loader, eval_loader=None):
        """
        Create the model, loss criterion, optimizer, and dataloaders
        And anything else that might be needed during training. (e.g. device type)
        """
        super().__init__(config)    
        # build model architecture, then print to console
        
        self.use_root_relative = config["use_root_relative"]
        self.use_pose_norm = config["use_pose_norm"]
        self.use_projection = config["use_projection"]
        self.use_dct = config["use_dct"]
        self.dct_config = config["dct_config"]
        self.curriculum = config["curriculum"]
        self.qsample = config["qsample"]
        self.is_teacher_forcing = config["is_teacher_forcing"]
        # Set DataLoaders
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        
        self.log_step = 100 # arbitrary
        # Prepare Metrics
        
        
        self.history_window = self.curriculum["history_window"]
        self.max_history_window = self.curriculum["max_history_window"]
        self.future_window = self.curriculum["future_window"]
        self.max_future_window = self.curriculum["max_future_window"]
        
        self.qbatch_index = self.qsample["batch_index"]
        self.qsequence_index = self.qsample["sequence_index"]
        self.qperiod = self.qsample["period"]
        self.hdct_n = self.dct_config["hdct_n"]
        self.fdct_n = self.dct_config["fdct_n"]
        self.count = 0
        self.kld_weight = 0.001
    
    def gauss(self, x, mu, sigma):
        
        return np.exp(-(x-mu)**2 / (2*(sigma**2))) / (sigma*np.sqrt(2*np.pi))
    

    
    def get_imgs(self, history, future, output2d, qsequence_index, batch_idx, name, final_mus, final_sigmas, weights=None):


        imgs = history[0][qsequence_index, -self.history_window:, ...].cpu().numpy()
        

        
        # (1, S, N, 2)
        hist_pose2d = history[1][qsequence_index, -self.history_window:, :14, :].unsqueeze(0).cpu().numpy()
        hist_root2d = history[2][qsequence_index, -self.history_window:, ...].unsqueeze(0).cpu().numpy()
        gt_pose2d = future[0][qsequence_index, :self.future_window, :14, :].unsqueeze(0).cpu().numpy()
        gt_root2d = future[1][qsequence_index, :self.future_window, ...].unsqueeze(0).cpu().numpy()
        mus = final_mus[qsequence_index].cpu()
        sigmas = final_sigmas[qsequence_index].cpu()
        # gt_pose2d = np.concatenate([hist_pose2d, gt_pose2d], axis=1)
        # gt_root2d = np.concatenate([hist_root2d, gt_root2d], axis=1)
        
        root2d = None
        # (1, S, N, 2) or (S, N+1, 2)
        pred2d = output2d[qsequence_index, ...].unsqueeze(0)
        
        prob = torch.zeros((mus.shape[0], imgs.shape[-3], imgs.shape[-2], 1))
        eps = torch.randn(mus.shape[0], 1000, 2)
        values = mus[:, 13:14, :] + eps*torch.exp(0.5 * sigmas[:, 13:14, :])
        values = cvt_absolute_pose(pred2d[..., 0, :].cpu(), values.unsqueeze(0))
        values *= imgs.shape[-3]
        values = values.to(int)

        values = torch.clip(values, 0, 223)
        for mp in range(mus.shape[0]):
            
            prob[mp, values[0, mp, :, 1], values[0, mp, :, 0]] = 255

        
        maps = prob.to(torch.uint8).numpy()

#         prob_x = np.arange(imgs.shape[-3] - 1, -1, -1) / (imgs.shape[-3] - 1)
#         prob_x = np.expand_dims(prob_x, -1)
#         prob_x = np.tile(prob_x, (1, imgs.shape[-2]))
        
        
#         prob_y = np.arange(imgs.shape[-2] - 1, -1, -1) / (imgs.shape[-2] - 1)
#         prob_y = np.expand_dims(prob_y, 0)
#         prob_y = np.tile(prob_y, (imgs.shape[-3], 1))
#         maps = []

#         for mp in range(mus.shape[0]):
#             root_x = pred2d[0, mp, 0, 0].cpu().numpy()
#             root_y = pred2d[0, mp, 0, 1].cpu().numpy()
#             prob = self.gauss(prob_x - root_x, mus[mp, 13, 0], np.exp(0.5 * sigmas[mp, 13, 0])) * self.gauss(prob_y - root_y, mus[mp, 13, 1], np.exp(0.5 * sigmas[mp, 13, 1]))
#             prob = (prob * 255).astype(np.uint8)
#             maps.append(np.expand_dims(np.expand_dims(prob, 0), -1))
        
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

            
        for j in range(hist_pose2d.shape[1]):
            annotated_img = annotate_pose_2d(img=imgs[j, ...], pose=hist_pose2d[:, j, ...], color=(255, 0, 0), radius=2, thickness=2, text=False)
            annotated_img = annotate_root_2d(img=annotated_img, root=hist_root2d[:, j, ...], color=(0, 0, 255), thickness=3)

            # imgs_list.append(self.wandb.Image(annotated_img.astype(np.uint8)[..., ::-1]))
            imgs_list.append(np.expand_dims(annotated_img.astype(np.uint8)[..., ::-1], 0))
        
        
        

        
        weights_list = []
        for j in range(pred2d.shape[1] ):

            curr_img = deepcopy(imgs[-1, ...])
            gt_abs_pose = gt_pose2d[:, j, ...]
            gt_root_joint = gt_root2d[:, j, ...]

            pred_abs = pred2d[:, j, ...].astype(np.uint8)
            
            annotated_img = annotate_pose_2d(img=curr_img, pose=gt_abs_pose, color=(0, 255, 0), radius=2, thickness=2, text=False)
            annotated_img = annotate_root_2d(img=annotated_img,root=gt_root_joint, color=(0, 255, 255), thickness=3)
            
            
            
            annotated_img = annotate_pose_2d(img=annotated_img, pose=pred_abs, color=(0, 0, 255), radius=2, thickness=2, text=False)
            if root2d is not None:
                pred_root = root2d[:, j, ...].astype(np.uint8)
                annotated_img = annotate_root_2d(img=annotated_img, root=pred_root, color=(255, 0, 255), thickness=3)
            # cv2.imwrite(ospj(folder_path, str(i)+".jpg"), annoted_img)

            # imgs_list.append(self.wandb.Image(annotated_img.astype(np.uint8)[..., ::-1]))
            imgs_list.append(np.expand_dims(annotated_img.astype(np.uint8)[..., ::-1], 0))
            
        for j in range(len(weights[1])):
            if weights is not None:

                # weights_list.append(self.wandb.Image((weights[2][j][qsequence_index, 10, 1:].cpu().numpy().reshape(14, 14) * 255).astype(np.uint8)))
                # weights_list.append(self.wandb.Image((weights[2][j][qsequence_index, 13, 1:].cpu().numpy().reshape(14, 14) * 255).astype(np.uint8)))
                if weights[1][j].requires_grad:
                    weights_list.append((weights[1][j][qsequence_index, ...].unsqueeze(0).unsqueeze(-1).cpu().detach().numpy() * 255).astype(np.uint8))
                    weights_list.append((weights[1][j][qsequence_index, ...].unsqueeze(0).unsqueeze(-1).cpu().detach().numpy() * 255).astype(np.uint8))
                else:

                    weights_list.append((weights[1][j][qsequence_index, ...].unsqueeze(0).unsqueeze(-1).cpu().numpy() * 255).astype(np.uint8))
                    weights_list.append((weights[1][j][qsequence_index, ...].unsqueeze(0).unsqueeze(-1).cpu().numpy() * 255).astype(np.uint8))
        # print((weights[2][0][qsequence_index, 0, 1:].cpu().detach().numpy().reshape(14, 14) * 255).astype(np.uint8))            
        # # print(weights[2][0][qsequence_index, 10, 1:].cpu().detach().numpy().reshape(14, 14))            
        # plt.imshow((weights[2][0][qsequence_index, 0, 1:].cpu().detach().numpy().reshape(14, 14)), cmap="gray")
        # self.wandb.log({f"{name}-images_{qsequence_index}_{batch_idx}": imgs_list})
        i = self.wandb.Video(np.transpose(np.concatenate(imgs_list, axis=0), (0, 3, 1, 2)), fps=4, format="gif")
        # m = self.wandb.Video(np.transpose(np.concatenate(maps, 0), (0, 3, 1, 2)), fps=4, format="gif")
        m = self.wandb.Video(np.transpose(maps, (0, 3, 1, 2)), fps=4, format="gif")
        w = None
        if weights is not None:
            w = self.wandb.Video(np.transpose(np.concatenate(weights_list, axis=0), (0, 3, 1, 2)), fps=4, format="gif")
            # self.wandb.log({f"{name}-attentions_{qsequence_index}_{batch_idx}": weights_list})
        return i, w, m

    def kld(self, mu, log_var):
        
        return torch.mean(torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1), dim=-1))
        
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
        
        hdct_n = self.hdct_n if self.hdct_n <= self.history_window else self.history_window
        fdct_n = self.fdct_n if self.fdct_n <= self.future_window else self.future_window

        is_teacher_forcing = self.is_teacher_forcing
            
#         if self.is_teacher_forcing and self.current_epoch <= 25:
#             is_teacher_forcing = True
        
#         if self.is_teacher_forcing and self.current_epoch >= 25:
#             is_teacher_forcing = False
        
        self.logger.debug(f"Teacher Forcing: {is_teacher_forcing} {self.count}")
        running_loss = 0
        running_vim = 0
        pbar = tqdm(total=len(self._train_loader) * self._train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        img_list = []
        weight_list = []
        maps = []
        for batch_idx, (history, future) in enumerate(self._train_loader):

            img_seq = history[0][:, -self.history_window:, ...].float().to(self._device)
            history_pose2d_seq = history[1][:, -self.history_window:, :14, :].float().to(self._device)
            history_root_seq = history[2][:, -self.history_window:, ...].float().to(self._device)
            history_mask = history[3].float().to(self._device)
            history_pose3d_seq = history[4][:, -self.history_window:, ...].to(self._device)
            history_trans_seq = history[5][:, -self.history_window:, ...].to(self._device)
            
            history_pose2d_mask = _generate_key_padding_mask(history_pose2d_seq)
            history_root_mask = _generate_key_padding_mask(history_root_seq)
            
            future_pose2d_seq = future[0][:, :self.future_window, :14, :].float().to(self._device)
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
                future_poses3d = future_pose3d_seq  
                history_pose_mask = history_pose2d_mask
                future_pose_mask = future_pose2d_mask
            
            if self.use_dct:
                dct_m, idct_m = get_dct_matrix(self.history_window)
                
                dct_m = dct_m.to(self._device)
                idct_m = idct_m.to(self._device)
                
                b, s, n, e = history_poses2d.shape
                
                history_poses2d = history_poses2d.permute(1, 0, 2, 3).contiguous().view(self.history_window, -1)
                history_root_seq = history_root_seq.permute(1, 0, 2).contiguous().view(self.history_window, -1)
                
                history_poses2d = torch.matmul(dct_m[:hdct_n, :], history_poses2d)
                history_root_seq = torch.matmul(dct_m[:hdct_n, :], history_root_seq)
                
                history_poses2d = history_poses2d.view(self.history_window, b, n, e).permute(1, 0, 2, 3).contiguous()
                history_root_seq = history_root_seq.view(self.history_window, b, e).permute(1, 0, 2).contiguous()
                
            
            self.optimizer.zero_grad()
            
#             if self.use_dct:
#                 dct_m, idct_m = get_dct_matrix(history_poses2d.shape[1])
                
#                 history_poses2d = torch.matmul(dct_m[:dct_n, :], history_poses2d.permute())

            output2d, proj_history_poses, mus, sigmas, final_mus, final_sigmas, weights = self.model(
                img_seq=img_seq, 
                history_pose=history_poses2d, 
                img_mask=history_mask, 
                history_pose_mask=history_pose_mask,
                future_window=self.future_window, 
                history_window=self.history_window,
                is_teacher_forcing=is_teacher_forcing, 
                future_pose=future_poses2d, 
                future_pose_mask=future_pose_mask
            )                
            

            if self.use_dct:
                dct_m, idct_m = get_dct_matrix(self.future_window)
                
                dct_m = dct_m.to(self._device)
                idct_m = idct_m.to(self._device)
                
                b, s, n, e = output2d.shape
                
                out_poses2d = output2d[..., 1:, :].permute(1, 0, 2, 3).contiguous().view(self.future_window, -1)
                out_root_seq = output2d[..., 0, :].permute(1, 0, 2).contiguous().view(self.future_window, -1)
                
                out_poses2d = torch.matmul(idct_m[:fdct_n, :], out_poses2d)
                out_root_seq = torch.matmul(idct_m[:fdct_n, :], out_root_seq)
                
                out_poses2d = out_poses2d.view(self.future_window, b, n-1, e).permute(1, 0, 2, 3).contiguous()
                out_root_seq = out_root_seq.view(self.future_window, b, e).permute(1, 0, 2).contiguous()
                
                output2d = torch.cat([out_root_seq.unsqueeze(2), out_poses2d], dim=2)
                
            pose2d_mask = torch.cat([history_pose_mask, future_pose_mask], dim=1)
            poses2d = torch.cat([history_poses2d, future_poses2d], dim=1) 
            loss_2d = self.criterion(output2d, future_poses2d, future_pose_mask)
            kld_loss = self.kld(mus, sigmas)
            loss_2d += self.kld_weight * kld_loss
            # loss_2d_history_proj = self.criterion(proj_history_poses, history_poses2d, history_pose_mask)
            # loss_2d_history_proj = 0
            
            loss = loss_2d 

            if self.use_projection:
                poses3d = torch.cat([history_poses3d, future_poses3d], dim=1) 
                
                self.model_proj.zero_grad()
                
                output3d = self.model_proj(poses2d)
                loss_3d = self.criterion(output3d, poses3d)
                loss += loss_3d
                
                output3d = self.model_proj(output2d)
                self.epoch_metrics.update('loss3d', loss_3d.item())
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            met2d = None
            met3d = None
            if self.writer is not None: self.writer.set_step((self.current_epoch - 1) * len(self._train_loader) + batch_idx)
            self.epoch_metrics.update('loss2d', loss_2d.item())
            for metric in self.metric_ftns:
                if str(metric) == "vim2d":
                    met2d = metric.compute(output2d, future_poses2d, self.use_pose_norm, self.use_root_relative)
                    running_vim += met2d
                    self.epoch_metrics.update(str(metric), met2d.item())

                if str(metric) == "vim3d" and self.use_projection:
                    met3d = metric.compute(output3d, future_poses3d, False, self.use_root_relative) * 100 # convert to centimeters
                    self.epoch_metrics.update(str(metric), met3d.item())
            
            if self.use_projection:
                
                pbar.set_description(f"Train epoch: {self.current_epoch} loss2d: {running_loss/(batch_idx+1):.6f} loss3d: {loss_3d.item():.6f} vim2d: {met2d.item() if met2d is not None else None:.5f} vim3d: {met3d.item() if met3d is not None else None:.4f}")
            else:
                
                pbar.set_description(f"Train epoch: {self.current_epoch} loss2d: {running_loss/(batch_idx+1):.6f} kld: {kld_loss.item():.6f} vim2d: {running_vim/(batch_idx+1):.6f}")

            # if batch_idx % self.log_step == 0:
            #     # self.logger.debug('Train Epoch: {} Loss: {:.6f}'.format(self.current_epoch, loss.item()))

            #     ## Log to Tensorboard
            #     if self.writer is not None:
            #         self.writer.add_image('input_train', make_grid(history.cpu(), nrow=8, normalize=True))

            pbar.update(self._train_loader.batch_size)
            
            if self.wandb_enabled:
                
                if self.current_epoch % 3 == 0:
                    for qsequence_index in self.qsequence_index:
                        for index in self.qbatch_index:
                            if index == batch_idx:
                                img, weight, mp = self.get_imgs(history, future, output2d.detach(), qsequence_index, index, "train", final_mus.detach(), final_sigmas.detach(), weights)
                                img_list.append(img)
                                maps.append(mp)
                                if weight is not None:
                                    weight_list.append(weight)
        if self.wandb_enabled:

            if len(img_list):
                self.wandb.log({"train_img": img_list})

            if len(weight_list):
                self.wandb.log({"train_atten_weights": weight_list})
            
            if len(maps):
                self.wandb.log({"train_maps": maps})

        log_dict = self.epoch_metrics.result()
        pbar.close()
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
        running_loss = 0
        running_vim = 0

        self.model.eval()
        self.eval_metrics.reset()
        loss3d = None
        self.logger.debug(f"++> Evaluate at epoch {self.current_epoch} ...")

        pbar = tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        if history_window is not None:
            self.history_window = history_window
            
        if future_window is not None:
            self.future_window = future_window
        
        
        hdct_n = self.hdct_n if self.hdct_n <= self.history_window else self.history_window
        fdct_n = self.fdct_n if self.fdct_n <= self.future_window else self.future_window
        img_list = []
        weight_list = []
        maps = []
        for batch_idx, (history, future) in enumerate(loader): 


            img_seq = history[0][:, -self.history_window:, ...].float().to(self._device)
            history_pose2d_seq = history[1][:, -self.history_window:, :14, :].float().to(self._device)
            history_root_seq = history[2][:, -self.history_window:, ...].float().to(self._device)
            history_mask = history[3].float().to(self._device)
            history_pose3d_seq = history[4][:, -self.history_window:, ...].to(self._device)
            history_trans_seq = history[5][:, -self.history_window:, ...].to(self._device)
            
            history_pose2d_mask = _generate_key_padding_mask(history_pose2d_seq)
            history_root_mask = _generate_key_padding_mask(history_root_seq)
            
            future_pose2d_seq = future[0][:, :self.future_window, :14, :].float().to(self._device)
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
                future_poses3d = future_pose3d_seq  
                history_pose_mask = history_pose2d_mask
                future_pose_mask = future_pose2d_mask
            
            if self.use_dct:
                dct_m, idct_m = get_dct_matrix(self.history_window)
                
                dct_m = dct_m.to(self._device)
                idct_m = idct_m.to(self._device)
                
                b, s, n, e = history_poses2d.shape
                
                history_poses2d = history_poses2d.permute(1, 0, 2, 3).contiguous().view(self.history_window, -1)
                history_root_seq = history_root_seq.permute(1, 0, 2).contiguous().view(self.history_window, -1)
                
                history_poses2d = torch.matmul(dct_m[:hdct_n, :], history_poses2d)
                history_root_seq = torch.matmul(dct_m[:hdct_n, :], history_root_seq)
                
                history_poses2d = history_poses2d.view(self.history_window, b, n, e).permute(1, 0, 2, 3).contiguous()
                history_root_seq = history_root_seq.view(self.history_window, b, e).permute(1, 0, 2).contiguous()

            
#             if self.use_dct:
#                 dct_m, idct_m = get_dct_matrix(history_poses2d.shape[1])
                
#                 history_poses2d = torch.matmul(dct_m[:dct_n, :], history_poses2d.permute())

            output2d, proj_history_poses, mus, sigmas, final_mus, final_sigmas, weights = self.model(
                img_seq=img_seq, 
                history_pose=history_poses2d, 
                img_mask=history_mask, 
                history_pose_mask=history_pose_mask,
                future_window=self.future_window, 
                history_window=self.history_window,
                is_teacher_forcing=False, 
                future_pose=None, 
                future_pose_mask=None
            )                
            
            if self.use_dct:
                dct_m, idct_m = get_dct_matrix(self.future_window)
                
                dct_m = dct_m.to(self._device)
                idct_m = idct_m.to(self._device)
                
                b, s, n, e = output2d.shape
                
                out_poses2d = output2d[..., 1:, :].permute(1, 0, 2, 3).contiguous().view(self.future_window, -1)
                out_root_seq = output2d[..., 0, :].permute(1, 0, 2).contiguous().view(self.future_window, -1)
                
                out_poses2d = torch.matmul(idct_m[:fdct_n, :], out_poses2d)
                out_root_seq = torch.matmul(idct_m[:fdct_n, :], out_root_seq)
                
                out_poses2d = out_poses2d.view(self.future_window, b, n-1, e).permute(1, 0, 2, 3).contiguous()
                out_root_seq = out_root_seq.view(self.future_window, b, e).permute(1, 0, 2).contiguous()
                
                output2d = torch.cat([out_root_seq.unsqueeze(2), out_poses2d], dim=2)
                
            pose2d_mask = torch.cat([history_pose_mask, future_pose_mask], dim=1)
            poses2d = torch.cat([history_poses2d, future_poses2d], dim=1) 
            loss_2d = self.criterion(output2d, future_poses2d, future_pose_mask)
            # loss_2d_history_proj = self.criterion(proj_history_poses, history_poses2d, history_pose_mask)
            # loss_2d_history_proj = 0
            kld_loss = self.kld(mus, sigmas)
            loss_2d += self.kld_weight * kld_loss
            loss = loss_2d
            running_loss += loss.item()
            
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
                    running_vim += met2d
                    self.eval_metrics.update(str(metric), met2d.item())

                if str(metric) == "vim3d" and self.use_projection:
                    met3d = metric.compute(output3d, future_poses3d, False, self.use_root_relative) * 100 # convert to centimeters
                    self.eval_metrics.update(str(metric), met3d.item())
            
            if self.use_projection:
                pbar.set_description(f"Eval epoch: {self.current_epoch} loss2d: {loss_2d.item():.6f} loss3d: {loss_3d.item():.6f if loss3d is not None else ''} vim2d: {met2d.item() if met2d is not None else None:.5f} vim3d: {met3d.item() if met3d is not None else None:.4f}")
            
            else:
                pbar.set_description(f"Eval epoch: {self.current_epoch} loss2d: {running_loss/(batch_idx+1):.6f} kld: {kld_loss.item():.6f} vim2d: {running_vim/(batch_idx+1):.6f}")
            # if self.writer is not None: self.writer.add_image('input_valid', make_grid(history.cpu(), nrow=8, normalize=True))
            pbar.update(loader.batch_size)
            
            if self.wandb_enabled:

                if self.current_epoch % self.qperiod == 0:
                    for qsequence_index in self.qsequence_index:
                        for index in self.qbatch_index:
                            if index == batch_idx:
                                img, weight, mp = self.get_imgs(history, future, output2d, qsequence_index, index, "val", final_mus, final_sigmas, weights)
                                img_list.append(img)
                                maps.append(mp)
                                if weight is not None:
                                    weight_list.append(weight)
        if self.wandb_enabled:

            if len(img_list):
                self.wandb.log({"val_img": img_list})

            if len(weight_list):
                self.wandb.log({"val_atten_weights": weight_list})
            
            if len(maps):
                self.wandb.log({"val_maps": maps})
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