

import torch

from models.hppw.transforms import cvt_absolute_pose

class VIM:
    
    def __init__(self, name, img_size: int=224, is_inp_abs: bool=False, is_pose_norm: bool=False):
        self.img_size = img_size
        self.is_inp_abs = is_inp_abs
        self.name = name
        
    def compute(self, prediction, future):
        
        if is_pose_norm:
            future = future * self.img_size
            prediction = prediction * self.img_size
                
        if not self.is_inp_abs:
            root_joints_gt = future[..., 0, :]
            root_joints_pred = prediction[..., 0, :]
            
            relative_poses_gt = future[..., 1:, :]
            relative_poses_pred = prediction[..., 1:, :]
            
            gt = cvt_absolute_pose(root_joints=root_joints_gt, relative_poses=relative_poses_gt)
            pred = cvt_absolute_pose(root_joints=root_joints_pred, relative_poses=relative_poses_pred)
        
        else:
            pred = prediction
            gt = future
            
        sum_per_joint = torch.sum((gt - pred) ** 2, dim=-1)
        norm_per_joint = torch.sqrt(sum_per_joint)
        mean = torch.mean(norm_per_joint)
        
        return mean
    
    def __str__(self):
    
        return self.name
    
