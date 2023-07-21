

import torch

from models.hppw.transforms import cvt_absolute_pose

class VIM:
    
    def __init__(self, name, img_size: int=224):
        self.img_size = img_size
        self.name = name
        
    def compute(self, prediction, future, is_pose_norm: bool=False, is_root_relative: bool=False):
                
        if is_root_relative:
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
        
        if is_pose_norm:
            mean *= self.img_size
        
        return mean
    
    def __str__(self):
    
        return self.name
    
