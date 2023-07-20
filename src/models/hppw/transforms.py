import torch

def cvt_absolute_pose(root_joints: torch.Tensor, relative_poses: torch.Tensor):
    
    root_joints = root_joints.unsqueeze(2)
    abs_poses = relative_poses + torch.tile(root_joints, (1, 1, relative_poses.shape[2], 1))
    return  abs_poses

def cvt_relative_pose(root_joints: torch.Tensor, abs_poses: torch.Tensor):
    
    root_joints = root_joints.unsqueeze(2)
    relative_poses = abs_poses - torch.tile(root_joints, (1, 1, relative_poses.shape[2], 1))
    return relative_poses