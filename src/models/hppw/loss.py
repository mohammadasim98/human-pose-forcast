
import torch
from typing import Union
from models.hppw.transforms import cvt_absolute_pose

# def cvt_root_relative(root_joint: torch.Tensor, pose: torch.Tensor, eps: float=1e-3):
#     """Convert absolute pose to root relative pose given a root joint.
#                               (abs_pose - root)
#              norm_pose = ---------------------------
#                          (abs_pose + root + epsilon)
#     Args:
#         root_joint (numpy.ndarray): A (np, 3) numpy array with np as number of people.
#         pose (np.ndarray): A (np, 18, 3) numpy array with np as number of people.
#         eps (float, optional): A floating value for numerical stability. Defaults to 0.0001.

#     Returns:
#         numpy.ndarray: A (np, 18, 3) root relative pose with np number of people.
#     """
#     root_joint = root_joint.unsqueeze(2)
#     root_joint = torch.tile(root_joint, (1, 1, pose.shape[2], 1))
#     return (pose - root_joint) / (root_joint+pose+eps)

def mpjpe(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    sum_per_joint = torch.sum((future - pred) ** 2, dim=-1)

    if mask is not None:
        # (batch, seq, num_joint+1)
        inv_mask = ~mask
        inv_mask = inv_mask.float()
        # (batch, seq)
        denom = torch.sum(inv_mask, -1)
        # (batch, seq, num_joint+1) -> # (batch, seq, num_joint+1)
        norm_per_joint = torch.sqrt(sum_per_joint) * inv_mask
        
        # (batch, seq, num_joint+1) -> # (batch, seq)
        per_pose_mean = torch.sum(norm_per_joint, dim=-1) / (denom + 1e-6)
    
    else:
        norm_per_joint = torch.sqrt(sum_per_joint)
        per_pose_mean = torch.mean(norm_per_joint, dim=-1)
        
    mean = torch.mean(per_pose_mean)

    return mean

def mpjpev2(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    sum_per_joint = torch.sum((future - pred) ** 2, dim=-1)

    if mask is not None:
        # (batch, seq, num_joint+1)
        inv_mask = ~mask
        inv_mask = inv_mask.float()
        # (batch, seq)
        denom = torch.sum(inv_mask, -1)
        # (batch, seq, num_joint+1) -> # (batch, seq, num_joint+1)
        norm_per_joint = sum_per_joint * inv_mask
        
        # (batch, seq, num_joint+1) -> # (batch, seq)
        per_pose_mean = torch.sum(norm_per_joint, dim=-1) / (denom + 1e-6)
    
    else:
        norm_per_joint = sum_per_joint
        per_pose_mean = torch.mean(norm_per_joint, dim=-1)
        
    mean = torch.mean(per_pose_mean)

    return mean

def mpjpev3(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
    root_pred = pred[..., 0, :]
    root_gt = future[..., 0, :]
    pose_pred = pred[..., 1:, :]
    pose_gt = future[..., 1:, :]
    
    pose_norm = torch.sum((pose_gt - pose_pred) ** 2, dim=-1)
    root_norm = torch.sum((root_gt - root_pred) ** 2, dim=-1)

    
    if mask is not None:
        # (batch, seq, num_joint+1)
        pose_mask = mask[..., 1:]
        root_mask = mask[..., 0]
        
        inv_pose_mask = ~pose_mask
        inv_root_mask = ~root_mask
        
        inv_pose_mask = inv_pose_mask.float()
        inv_root_mask = inv_root_mask.float()
        
        # (batch, seq)
        denom_pose = torch.sum(inv_pose_mask, -1)
        
        # (batch, seq, num_joint+1) -> # (batch, seq, num_joint+1)
        pose_norm *= inv_pose_mask
        root_norm *= inv_root_mask
        
        # (batch, seq, num_joint+1) -> # (batch, seq)
        per_pose_mean = torch.sum(pose_norm, dim=-1) / (denom_pose + 1e-6)
        
    else:
        per_pose_mean = torch.mean(pose_norm, dim=-1)
        
    mean = torch.mean(per_pose_mean + root_norm)

    return mean

def mpjpev4(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
    root_pred = pred[..., 0, :]
    root_gt = future[..., 0, :]
    pose_pred = pred[..., 1:, :]
    pose_gt = future[..., 1:, :]
    

    
    if mask is not None:
        # (batch, seq, num_joint+1)
        pose_mask = mask[..., 1:]
        root_mask = mask[..., 0]
        
        inv_pose_mask = ~pose_mask
        inv_root_mask = ~root_mask
        
        inv_pose_mask = inv_pose_mask.float()
        inv_root_mask = inv_root_mask.float()
        
        # (batch, seq)
        denom_pose = torch.sum(inv_pose_mask, -1)
        
        pose_norm = torch.sum((pose_gt - pose_pred) ** 2, dim=-1)
        root_norm = torch.sum((root_gt - root_pred) ** 2, dim=-1)
        
        # (batch, seq, num_joint+1) -> # (batch, seq, num_joint+1)
        pose_norm *= inv_pose_mask
        root_norm *= inv_root_mask
        
        # (batch, seq, num_joint+1) -> # (batch, seq)
        per_pose = torch.sqrt(torch.sum(pose_norm, dim=-1))
        per_root = torch.sqrt(root_norm)
        
    else:
        pose_norm = torch.sum((pose_gt - pose_pred) ** 2, dim=-1)
        root_norm = torch.sum((root_gt - root_pred) ** 2, dim=-1)
        
        per_pose = torch.sqrt(torch.sum(pose_norm, dim=-1))
        per_root = torch.sqrt(root_norm)

    mean = torch.mean(per_pose + per_root)

    return mean

    
    
    
def mpjpev5(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
    root_pred = pred[..., 0, :]
    root_gt = future[..., 0, :]
    pose_pred = pred[..., 1:, :]
    pose_gt = future[..., 1:, :]
    
    pose_norm = torch.sum(torch.abs(pose_gt - pose_pred), dim=-1)
    root_norm = torch.sum((root_gt - root_pred) ** 2, dim=-1)

    
    if mask is not None:
        # (batch, seq, num_joint+1)
        pose_mask = mask[..., 1:]
        root_mask = mask[..., 0]
        
        inv_pose_mask = ~pose_mask
        inv_root_mask = ~root_mask
        
        inv_pose_mask = inv_pose_mask.float()
        inv_root_mask = inv_root_mask.float()
        
        # (batch, seq)
        denom_pose = torch.sum(inv_pose_mask, -1)
        
        # (batch, seq, num_joint+1) -> # (batch, seq, num_joint+1)
        pose_norm *= inv_pose_mask
        root_norm *= inv_root_mask
        
        # (batch, seq, num_joint+1) -> # (batch, seq)
        per_pose_mean = torch.sum(pose_norm, dim=-1) / (denom_pose + 1e-6)
        
    else:
        per_pose_mean = torch.mean(pose_norm, dim=-1)
        
    mean = torch.mean(per_pose_mean + root_norm)

    return mean
    
        
def mpjpev6(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
    root_pred = pred[..., 0, :]
    root_gt = future[..., 0, :]
    pose_pred = pred[..., 1:, :]
    pose_gt = future[..., 1:, :]
    
    pose_norm = torch.sum((pose_gt - pose_pred) ** 2, dim=-1)
    root_norm = torch.sum((root_gt - root_pred) ** 2, dim=-1)

    
    if mask is not None:
        # (batch, seq, num_joint+1)
        pose_mask = mask[..., 1:]
        root_mask = mask[..., 0]
        
        inv_pose_mask = ~pose_mask
        inv_root_mask = ~root_mask
        
        inv_pose_mask = inv_pose_mask.float()
        inv_root_mask = inv_root_mask.float()
        
        # (batch, seq)
        denom_pose = torch.sum(inv_pose_mask, -1)
        
        # (batch, seq, num_joint+1) -> # (batch, seq, num_joint+1)
        pose_norm *= inv_pose_mask
        root_norm *= inv_root_mask
        
        # (batch, seq, num_joint+1) -> # (batch, seq)
        per_pose_mean = torch.sum(pose_norm, dim=-1) 
        per_pose_mean = torch.sum(per_pose_mean, dim=-1) 
        root_norm = torch.sum(root_norm, dim=-1) 
        
    else:
        per_pose_mean = torch.sum(pose_norm, dim=-1)
        per_pose_mean = torch.sum(per_pose_mean, dim=-1) 
        root_norm = torch.sum(root_norm, dim=-1) 
    
    mean = torch.mean(per_pose_mean + root_norm)

    return mean


def mpjpev7(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
#     root_pred = pred[..., 0, :]
#     root_gt = future[..., 0, :]
#     pose_pred = cvt_absolute_pose(root_pred, pred[..., 1:, :])
#     pose_gt = cvt_absolute_pose(root_gt, future[..., 1:, :])
    
    pose_norm = torch.sum((pred - future) ** 2, dim=-1)

    
    if mask is not None:
        # (batch, seq, num_joint+1)
        pose_mask = mask
        
        inv_pose_mask = ~pose_mask
        
        inv_pose_mask = inv_pose_mask.float()
        
        # (batch, seq)
        denom_pose = torch.sum(inv_pose_mask, -1)
        
        # (batch, seq, num_joint+1) -> # (batch, seq, num_joint+1)
        pose_norm *= inv_pose_mask
        
        # (batch, seq, num_joint+1) -> # (batch, seq)
        per_pose_mean = torch.sum(pose_norm, dim=-1) / (denom_pose + 1e-6)
        
    else:
        per_pose_mean = torch.mean(pose_norm, dim=-1)
        
    mean = torch.mean(per_pose_mean)

    return mean

def mpjpev7(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
#     root_pred = pred[..., 0, :]
#     root_gt = future[..., 0, :]
#     pose_pred = cvt_absolute_pose(root_pred, pred[..., 1:, :])
#     pose_gt = cvt_absolute_pose(root_gt, future[..., 1:, :])
    
    pose_norm = torch.sum(torch.abs(pred - future), dim=-1)

    
    if mask is not None:
        # (batch, seq, num_joint+1)
        pose_mask = mask
        
        inv_pose_mask = ~pose_mask
        
        inv_pose_mask = inv_pose_mask.float()
        
        # (batch, seq)
        denom_pose = torch.sum(inv_pose_mask, -1)
        
        # (batch, seq, num_joint+1) -> # (batch, seq, num_joint+1)
        pose_norm *= inv_pose_mask
        
        # (batch, seq, num_joint+1) -> # (batch, seq)
        per_pose_mean = torch.sum(pose_norm, dim=-1) / (denom_pose + 1e-6)
        
    else:
        per_pose_mean = torch.mean(pose_norm, dim=-1)
        
    mean = torch.mean(per_pose_mean)

    return mean


def modified_mpjpe(pred, future):
    
    root_pred = pred[..., 0, :]
    root_gt = future[..., 0, :]
    pose_pred = pred[..., 1:, :]
    pose_gt = future[..., 1:, :]

    sum_per_joint = torch.sum((pose_gt - pose_pred) ** 2, dim=-1)
    sum_per_pose = torch.sqrt(sum_per_joint)
    pose_mean = torch.mean(sum_per_pose)
    
    sum_per_root = torch.sum((root_gt - root_pred) ** 2, dim=-1)
    norm_per_root = torch.sqrt(sum_per_root)
    root_mean = torch.mean(norm_per_root)
    
    # print("p: ", pose_mean)
    # print("r: ", root_mean)
    
    return pose_mean + root_mean


    