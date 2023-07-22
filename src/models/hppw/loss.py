
import torch
from typing import Union
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

def mpjpe(pred, future, mask: Union[torch.Tensor, None]=None):

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


    