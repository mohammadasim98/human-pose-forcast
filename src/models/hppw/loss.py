
import torch
from typing import Union
from models.hppw.transforms import cvt_absolute_pose
from utils.viz import CONNECTIONS_2D
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


point_connection2d = [
    [0, 1], #0
    [1, 2], #1
    [1, 5], #2
    [2, 3], #3
    [3, 4], #4
    [5, 6], #5
    [6, 7], #6
    [2, 8], #7
    [8, 9], #8
    [8, 11], #9
    [9, 10], #10
    [5, 11], #11
    [11, 12], #12
    [12, 13] #13
]

link_connection2d = [
    [0, 1],
    [0, 2],
    [1, 2],
    [1, 3],
    [3, 4],
    [2, 5],
    [5, 6],
    [3, 7],
    [5, 11],
    [1, 7],
    [2, 11],
    [7, 9],
    [11, 9],
    [7, 8],
    [11, 12],
    [8, 9],
    [12, 9],
    [8, 10],
    [12, 13],

]

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

def mpjpev8(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

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

def mpjpev9(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
#     root_pred = pred[..., 0, :]
#     root_gt = future[..., 0, :]
#     pose_pred = cvt_absolute_pose(root_pred, pred[..., 1:, :])
#     pose_gt = cvt_absolute_pose(root_gt, future[..., 1:, :])
    
    pose_norm = torch.sum(torch.square(pred - future), dim=-1)

    
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
    
    per_pose_mean = torch.sqrt(torch.sum(pose_norm, dim=-1))
        
    mean = torch.mean(per_pose_mean)

    return mean

def mpjpev10(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
#     root_pred = pred[..., 0, :]
#     root_gt = future[..., 0, :]
#     pose_pred = cvt_absolute_pose(root_pred, pred[..., 1:, :])
#     pose_gt = cvt_absolute_pose(root_gt, future[..., 1:, :])
    
    pose_norm = torch.sum(torch.square(pred - future), dim=-1)

    
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
    root_norm = pose_norm[..., 0]
    relative_norm = torch.sum(pose_norm[..., 1:], dim=-1)
    per_root_sqrt = torch.sqrt(root_norm)
    per_pose_sqrt = torch.sqrt(relative_norm)
        
    mean = torch.mean(per_pose_sqrt + per_root_sqrt)

    return mean


def mpjpev11(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
#     root_pred = pred[..., 0, :]
#     root_gt = future[..., 0, :]
#     pose_pred = cvt_absolute_pose(root_pred, pred[..., 1:, :])
#     pose_gt = cvt_absolute_pose(root_gt, future[..., 1:, :])
    
    pose_norm = torch.sum(torch.square(pred - future), dim=-1)

    
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
    num_joints = pose_norm.shape[-1] - 1
    root_norm = pose_norm[..., 0]
    relative_norm = torch.sum(pose_norm[..., 1:], dim=-1)
    per_root_sqrt = torch.sqrt(root_norm)
    per_pose_sqrt = torch.sqrt(relative_norm)
        
    mean = torch.mean((per_pose_sqrt + num_joints*per_root_sqrt) / num_joints)

    return mean

def mpjpev12(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

    # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
    
#     root_pred = pred[..., 0, :]
#     root_gt = future[..., 0, :]
#     pose_pred = cvt_absolute_pose(root_pred, pred[..., 1:, :])
#     pose_gt = cvt_absolute_pose(root_gt, future[..., 1:, :])
    
    pose_norm = torch.sum(torch.square(pred - future), dim=-1)

    
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
    num_joints = pose_norm.shape[-1] - 1
    root_norm = pose_norm[..., 0]
    relative_norm = torch.sum(pose_norm[..., 1:], dim=-1)

        
    mean = torch.mean(relative_norm + root_norm)

    return mean

def mpjpev13(pred, future, mask: Union[torch.Tensor, None]=None, l1=None):

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
    num_joints = pose_norm.shape[-1] - 1
    root_norm = pose_norm[..., 0]
    relative_norm = torch.sum(pose_norm[..., 1:], dim=-1)

        
    mean = torch.mean(relative_norm + root_norm)

    return mean

# def mpjpe_plus_angle(pred, future, connections, mask: Union[torch.Tensor, None]=None):
#     # (batch, seq, num_joint+1, 3) -> # (batch, seq, num_joint+1)
#     mpjpe_err = mpjpev10(pred, future, mask=mask)
    
#     links_pred = pred[:, :, connections]
#     links_gt = future[:, :, connections]
    
#     links_angle_pred = 



def mse(pred, gt, sum_dim=-1):
    
    return torch.mean(torch.sum((pred - gt)**2, dim=sum_dim))


def angular_loss(norm_direc_pred, norm_direc_gt):
    
    link_pair_pred = norm_direc_pred[..., link_connection2d, :]
    link_pair_gt = norm_direc_gt[..., link_connection2d, :]
    
    link_pair_pred_dot = torch.sum(link_pair_pred[..., 0, :] * link_pair_pred[..., 1, :], dim=-1)
    link_pair_gt_dot = torch.sum(link_pair_gt[..., 0, :] * link_pair_gt[..., 1, :], dim=-1)
    
    # print(link_pair_gt_dot)
    # print(link_pair_pred_dot)
    
    angle_pred = torch.acos(link_pair_pred_dot)
    angle_gt = torch.acos(link_pair_gt_dot)

    
#     angle_gt = torch.nan_to_num(angle_gt, nan=0.0)
#     angle_pred = torch.nan_to_num(angle_pred, nan=0.0)
    
    return mse(angle_pred, angle_gt)

def angular_loss(norm_direc_pred, norm_direc_gt):
    
    link_pair_pred = norm_direc_pred
    link_pair_gt = norm_direc_gt
    
    link_pair_pred_dot = torch.sum(link_pair_pred[0] * link_pair_pred[1], dim=-1)
    link_pair_gt_dot = torch.sum(link_pair_gt[0] * link_pair_gt[1], dim=-1)
    
    # print(link_pair_gt_dot)
    # print(link_pair_pred_dot)
    
    angle_pred = torch.acos(link_pair_pred_dot)
    angle_gt = torch.acos(link_pair_gt_dot)

    
#     angle_gt = torch.nan_to_num(angle_gt, nan=0.0)
#     angle_pred = torch.nan_to_num(angle_pred, nan=0.0)
    
    return mse(angle_pred, angle_gt)

def combined_loss(pred, future, mask: Union[torch.Tensor, None]=None):
    
    relative_pred = pred[..., 1:, :]
    relative_gt = future[..., 1:, :]
    
    links_pred = relative_pred[..., point_connection2d, :]
    links_gt = relative_gt[..., point_connection2d, :]

    direc_pred = links_pred[..., 0, :] - links_pred[..., 1, :]
    direc_gt = links_gt[..., 0, :] - links_gt[..., 1, :]

    len_pred = torch.linalg.norm(direc_pred, ord=2, dim=-1, keepdim=True)
    len_gt = torch.linalg.norm(direc_gt, ord=2, dim=-1, keepdim=True)
    
    norm_direc_pred = direc_pred / (len_pred + 1e-5)
    norm_direc_gt = direc_gt / (len_gt + 1e-5)
    
    loss = 0
    loss += angular_loss(norm_direc_pred, norm_direc_gt)
    loss += mse(len_pred, len_gt, sum_dim=[-1, -2])
    # loss += mpjpev10(pred, future, mask)
    
    return loss

def combined_loss(pred, future, mask: Union[torch.Tensor, None]=None):
    
    relative_pred = pred[..., 1:, :]
    relative_gt = pred[..., 1:, :]
    abs_pose_pred = cvt_absolute_pose(pred[..., 0, :], relative_pred)
    abs_pose_gt = cvt_absolute_pose(pred[..., 0, :], relative_pred)
    links_pred = relative_pred[..., point_connection2d, :]
    links_gt = relative_gt[..., point_connection2d, :]

    direc_pred = relative_pred
    direc_gt = relative_gt

    len_pred = torch.linalg.norm(direc_pred, ord=2, dim=-1, keepdim=True)
    len_gt = torch.linalg.norm(direc_gt, ord=2, dim=-1, keepdim=True)
    
    norm_direc_pred = direc_pred / (len_pred + 1e-5)
    norm_direc_gt = direc_gt / (len_gt + 1e-5)
    
    loss = 0
    loss += 0.1*angular_loss(norm_direc_pred, norm_direc_gt)
    loss += mse(len_pred, len_gt, sum_dim=[-1, -2])
    loss += mpjpev10(pred, future, mask)
    
    return loss
    
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


    