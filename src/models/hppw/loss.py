
import torch


def mpjpe(pred, future):

    sum_per_joint = torch.sum((future - pred) ** 2, dim=-1)
    sum_per_pose = torch.sum(sum_per_joint, dim=-1)
    norm_per_joint = torch.sqrt(sum_per_pose)
    mean = torch.mean(norm_per_joint)

    return mean

def modified_mpjpe(pred, future):

    sum_per_joint = torch.sum((future[..., 1:, :] - pred[..., 1:, :]) ** 2, dim=-1)
    sum_per_pose = torch.sum(sum_per_joint, dim=-1)
    norm_per_pose = torch.sqrt(sum_per_pose)
    pose_mean = torch.mean(norm_per_pose)
    
    sum_per_root = torch.sum((future[..., 0, :] - pred[..., 0, :]) ** 2, dim=-1)
    norm_per_root = torch.sqrt(sum_per_root)
    root_mean = torch.mean(norm_per_root)

    return pose_mean + root_mean


    