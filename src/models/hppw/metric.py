

import torch


def cvt_absolute_pose(root_joint: torch.Tensor, norm_pose: torch.Tensor):
    """Convert root relative pose to absolute pose given a root joint.
                                (1 + norm_pose)
              abs_pose = root X ---------------
                                (1 - norm_pose)
                         
    Args:
        root_joint (numpy.ndarray): A (b, seq_len, 3) numpy array with np as number of people.
        norm_pose (np.ndarray): A (b, seq_len, 18, 3) numpy array with np as number of people.
                              
    Returns:
        numpy.ndarray: A (b, seq_len, 18, 3) absolute pose with np number of people.
    """
    root_joint = root_joint.unsqueeze(2)
    
    root_joint = torch.tile(root_joint, (1, 1, norm_pose.shape[2], 1))
    
    return ((1 + norm_pose) * root_joint / (1 - norm_pose))


class VIM:
    def __init__(self, img_size: int=224):
        self.img_size = img_size
        
    def compute_2d(self, prediction, future):
    
        root_joints_gt = future[..., 0, :] * self.img_size
        root_relative_poses_gt = future[..., 1:, :]

        gt = cvt_absolute_pose(root_joint=root_joints_gt, norm_pose=root_relative_poses_gt)

        root_joints_pred = prediction[..., 0, :] * self.img_size
        root_relative_poses_pred = prediction[..., 1:, :]

        pred = cvt_absolute_pose(root_joint=root_joints_pred, norm_pose=root_relative_poses_pred)
        
        sum_per_joint = torch.sum((gt - pred) ** 2, dim=-1)
        norm_per_joint = torch.sqrt(sum_per_joint)
        mean = torch.mean(norm_per_joint)
        
        return mean
        
    def compute_3d(self, prediction, future):
    
        prediction = cvt_absolute_pose(prediction[..., 0, :], prediction[..., 1:, :])
        
        sum_per_joint = torch.sum((future[..., 1:, :] - prediction) ** 2, dim=-1)
        norm_per_joint = torch.sqrt(sum_per_joint)
        mean = torch.mean(norm_per_joint)
        
        return mean
    
    def __str__(self):
    
        return "vim"