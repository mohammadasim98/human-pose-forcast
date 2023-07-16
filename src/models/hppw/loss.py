
import torch

def mpjpe(pred, future):

    root_relative_poses = future[2]
    root_joints = future[1]

    future_poses = torch.cat([root_joints.unsqueeze(1), root_relative_poses], dim=1)

    pass


    