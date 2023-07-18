
import torch

def mpjpe(pred, future):

    sum_per_joint = torch.sum((future - pred) ** 2, dim=-1)
    norm_per_joint = torch.sqrt(sum_per_joint)
    mean = torch.mean(norm_per_joint)

    return mean


    