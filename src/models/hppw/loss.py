
import torch

def mpjpe(prediction, future):
    
    
    
    return torch.mean((prediction - future)**2)