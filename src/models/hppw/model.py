import torch
import torch.nn as nn

from embedding import *
from models.pose.encoder import PoseEncoder
from models.temporal.encoder import TemporalEncoder
from models.vit.model import VisionTransformer
from models.pose.decoder import PoseDecoder


class HumanPosePredictorModel(nn.Module):
    
    def __init__(
        self,
        encoder: dict,
        decoder: dict,
        history_window: int=15,
        future_window: int=15,
        batch_first: bool=True
    ):
        super().__init__()
        
        self.pose_encoder_args = encoder["pose"]
        self.image_encoder_args = encoder["image"]
        self.temporal_encoder_args = encoder["temporal"]
        self.pose_decoder_args = decoder["pose"]
        
        self.pose_encoder = PoseEncoder(**self.pose_encoder_args)
        self.image_encoder = VisionTransformer(**self.image_encoder_args)
        self.temporal_encoder = TemporalEncoder(**self.temporal_encoder_args)
        
        self.pose_decoder = PoseDecoder(**self.pose_decoder_args)
        
        ######################################
        # TODO: Need to implement
        
        raise NotImplementedError
        
    def forward(self, history, future):
        """Perform forward pass

        Args:
            history (list(torch.tensor, torch.tensor, torch.tensor, torch.tensor)): 
                A list of [imgs, norm_poses, root_joints, mask] each with their respective shape as
                [
                    (batch_size, history_window, H, W, C), 
                    (batch_size, history_window, num_joints, 3), 
                    (batch_size, history_window, 3), 
                    (batch_size, H, W)
                ]
                
            future (list(torch.tensor, torch.tensor)): 
                A list of [norm_poses, root_joints] each with their respective shape as
                [
                    (batch_size, future_window, num_joints, 3), 
                    (batch_size, future_window, 3), 
                ]
  
        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(input.dim() == 3, f"Expected (batch_size, num_joints, hidden_dim) got {input.shape}")

        ######################################################################
        # TODO: Need to implement a forward method for pose encoder
        ######################################################################
        raise NotImplementedError