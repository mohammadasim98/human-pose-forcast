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
        vit_weights: dict,
        history_window: int=15,
        future_window: int=15,
    ):
        super().__init__()
        
        self.pose_encoder_args = encoder["pose"]
        self.image_encoder_args = encoder["image"]
        self.temporal_encoder_args = encoder["temporal"]
        
        self.pose_decoder_args = decoder["pose"]
        
        self.pose_encoder = PoseEncoder(**self.pose_encoder_args)
        self.image_encoder = VisionTransformer(**self.image_encoder_args)
        self.im_temporal_encoder = TemporalEncoder(**self.temporal_encoder_args)
        self.pose_temporal_encoder = TemporalEncoder(**self.temporal_encoder_args)
        
        self.decoder = PoseDecoder(**self.pose_decoder_args)
        
        self.vit_weights = vit_weights
        self.history_window = history_window
        self.future_window = future_window
        
        # Load Vit weights
        self.image_encoder.load_state_dict(self.vit_weights, strict=True)

        self.training = False
        
        ######################################
        # TODO: Need to implement
        
        raise NotImplementedError
        
    def forward(self, history, future=None, training=False):
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
        # example: 
        
        if training:
            # Use concatenated history and future poses
             
            # Out Shape: (B, future_window + history_window, J, 3)
            relative_poses = torch.cat([history[1], future[0]])
            
            # Out Shape: (B, future_window + history_window, 3)
            root_joints = torch.cat([history[2], future[1]])
            
            # Out Shape: (B, future_window + history_window, J, E) and (B, future_window + history_window, E)
            local_pose_feat, global_pose_feat = self.pose_encoder(root_joint=root_joints, relative_pose=relative_poses)
        else:
            # Use only history for inference/test 
            
            # Out Shape: (B, history_window, J, E) and (B, history_window, E)
            local_pose_feat, global_pose_feat = self.pose_encoder(root_joint=history[2], relative_pose=history[1])
        
        # Get local and global features from sequences of images and mask
        # Out Shape: (B, history_window, num_patches + 1, E) and (B, history_window, E)
        local_im_feat, global_im_feat = self.image_encoder(inputs=history[0], mask=history[3])

        # Get local and global temporally encoded features of sequences of images and poses
        # Out Shape: (B, history_window, E) and (B, history_window, E)
        local_im_feat, global_im_feat = self.im_temporal_encoder(local_im_feat, global_im_feat)
        local_pose_feat, global_pose_feat = self.pose_temporal_encoder(local_pose_feat, global_pose_feat)

        # Autoregressive decoder with "dual" conditioning
        # Out Shape: (B, future_window, J, 3)
        out_poses = self.decoder(local_im_feat, global_im_feat, local_pose_feat, global_pose_feat, training=training)
                
        return out_poses