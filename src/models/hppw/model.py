import torch
import torch.nn as nn

from typing import Union

from ..embedding import *
from models.pose.encoder import PoseEncoder
from models.temporal.encoder import TemporalEncoder
from models.vit.model import VisionTransformer
from models.pose.decoder import PoseDecoder



class HumanPosePredictorModel(nn.Module):
    
    def __init__(
        self,
        pose: dict,
        image: dict,
        temporal: dict,
        vit_weights: str
    ) -> None:
        super().__init__()
        
        self.pose_encoder_args = pose["encoder"]
        self.image_encoder_args = image["encoder"]
        self.temporal_encoder_args = temporal["encoder"]
        self.pose_decoder_args = pose["decoder"]
        
        self.pose_encoder = PoseEncoder(**self.pose_encoder_args)
        self.image_encoder = VisionTransformer(**self.image_encoder_args)
        self.im_temporal_encoder = TemporalEncoder(**self.temporal_encoder_args)
        self.pose_temporal_encoder = TemporalEncoder(**self.temporal_encoder_args)
        
        self.decoder = PoseDecoder(**self.pose_decoder_args)
        
        self.vit_weights = vit_weights
        
        # Load Vit weights
        self.image_encoder.load_state_dict(self.vit_weights, strict=True)

        
        ######################################
        # TODO: Need to implement
        
        raise NotImplementedError
    
    def pose_encoding(self, relative_poses, root_joints, history_window):
        """ Perform spatial and temporal attention on poses

        Args:
            relative_poses (_type_): _description_
            root_joints (_type_): _description_
        """
        
        
        
        memory_local, memory_global = self.pose_encoder(root_joint=root_joints, relative_pose=relative_poses)
        # Out Shape: (B, num_joints, E) and (B, history_window, E)
        memory_temp_local, memory_temp_global = self.pose_temporal_encoder(memory_local[:, :history_window, ...], memory_global[:, :history_window, ...])
        memory = torch.cat([memory_temp_local, memory_temp_global], dim=1) # concatenate along sequence dimension (B, num_patches + history_window + 1, E)
        # Need to add -1 to also include the current pose features
        # This will allow to shift the output by 1 to the right and 
        # can act as a <start> token.
        # Return memory for conditioning the decoder
        # Return target poses for decoder from future including the current pose
        tgt_pose_feat = torch.cat([memory_local[:, history_window-1:, ...], memory_global[:, history_window-1:, ...]])
        return memory, tgt_pose_feat 
        
    
    def image_encoding(self, img_seq, mask):
        """ Perform local and global spatial and temporal encoding on image sequences

        Args:
            img_seq (torch.Tensor): An input tensor of shape (batch_size, history_window, H, W, C)
            mask (torch.Tensor): An input padding mask (batch_size, H, W) for the whole hisotry sequence

        Returns:
            torch.tensor: A tensor containing concatenated spatially and temporally encoded 
                local and global image features
        """
        # Out Shape: (B, history_window, num_patches + 1, E) and (B, history_window, E)
        memory_local, memory_global = self.image_encoder(inputs=img_seq, mask=mask)

        # Get local and global temporally encoded features of sequences of images and poses
        # Out Shape: (B, num_patches + 1, E) and (B, history_window, E)
        memory_local, memory_global = self.im_temporal_encoder(memory_local, memory_global)
        
        return torch.cat([memory_local, memory_global], dim=1) # concatenate along sequence dimension (B, num_patches + history_window + 1, E)
        
    def forward(self, history: list, future: Union[list, None], is_teacher_forcing: bool=False):
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
            is_teacher_forcing (bool): A boolean if True set to use teacher forcing method 
                for autoregressive. If False non-autoregressive decoding (may not be good to use all the time during training). 
                    Defaults to False.
  
        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(input.dim() == 3, f"Expected (batch_size, num_joints, hidden_dim) got {input.shape}")

        ######################################################################
        # TODO: Need to implement a forward method for pose encoder
        # example:
        
        img_seq = history[0]
        history_window = img_seq.shape[1]
        
        mask = history[3]
        if is_teacher_forcing:
            # Use concatenated history and future poses

            # Out Shape: (B, future_window + history_window, J, 3)
            relative_poses = torch.cat([history[1], future[0]])

            # Out Shape: (B, future_window + history_window, 3)
            root_joints = torch.cat([history[2], future[1]])

        else:
            # Use only history for inference/test
            # Shape: (B, history_window, J, 3) and (B, history_window, 3)
            relative_poses = history[1]
            root_joints = history[2]
            
        ##########################################################
        # TODO: Need to Embed the root_joints and relative_poses
        # Out Shape: (B, future_window + history_window, J, E) and (B, future_window + history_window, E)
        
        # Get combined* local and global features from sequences of images with padding mask
        memory_img = self.image_encoding(img_seq=img_seq, mask=mask)
        
        # Get combined* local and global features from sequences of poses
        memory_poses, tgt_poses = self.pose_encoding(relative_poses=relative_poses, root_joints=root_joints, history_window=history_window)
        
        # Autoregressive decoder with "dual" conditioning
        # Currently uses only combined local and global features. Need to modify it later for further evaluation.
        # Out Shape: (B, future_window, J, 2)
        
        out_poses = self.decoder(img_encoding=memory_img, pos_encoding=memory_poses, target=tgt_poses, is_teacher_forcing=is_teacher_forcing)
        
        return out_poses