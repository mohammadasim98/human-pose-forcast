import torch
import torch.nn as nn

from typing import Union

from models.embedding import *
from models.pose.encoder import PoseEncoder
from models.temporal.encoder import TemporalEncoder
from models.vit.model import VisionTransformer
from models.pose.decoder import PoseDecoder
from models.embedding.fourier import FourierEncoding, FourierMLPEncoding



class HumanPosePredictorModel(nn.Module):
    
    def __init__(
        self,
        pose: dict,
        image: dict,
        activation: dict,
        future_window: int,
        unroll: bool=True,
    ) -> None:
        super().__init__()
        
        self.image_spatial_encoder_args = image["spatial"]["encoder"]
        self.im_temporal_encoder_args = image["temporal"]["encoder"]
        
        self.pose_spatial_encoder_args = pose["spatial"]["encoder"]
        self.pose_temporal_encoder_args = pose["temporal"]["encoder"]
        
        self.pose_spatiotemporal_temporal_encoder_args = pose["spatiotemporal"]["temporal"]
        self.pose_spatiotemporal_decoder_args = pose["spatiotemporal"]["spatial"]["decoder"]
        
        # Our method's building blocks
        self.pose_encoder = PoseEncoder(**self.pose_spatial_encoder_args)
        self.image_encoder = VisionTransformer(**self.image_spatial_encoder_args)
        
        self.im_temporal_encoder = TemporalEncoder(**self.im_temporal_encoder_args)
        self.pose_temporal_encoder = TemporalEncoder(**self.pose_temporal_encoder_args)
        
        self.decoder = PoseDecoder(
            **self.pose_spatiotemporal_decoder_args, 
            temporal=self.pose_spatiotemporal_temporal_encoder_args, 
            future_window=future_window,
            img_dim=image["spatial"]["encoder"]["hidden_dim"],
            pose_dim=pose["spatial"]["encoder"]["hidden_dim"]
            )
        
        self.pose_emb_dim = pose["spatial"]["encoder"]["hidden_dim"]
        self.embed_root = FourierMLPEncoding(num_freq=64, d_model=self.pose_emb_dim, n_input_dim=3)
        self.embed_relative_pose = FourierMLPEncoding(num_freq=128, d_model=self.pose_emb_dim, n_input_dim=3)
        
        self.linear = nn.Linear(self.pose_emb_dim, 3)
        

        self.unroll = unroll
        

    
    def pose_encoding(self, relative_poses, root_joints, history_window, unroll: bool=False):
        """ Perform spatial and temporal attention on poses

        Args:
            relative_poses (torch.Tensor) : An input of poses of shape 
                (batch_size, history_window {+ future_window}, num_joints, E)
            root_joints (torch.Tensor): An input of rootjoints of shape 
                (batch_size, history_window {+ future_window}, E)
            
        Returns:
            memory (torch.Tensor): A (batch_size, num_joints + history_window, E) Spatiotemporally encoded memory 
            tgt_pose_feat (torch.Tensor): A (batch_size, future_window + 1, E) targets for the decoder. If future is not provided, 
                target will be the current sample hence + 1. 
        """
        torch._assert(relative_poses.dim() == 4, "relative_pose dimensions must be of length 4")
        torch._assert(root_joints.dim() == 3, "relative_pose dimensions must be of length 3")
        # sequence_length can be combined with future_window + history_window or just history_window
        _, sequence_length, num_joints, dim = relative_poses.shape
        if unroll:
            relative_poses = relative_poses.view(-1, num_joints, dim)
            root_joints = root_joints.view(-1, dim)
            
            # Out Shape: (batch_size*sequence_length, num_joints, E) and (batch_size*sequence_length, E)
            memory_local, memory_global = self.pose_encoder(root_joints=root_joints, relative_poses=relative_poses)
            
            # Out Shape: (batch_size, sequence_length, num_joints, E) and (batch_size, sequence_length, E)
            memory_local = memory_local.view(-1, sequence_length, num_joints, dim)
            memory_global = memory_global.view(-1, sequence_length, dim)
            
        else:
            memory_local = []
            memory_global = []
            for i in range(sequence_length):
                mem_local, mem_global = self.pose_encoder(root_joints=root_joints[:, i, ...], relative_poses=relative_poses[:, i, ...])
                memory_local.append(mem_local.unsqueeze(1))
                memory_global.append(mem_global.unsqueeze(1))
                
            memory_local = torch.cat(memory_local, dim=1)
            memory_global = torch.cat(memory_global, dim=1)
            
        # Out Shape: (B, num_joints, E) and (B, history_window, E)
        memory_temp_local, memory_temp_global = self.pose_temporal_encoder(memory_local[:, :history_window, ...], memory_global[:, :history_window, ...])
        
        # concatenate along sequence dimension (B, num_joints + history_window, E)
        memory = torch.cat([memory_temp_local, memory_temp_global], dim=1) 

        # Need to add -1 to also include the current pose features
        # This will allow to shift the output by 1 to the right and 
        # can act as a <start> token.
        # Return memory for conditioning the decoder
        # Return target poses for decoder from future including the current pose
        # concatenate along num_joint dimension (B, future_window + 1, num_joints + 1, E)
        tgt_pose_feat = torch.cat([memory_global[:, history_window-1:, ...].unsqueeze(1), memory_local[:, history_window-1:, ...]], dim=2)

        return memory, tgt_pose_feat 
        
    
    def image_encoding(self, img_seq, mask, unroll: bool=False):
        """ Perform local and global spatial and temporal encoding on image sequences

        Args:
            img_seq (torch.Tensor): An input tensor of shape (batch_size, history_window, H, W, C) 
            mask (torch.Tensor): An input padding mask (batch_size, H, W) 
            unroll (bool): if True use unrolled version i.e, batch_size*sequence_length. Else, 
                use iteration. Defaults to False.
             
        Returns:
            torch.tensor: A tensor containing concatenated spatially and temporally encoded 
                local and global image features
        """
        torch._assert(img_seq.dim() == 5, "input image data must be of shape (batch_size, history_window, H, W, C)")
        torch._assert(mask.dim() == 3 or mask.dim() == 4, "input image data must be of shape (batch_size, H, W) or (batch_size, H, W, 1)")
        
        img_seq = img_seq.permute(0, 1, 4, 2, 3)
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        
            
        _, history_window, C, H, W = img_seq.shape

        if unroll:
            mask = mask.unsqueeze(1)
            img_seq = img_seq.view(-1, C, H, W)
            mask= mask.repeat(1, history_window, 1, 1, 1)
            mask = mask.view(-1, H, W, 1)
            mask = mask.permute(0, 3, 1, 2) # ViT takes channel first format

            # Out Shape: (batch_size*history_window, num_patches + 1, E) and (batch_size*history_window, E)
            memory_local, memory_global = self.image_encoder(x=img_seq, key_padding_mask=mask)
            
            # Out Shape: (batch_size, history_window, num_patches + 1, E) and (batch_size, history_window, E)
            memory_local = memory_local.view(-1, history_window, memory_local.shape[1], memory_local.shape[2])
            memory_global = memory_global.view(-1, history_window, memory_global.shape[1])
            
        else:
            memory_local = []
            memory_global = []
            mask = mask.permute(0, 3, 1, 2)
            for i in range(history_window):
                # Out Shape: (batch_size, num_patches + 1, E) and (batch_size, E)
                mem_local, mem_global = self.image_encoder(x=img_seq[:, i, ...], key_padding_mask=mask)
                memory_local.append(mem_local.unsqueeze(1))
                memory_global.append(mem_global.unsqueeze(1))
            
            # Out Shape: (batch_size, history_window, num_patches + 1, E) and (batch_size, history_window, E)
            memory_local = torch.cat(memory_local, dim=1)
            memory_global = torch.cat(memory_global, dim=1)
            
        # Get local and global temporally encoded features of sequences of images and poses
        # Out Shape: (B, num_patches + 1, E) and (B, history_window, E)
        memory_local, memory_global = self.im_temporal_encoder(memory_local, memory_global)
        
        # concatenate along sequence dimension (B, num_patches + history_window + 1, E)
        return torch.cat([memory_local, memory_global], dim=1) 
        
    def forward(self, history: list, future: Union[list, None], is_teacher_forcing: bool=False):
        """Perform forward pass

        Args:
            history (list(torch.tensor, torch.tensor, torch.tensor, torch.tensor)): 
                A list of [imgs, norm_poses, root_joints, mask] each with their respective shape as
                [
                    (batch_size, history_window, H, W, C), 
                    (batch_size, history_window, num_joints, 2), 
                    (batch_size, history_window, 2), 
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
        torch._assert(history[0].dim() == 5, f"Expected (batch_size, history_window, H, W, C) got {history[0].shape}")
        torch._assert(history[1].dim() == 4, f"Expected (batch_size, history_window, num_joints, 2) got {history[1].shape}")
        torch._assert(history[2].dim() == 3, f"Expected (batch_size, history_window, 2) got {history[2].shape}")
        torch._assert(history[3].dim() == 3, f"Expected (batch_size, H, W) got {history[3].shape}")
        torch._assert(future[0].dim() == 4, f"Expected (batch_size, future_window, num_joints, 2) got {future[0].shape}")
        torch._assert(future[1].dim() == 3, f"Expected (batch_size, future_window, 2) got {future[1].shape}")

        ######################################################################
        # TODO: Need to implement a forward method for pose encoder
        # example:
        
        img_seq = history[0].float()
        b, history_window, H, W, C = img_seq.shape
        
        mask = history[3].float()
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
        
        _, seq_length, num_joints, pose_dim = relative_poses.shape 
        ##########################################################
        # TODO: Need to Embed the root_joints and relative_poses
        # Out Shape: (B, future_window + history_window, J, E) and (B, future_window + history_window, E)
        
        relative_poses = self.embed_relative_pose(relative_poses.view(-1, num_joints, pose_dim))
        relative_poses = relative_poses.view(-1, seq_length, num_joints, self.pose_emb_dim)
        root_joints = self.embed_root(root_joints)
        
        # Get combined* local and global features from sequences of images with padding mask
        
            
        memory_img = self.image_encoding(img_seq=img_seq, mask=mask, unroll=self.unroll)
        
        # Get combined* local and global features from sequences of poses
        memory_poses, tgt_poses = self.pose_encoding(relative_poses=relative_poses, root_joints=root_joints, history_window=history_window, unroll=self.unroll)
        
        # Autoregressive decoder with "dual" conditioning
        # Currently uses only combined local and global features. Need to modify it later for further evaluation.
        # Out Shape: (B, future_window + 1, J, 2)
        
        
        out_poses = self.decoder(img_encoding=memory_img, pos_encoding=memory_poses, tgt=tgt_poses, is_teacher_forcing=is_teacher_forcing)

        return self.linear(out_poses)