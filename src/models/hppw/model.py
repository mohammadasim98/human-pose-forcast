import torch
import torch.nn as nn

from typing import Union

from models.embedding import *
from models.pose.encoder import PoseEncoder
from models.temporal.encoder import TemporalEncoder
from models.vit.model import VisionTransformer
from models.pose.decoder import PoseDecoder
from models.embedding.fourier import FourierEncoding, FourierMLPEncoding
from models.projection.model import LinearProjection



class HumanPosePredictorModel(nn.Module):
    
    def __init__(
        self,
        pose: dict,
        image: dict,
        activation: dict,
        future_window: int,
        unroll: bool=True,
        device: str="cuda:0"
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
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False

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
        self.embed_root = FourierMLPEncoding(num_freq=64, d_model=self.pose_emb_dim, n_input_dim=2)
        self.embed_relative_pose = FourierMLPEncoding(num_freq=128, d_model=self.pose_emb_dim, n_input_dim=2)
        
        self.linear = nn.Linear(self.pose_emb_dim, 2)

        self.unroll = unroll
        self.device = device

    
    def pose_encoding(self, relative_poses, root_joints, history_window, unroll: bool=False):
        """ Perform spatial and temporal attention on poses

        Args:
            relative_poses (torch.Tensor) : An input of poses of shape 
                (batch_size, history_window {+ future_window}, num_joints, 2)
            root_joints (torch.Tensor): An input of rootjoints of shape 
                (batch_size, history_window {+ future_window}, 2)
            
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
            
            relative_poses = self.embed_relative_pose(relative_poses)
            root_joints = self.embed_root(root_joints)
            # Out Shape: (batch_size*sequence_length, num_joints, E) and (batch_size*sequence_length, E)
            memory_local, memory_global = self.pose_encoder(root_joints=root_joints, relative_poses=relative_poses)
            
            # Out Shape: (batch_size, sequence_length, num_joints, E) and (batch_size, sequence_length, E)
            memory_local = memory_local.view(-1, sequence_length, num_joints, self.pose_emb_dim)
            memory_global = memory_global.view(-1, sequence_length, self.pose_emb_dim)
            
        else:
            memory_local = []
            memory_global = []
            for i in range(sequence_length):
                relative_poses = self.embed_relative_pose(relative_poses[:, i, ...])
                root_joints = self.embed_root(root_joints[:, i, ...])
                mem_local, mem_global = self.pose_encoder(root_joints=root_joints, relative_poses=relative_poses)
                memory_local.append(mem_local.unsqueeze(1))
                memory_global.append(mem_global.unsqueeze(1))
                
            memory_local = torch.cat(memory_local, dim=1)
            memory_global = torch.cat(memory_global, dim=1)
            
        # Out Shape: (B, num_joints, E) and (B, history_window, E)
        memory_temp_local, memory_temp_global, attention_weights = self.pose_temporal_encoder(memory_local[:, :history_window, ...], memory_global[:, :history_window, ...])
        
        # concatenate along sequence dimension (B, num_joints + history_window, E)
        memory = torch.cat([memory_temp_local, memory_temp_global], dim=1) 

        # Need to add -1 to also include the current pose features
        # This will allow to shift the output by 1 to the right and 
        # can act as a <start> token.
        # Return memory for conditioning the decoder
        # Return target poses for decoder from future including the current pose
        # concatenate along num_joint dimension (B, future_window + 1, num_joints + 1, E)
        tgt_pose_feat = torch.cat([memory_global[:, history_window-1:, ...].unsqueeze(1), memory_local[:, history_window-1:, ...]], dim=2)

        return memory, tgt_pose_feat, attention_weights
        
    
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
        memory_local, memory_global, attention_weights = self.im_temporal_encoder(memory_local, memory_global)
        
        # concatenate along sequence dimension (B, num_patches + history_window + 1, E)
        return torch.cat([memory_local, memory_global], dim=1), attention_weights
        
    def forward(self, img_seq: torch.Tensor, relative_pose_seq: torch.Tensor, root_joint_seq: torch.Tensor, mask: Union[torch.Tensor, None]):
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
                    (batch_size, future_window, num_joints, 2), 
                    (batch_size, future_window, 2), 
                ]
            is_teacher_forcing (bool): A boolean if True set to use teacher forcing method 
                for autoregressive. If False non-autoregressive decoding (may not be good to use all the time during training). 
                    Defaults to False.
  
        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(img_seq.dim() == 5, f"Expected (batch_size, history_window, H, W, C) got {img_seq.shape}")
        torch._assert(relative_pose_seq.dim() == 4, f"Expected (batch_size, history_window, num_joints, 2) got {relative_pose_seq.shape}")
        torch._assert(root_joint_seq.dim() == 3, f"Expected (batch_size, history_window, 2) got {root_joint_seq.shape}")
        torch._assert(mask.dim() == 3, f"Expected (batch_size, H, W) got {mask.shape}")


        # mask = mask.float()
        # img_seq = img_seq.float()

        _, history_window, _, _, _ = img_seq.shape
        _, history_window, num_joints, pose_dim = relative_pose_seq.shape 


        # Get combined* local and global features from sequences of images with padding mask    
        memory_img, image_attentions = self.image_encoding(img_seq=img_seq, mask=mask, unroll=self.unroll)
        
        # Get combined* local and global features from sequences of poses
        memory_poses, tgt_poses, pose_attentions = self.pose_encoding(relative_poses=relative_pose_seq, root_joints=root_joint_seq, history_window=history_window, unroll=self.unroll)
        
        # Autoregressive decoder with "dual" conditioning
        # Currently uses only combined local and global features. Need to modify it later for further evaluation.
        # Out Shape: (B, future_window + 1, J, 2)
        
        
        out_poses, decoder_attentions = self.decoder(img_encoding=memory_img, pos_encoding=memory_poses, tgt=tgt_poses)

        return self.linear(out_poses[:, 1:, ...]), [image_attentions, pose_attentions, decoder_attentions]
    
    
    
class HumanPosePredictorModel3D(HumanPosePredictorModel):
    
    def __init__(
        self,
        pose: dict,
        image: dict,
        activation: dict,
        projection: dict,
        future_window: int,
        unroll: bool=True,
        device: str="cuda:0"
    ) -> None:
        super().__init__(pose, image, activation, future_window, unroll, device)
        
        self.projection = LinearProjection(**projection).cuda()

    def forward(self, img_seq: torch.Tensor, relative_pose_seq: torch.Tensor, root_joint_seq: torch.Tensor, mask: Union[torch.Tensor, None], proj_future: bool=True):
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
                    (batch_size, future_window, num_joints, 2), 
                    (batch_size, future_window, 2), 
                ]
            is_teacher_forcing (bool): A boolean if True set to use teacher forcing method 
                for autoregressive. If False non-autoregressive decoding (may not be good to use all the time during training). 
                    Defaults to False.
  
        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(img_seq.dim() == 5, f"Expected (batch_size, history_window, H, W, C) got {img_seq.shape}")
        torch._assert(relative_pose_seq.dim() == 4, f"Expected (batch_size, history_window, num_joints, 2) got {relative_pose_seq.shape}")
        torch._assert(root_joint_seq.dim() == 3, f"Expected (batch_size, history_window, 2) got {root_joint_seq.shape}")
        torch._assert(mask.dim() == 3, f"Expected (batch_size, H, W) got {mask.shape}")


        # mask = mask.float()
        # img_seq = img_seq.float()

        _, history_window, _, _, _ = img_seq.shape
        _, history_window, num_joints, pose_dim = relative_pose_seq.shape 


        # Get combined* local and global features from sequences of images with padding mask    
        memory_img, image_attentions = self.image_encoding(img_seq=img_seq, mask=mask, unroll=self.unroll)
        
        # Get combined* local and global features from sequences of poses
        memory_poses, tgt_poses, pose_attentions = self.pose_encoding(relative_poses=relative_pose_seq, root_joints=root_joint_seq, history_window=history_window, unroll=self.unroll)
        
        # Autoregressive decoder with "dual" conditioning
        # Currently uses only combined local and global features. Need to modify it later for further evaluation.
        # Out Shape: (B, future_window + 1, J, 2)
        
        
        out_poses, decoder_attentions = self.decoder(img_encoding=memory_img, pos_encoding=memory_poses, tgt=tgt_poses)

        out_2d = self.linear(out_poses[:, 1:, ...])
        history_poses = torch.cat([root_joint_seq.unsqueeze(2), relative_pose_seq], dim=2)
        
        if proj_future:
            proj = torch.cat([history_poses, out_2d], dim=1)
        else:
            proj = history_poses

        out_3d = self.projection(proj)
        return out_2d, out_3d, [image_attentions, pose_attentions, decoder_attentions]