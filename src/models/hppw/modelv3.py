import torch
import torch.nn as nn
import math
from typing import Union
import rff


from models.embedding import *
from models.pose.encoder import PoseEncoderV2
from models.temporal.encoderv3 import TemporalEncoderV3
from models.vit.model import VisionTransformer
from models.embedding.fourier import FourierEncoding, FourierMLPEncoding
from models.projection.model import LinearProjection
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        scale = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term * scale)
        pe[:, 0, 1::2] = torch.cos(position * div_term * scale)[:,:d_model//2]
        
        self.register_buffer('pe', pe)
#         self.register_parameter('pe', torch.nn.Parameter(pe))
#         self.pe = pe.to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
#         x = x + self.pe[:x.size(0)//num_people].repeat(num_people, 1, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class HumanPosePredictorModelV3(nn.Module):
    
    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        
        activation = config["activation"]
        if activation["type"] == "ReLU":
            activation = nn.ReLU
        elif activation["type"] == "GELU":
            activation = nn.GELU
        elif activation["type"] == "LeakyReLU":
            activation = nn.LeakyReLU
        elif activation["type"] == "Sigmoid":
            activation = nn.Sigmoid
        elif activation["type"] == "Tanh":
            activation = nn.Tanh
        
        
        self.image_encoder_args = config["image_encoder"]
        self.temporal_encoder_args = config["temporal_encoder"]
        
        self.pose_encoder_args = config["pose_encoder"]
        
        self.pose_decoder_args = config["pose_decoder"]
        
        self.pose_embedding_args = config["pose_embedding"]
        # Our method's building blocks
        self.pose_encoder = PoseEncoderV2(**self.pose_encoder_args, activation=nn.Tanh)
        self.image_encoder = VisionTransformer(**self.image_encoder_args, activation=activation)
        # print(self.image_encoder_args["patch_size"])
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False

        self.im_temporal_encoder = TemporalEncoderV3(**self.temporal_encoder_args, activation=activation, num_query=((self.image_encoder_args["image_size"] // self.image_encoder_args["patch_size"])**2 + 1))
        self.pose_temporal_encoder = TemporalEncoderV3(**self.temporal_encoder_args, activation=nn.Tanh, num_query=15)
        self.dual_attention = nn.MultiheadAttention(**config["dual_attention"])
        
        hidden_dim = self.pose_decoder_args["hidden_dim"]
        num_heads = self.pose_decoder_args["num_heads"]
        dropout = self.pose_decoder_args["dropout"]
        num_layers = self.pose_decoder_args["num_layers"]
        batch_first = self.pose_decoder_args["batch_first"]
        dim_feedforward = self.pose_decoder_args["mlp_dim"]


        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation=nn.Tanh(), batch_first=batch_first)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        self.pose_emb_dim = self.pose_encoder_args["hidden_dim"]
        self.pose_embedding = rff.layers.PositionalEncoding(sigma=self.pose_embedding_args["pose_sigma"], m=self.pose_embedding_args["nfreq"])
        self.root_embedding = rff.layers.PositionalEncoding(sigma=self.pose_embedding_args["root_sigma"], m=self.pose_embedding_args["nfreq"])
        # self.pose_embedding = FourierMLPEncoding(num_freq=self.pose_embedding_args["nfreq"], d_model=self.pose_embedding_args["embed_dim"], n_input_dim=self.pose_emb_dim, activation=activation)
        
        self.output_proj = nn.Linear(self.pose_emb_dim, 2)
        
        self.embed_proj = nn.Linear(2, int(self.pose_emb_dim / (self.pose_embedding_args["nfreq"] * 2)))
        # self.embed_proj = nn.Linear(2, self.pose_embedding_args["pose_sigma"])
        # self.embed_proj = nn.Linear(2, 256)

        self.device = config["device"]
        self.activation = nn.Tanh()
        
        self.need_weights = False
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.img_feat_proj = nn.Linear(self.image_encoder_args["hidden_dim"], self.pose_encoder_args["hidden_dim"])
        
        self.img_proj = nn.Linear(self.pose_encoder_args["hidden_dim"] * ((self.image_encoder_args["image_size"] // self.image_encoder_args["patch_size"])**2 + 1), 2 * 15)
#         self.pos_seq_emb = PositionalEncoding(self.pose_embedding_args["root_sigma"], 0.0)
#         self.pos_seq_emb_proj = nn.Linear(self.pose_embedding_args["root_sigma"], self.pose_emb_dim)
        
#         self.pos_joint_emb = PositionalEncoding(self.pose_embedding_args["pose_sigma"], 0.0)
#         self.pos_joint_emb_proj = nn.Linear(self.pose_embedding_args["pose_sigma"], self.pose_embedding_args["root_sigma"])
        
    def pose_encoding(self, poses, pose_mask: Union[torch.Tensor, None]=None):
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
        torch._assert(poses.dim() == 4, "relative_pose dimensions must be of length 4")
        
        # sequence_length can be combined with future_window + history_window or just history_window
        _, sequence_length, num_joints, dim = poses.shape


        # poses = poses.view(-1, num_joints, dim)
        poses = self.embed_proj(poses)
        # poses = self.pose_embedding(poses) + poses
        local_poses = poses[..., 1:, :]
        root_joints = poses[..., 0, :].unsqueeze(2)
        
        local_poses = local_poses.permute(1, 0, 2, 3)
        root_joints = root_joints.permute(1, 0, 2, 3)
        
        local_poses = self.pose_embedding(local_poses)
        root_joints = self.root_embedding(root_joints)

        poses = torch.cat([root_joints, local_poses], dim=2)
        
#         poses = poses.view(-1, num_joints,  self.pose_embedding_args["pose_sigma"])
#         poses = poses.permute(1, 0, 2).contiguous()
#         poses = self.pos_joint_emb(poses)
#         poses = self.pos_joint_emb_proj(poses)
#         poses = poses.permute(1, 0, 2).contiguous()

#         poses = poses.view(-1, sequence_length, num_joints,  self.pose_emb_dim)
        
#         poses = poses.permute(1, 0, 2, 3).contiguous()
#         poses = poses.view(sequence_length, -1, self.pose_embedding_args["root_sigma"])
        
#         poses = self.pos_seq_emb(poses)
#         poses = self.pos_seq_emb_proj(poses)
        
#         poses = poses.view(sequence_length, -1, num_joints, self.pose_emb_dim)
        poses = poses.permute(1, 0, 2, 3).contiguous()

        
        poses = poses.view(-1, num_joints,  self.pose_emb_dim)
        
        if pose_mask is not None:
            pose_mask = pose_mask.view(-1, num_joints)

        # Out Shape: (batch_size*sequence_length, num_joints, E) and (batch_size*sequence_length, E)
        encoded_poses = self.pose_encoder(poses=poses, pose_mask=pose_mask)

        # Out Shape: (batch_size, sequence_length, num_joints, E) and (batch_size, sequence_length, E)
        encoded_poses = encoded_poses.view(-1, sequence_length, num_joints, self.pose_emb_dim)

        if pose_mask is not None:

            pose_mask = pose_mask.view((-1, sequence_length, num_joints))

        return encoded_poses
        
    
    def image_encoding(self, img_seq, mask):
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
        img_seq = img_seq[..., [2, 1, 0]]
        img_seq = img_seq.permute(0, 1, 4, 2, 3)
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        
            
        _, history_window, C, H, W = img_seq.shape

        mask = mask.unsqueeze(1)
        img_seq = img_seq.view(-1, C, H, W)
        mask= mask.repeat(1, history_window, 1, 1, 1)
        mask = mask.view(-1, H, W, 1)
        mask = mask.permute(0, 3, 1, 2) # ViT takes channel first format

        # Out Shape: (batch_size*history_window, num_patches + 1, E) and (batch_size*history_window, E)
        memory_local, _, im_mask = self.image_encoder(x=img_seq, key_padding_mask=mask)
        memory_local = self.img_feat_proj(memory_local)
        # Out Shape: (batch_size, history_window, num_patches + 1, E) and (batch_size, history_window, E)
        memory_local = memory_local.view(-1, history_window, memory_local.shape[1], memory_local.shape[2])

        im_mask = im_mask.view(-1, history_window, im_mask.shape[1])
        
        
        flatten_memory_img = torch.flatten(memory_local, -2, -1)
        proj = self.img_proj(flatten_memory_img)
        proj_history_poses = proj.view(-1, history_window, 15, 2) 
        
        # Get local and global temporally encoded features of sequences of images and poses
        # Out Shape: (B, num_patches + 1, E) and (B, history_window, E)
        memory_temp, _, attention_weights = self.im_temporal_encoder(memory_local, mask=im_mask)
        # concatenate along sequence dimension (B, num_patches + history_window + 1, E)
        
        return memory_temp, im_mask, proj_history_poses, attention_weights
        
    def forward(
        self, 
        img_seq: torch.Tensor, 
        history_pose: torch.Tensor, 
        img_mask: Union[torch.Tensor, None]=None,
        history_pose_mask: Union[torch.Tensor, None]=None,
        future_window: int=15,
        history_window: int=15,
        is_teacher_forcing: bool=False,
        future_pose: Union[torch.Tensor, None]=None,
        future_pose_mask: Union[torch.Tensor, None]=None,
        need_weights: bool=False
        ):
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
        torch._assert(history_pose.dim() == 4, f"Expected (batch_size, history_window, num_joints, 2) got {history_pose.shape}")
        
        if future_pose is not None:
            torch._assert(future_pose.dim() == 4, f"Expected (batch_size, future_window, num_joints, 2) got {future_pose.shape}")
        

        torch._assert(img_mask.dim() == 3, f"Expected (batch_size, H, W) got {img_mask.shape}")


        # mask = mask.float()
        # img_seq = img_seq.float()

        _, history_window, _, _, _ = img_seq.shape
        _, history_window, num_joints, pose_dim = history_pose.shape 
        
        if future_pose is not None and is_teacher_forcing:
            poses = torch.cat([history_pose, future_pose], dim=1)

        else:
            poses = history_pose
        image_attentions = None
        # Get combined* local and global features from sequences of images with padding mask    
        # memory_img, img_mask, proj_history_poses, image_attentions = self.image_encoding(
        #     img_seq=img_seq, 
        #     mask=img_mask
        # )
        proj_history_poses = None
        
        
        if future_pose_mask is not None and is_teacher_forcing:
            pose_mask = torch.cat([history_pose_mask, future_pose_mask], dim=1)
        else:
            pose_mask = history_pose_mask

        prev_result = poses[:, :history_window, ...]
        
        
        # History
#         pose_encoding = self.pose_encoding(
#             poses=prev_result, 
#             pose_mask=history_pose_mask if history_pose_mask is not None else None
#         )
        pose_encoding = self.pose_encoding(
            poses=poses, 
            pose_mask=pose_mask if pose_mask is not None else None
        )
        
        encoded_poses = pose_encoding
        final = []
        mem_states = None
        temp_states = None
        prev = prev_result[:, history_window-1, ...]
        mask=None
        dual_attention_weights = []
        
        
        memory_pose, temp_states, pose_attentions = self.pose_temporal_encoder(pose_encoding[:, :history_window], mask=pose_mask, states=temp_states)
        
        encoded_poses = memory_pose.unsqueeze(1)
        
        for i in range(0, future_window):
            
            
            memory_pose, temp_states, pose_attentions = self.pose_temporal_encoder(encoded_poses, mask=pose_mask, states=temp_states)
            
            dual_attention_weight = None
            if is_teacher_forcing:
                encoded_poses = torch.cat([encoded_poses, pose_encoding[:, history_window + i, ...].unsqueeze(1)], dim=1)
                
            else:
                encoded_poses = torch.cat([encoded_poses, memory_pose.unsqueeze(1)], dim=1)
#             memory, dual_attention_weight = self.dual_attention(
#                 query=memory_pose, 
#                 key=memory_img, 
#                 value=memory_img, 
#                 need_weights=True,
#                 average_attn_weights=True,
#                 key_padding_mask=img_mask[:, -1, ...] if img_mask is not None else None
#             )
            
#             memory = memory + memory_pose
#             if True:
#                 dual_attention_weights.append(dual_attention_weight)
#             # memory, mem_state = self.lstm(memory, mem_states)

            
#             result = self.decoder_layer(tgt=pose_encoding[:, -1, ...], memory=memory, tgt_key_padding_mask=mask.squeeze(1) if mask is not None else None, 
#                                        memory_key_padding_mask=mask.squeeze(1) if mask is not None else None)
            
            # (B, J, 2)
            # result = self.output_proj(memory_pose)
            # prev = future_pose[:, i, ...] if is_teacher_forcing else result
            # prev = result
            # (B, N, J, E)
            # Newly generated poses
            

                
#             if future_pose_mask is not None and is_teacher_forcing:
#                 mask = future_pose_mask[:, i, ...].unsqueeze(1) 
            
#             else:
#                 mask = None
                
#             pose_encoding = self.pose_encoding(
#                 poses=future_pose[:, i, ...].unsqueeze(1) if is_teacher_forcing else result.unsqueeze(1), 
#                 pose_mask=mask
#             )
            
#             encoded_poses = torch.cat([encoded_poses, pose_encoding], dim=1)
            
            # final.append(result.unsqueeze(1))
        
            final.append(memory_pose.unsqueeze(1))

        if len(final):
            final = torch.cat(final, dim=1)
            final = self.output_proj(final)
        # Autoregressive decoder with "dual" conditioning
        # Currently uses only combined local and global features. Need to modify it later for further evaluation.
        # Out Shape: (B, future_window + 1, J, 2)
        

        return final, proj_history_poses, [image_attentions, pose_attentions, dual_attention_weights]
    
