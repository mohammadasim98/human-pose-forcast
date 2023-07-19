import torch
import torchvision
import torch.nn as nn
from ..base_model import BaseModel
from ..vit import model as vit_model


class LinearProjection(nn.module):
    
