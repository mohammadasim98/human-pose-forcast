import os
import typing
import cv2
import sys
import torch

import numpy as np
from tfrecord.torch.dataset import TFRecordDataset

from os.path import join as ospj
sys.path.append(ospj(".", 'src'))

from utils.viz import annotate
from utils.transforms import cvt_absolute_pose

tfrecord_path = os.path.join(".","data", "3DPWPreprocessed", "tfrecords_combined", "train.tfrecord")

index_path = None
feature = {
    "sequence": "byte", 
    'height': "int", # int
    'width': "int", # int
    'frames': "int", # int
    'image_raw': "byte", # np.uint8
    'pmask': "byte", # np.uint8
    'norm_poses': "byte", # np.float32
    'root_joints': "byte", # np.int32
    'abs_poses': "byte", # np.int32
}

def _parse(features):
    flatten_image = np.frombuffer(features["image_raw"], dtype=np.uint8)
    flatten_mask = np.frombuffer(features["pmask"], dtype=np.uint8)
    flatten_norm_pose = np.frombuffer(features["norm_poses"], dtype=np.float32)
    # flatten_abs_pose = np.frombuffer(features["abs_poses"], dtype=np.int32)
    flatten_root_joint = np.frombuffer(features["root_joints"], dtype=np.int32)

    image = np.reshape(flatten_image, (features["frames"][0], -1))
    mask = np.reshape(flatten_mask, (features["height"][0], features["width"][0], -1))
    imgs = []
    for i in range(features["frames"][0]):
        imgs.append(cv2.imdecode(np.frombuffer((image[i]), dtype=np.uint8), -1))
    imgs = np.array(imgs)
    norm_pose = np.reshape(flatten_norm_pose, (-1, features["frames"][0], 18, 3))
    # abs_pose = np.reshape(flatten_abs_pose, (-1, features["frames"][0], 18, 3))
    root_joint = np.reshape(flatten_root_joint, (-1, features["frames"][0], 3))
    
    
    
    return features["frames"][0], imgs, mask, norm_pose, root_joint

def collate_fn(batch):
    
    print(type(batch))
    print(len(batch))
    return batch




        

dataset = TFRecordDataset(tfrecord_path, index_path, feature, transform=_parse)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

data = next(iter(loader))[0]
print(data)
for i in range(data[0]):

    img = data[1]
    mask = data[2]
    norm_pose = data[3]
    root_joint = data[4]
    # img = cv2.imdecode(np.fromstring((image[i]), dtype=np.uint8), -1)
    img = img[i]
    abs_pose = cvt_absolute_pose(root_joint=root_joint[:, i, :], norm_pose=norm_pose[:, i, :, :])
    annoted_img = annotate(img=img, pose=abs_pose, root_joint=root_joint[:, i, :])
    cv2.imshow("Padded Resized Image", img)
    cv2.imshow("Padding Mask", mask*255)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break    
    
    
    
cv2.destroyAllWindows()
