
import numpy as np

def world2camera_transform(position, extrinsics, intrinsics):
    
    x = extrinsics @ position.T
    y = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
    result = np.transpose(y @ x)
    return result / result[:, 2].astype(int)
    
    