import os
import cv2

import numpy as np

from builtins import range
from math import sqrt, ceil


from transforms import sample, rescale

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G

def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H,W,C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G


def annotate(img, pose=None, root_joint=None, world_origin=None):
    """Annotate the given image with absolute pose-keypoints given root-relative poses 

    Args:
        img (numpy.ndarray): An input image
        pose (numpy.ndarray, optional): An (np, 18, 3) root-relative poses with np as the number of people. Defaults to None.
        root_joint (_type_, optional): An (np, 3) root joints with np as the number of people. Defaults to None.
        world_origin (_type_, optional): An (np, 3) transformed world coordinate given the camera extrinsics and intrinsics 
            (Not needed for the project). Defaults to None.

    Returns:
        numpy.ndarray: An annotated image. 
    """
    
    # Draw root joint
    #########################################################################
    # TODO:                                                                 #
    #  1. Root joint and relative pose keypoints can be improved            #
    #  to make it more representable e.g., adding links between joints etc. #
    #########################################################################
        
    for human_id in range(pose.shape[0]):
        
        
        
        if root_joint is not None:
            cv2.putText(img, "ROOT", root_joint[human_id, :2].astype(int) + np.array([0, 20]), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 5, cv2.LINE_AA)
        
        if pose is not None:
        
            for k in range(1, pose.shape[1]):
                cv2.putText(img, str(k), pose[human_id,  k, :2] + np.array([0, 20]), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2, cv2.LINE_AA)
                
                cv2.circle(img, pose[human_id,  k, :2].tolist(), 5, [255, 0, 0], 5)

    # Probably useless
    if world_origin is not None:
        cv2.circle(img, world_origin[0, :2].tolist(), 5, [0, 0, 255], 5)
        
    return img


if __name__ == "__main__":
    seq_name = "courtyard_golf_00"
    seq_dir = "D:\Saarland\HLCV\project\data\\3DPWPreprocessed\sequenceFiles"
    img_dir = "D:\Saarland\HLCV\project\data\\3DPWPreprocessed\imageFiles"
    folder = "train"
    seq_path = os.path.join(seq_dir, folder, seq_name + ".npz")
    seq = np.load(seq_path, allow_pickle=True)
    
    poses = seq["abs_poses"]
    
    trans = seq["trans"]
    jointPositions = seq["jointPositions"]
    cam_ext = seq["extrinsics"]
    cam_int = seq["intrinsics"]
    img_dir = os.path.join(img_dir, seq_name)    
    img_names = os.listdir(img_dir)

    cv2.namedWindow(f"Padded Resized Image", cv2.WINDOW_KEEPRATIO)  
    cv2.namedWindow(f"Padding Mask", cv2.WINDOW_KEEPRATIO)  
    print(cam_ext.shape)
    print(poses.shape)
    for id in range(poses.shape[1]):
        img, pmask, root_joint, norm_pose, abs_pose = sample(id, poses, img_names, img_dir,  imsize=(480, 480, 3))
        rt = cv2.Rodrigues(cam_ext[id][0:3,0:3])[0].ravel()
        t = cam_ext[id][0:3,3]
        f = np.array([cam_int[0,0],cam_int[1,1]])
        c = cam_int[:,2]


        k = np.zeros(5) 
        camera_mtx = np.array([[f[0], 0, c[0]],[0., f[1], c[1]],[0.,0.,1.]], dtype=np.float64)
        transformed_world_origin, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rt, t, camera_mtx, k)
        transformed_world_origin = np.concatenate([transformed_world_origin[0], np.ones((transformed_world_origin[0].shape[0], 1))], axis=-1)
        # print(np.ones((transformed_world_origin[0].shape[0], 1)).shape)
        # transformed_world_origin = world2camera_transform(np.array([[0, 0, 0, 1]]), cam_ext[id], cam_int).astype(int)
        # print("Root Joint Shape: ", root_joint.shape)
        # print("Normalized Pose Shape: ", norm_pose.shape)
        # print("Absolute Pose Shape: ", abs_pose.shape)
        # print("Padding Mask Shape: ", pmask.shape)
        # print("Padded Resized Image Shape: ", img.shape)
        # print("==========================================")
        rescaled_world_origin = rescale(transformed_world_origin, src_shape=(1920, 1920, 3), target_shape=(480, 480, 3))
        rescaled_world_origin = rescaled_world_origin.astype(int)
        
        img = annotate(img, abs_pose, root_joint, rescaled_world_origin)
        
        cv2.imshow("Padded Resized Image", img)
        cv2.imshow("Padding Mask", pmask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break    
    
    
    
    cv2.destroyAllWindows()