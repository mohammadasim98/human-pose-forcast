import os
import cv2

import numpy as np

from builtins import range
from math import sqrt, ceil


def annotate_root(img, root=None, color=(0, 255, 0), thickness=1):
    """Annotate the given image with given the root joint 

    Args:
        img (numpy.ndarray): An input image
        root (numpy.ndarray, optional): An (np, 3) root joints with np as the number of people. Defaults to None.


    Returns:
        numpy.ndarray: An annotated image. 
    """
    
    #########################################################################
    # TODO: Draw root joint                                                 
    #  1. Root joint can be improved to make it more representable e.g.,    
    #  adding links between joints etc.                                     
    #########################################################################
        
    for human_id in range(root.shape[0]):
              
        if root is not None:
            cv2.putText(img, "ROOT", root[human_id, :2].astype(int) + np.array([0, 20]), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)
        
        
    return img


def annotate_pose(img, pose=None, color=(255, 0, 0), radius=5, thickness=1, text=False):
    """Annotate the given image with absolute pose-keypoints.

    Args:
        img (numpy.ndarray): An input image
        pose (numpy.ndarray, optional): An (np, 18, 3) root-relative poses with np as the number of people. Defaults to None.


    Returns:
        numpy.ndarray: An annotated image. 
    """
    
    
    #########################################################################
    # TODO: Draw absolute pose                                              
    #  1. Pose keypoints can be improved to make it more representable      
    #  e.g., adding links between joints etc.                               
    #########################################################################
        
    for human_id in range(pose.shape[0]):
        if pose is not None:
            for k in range(1, pose.shape[1]):
                if pose[human_id,  k, 0] == 0 and pose[human_id,  k, 1] == 0:
                    continue
                
                if text:
                    cv2.putText(img, str(k), pose[human_id,  k, :2] + np.array([0, 20]), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], thickness, cv2.LINE_AA)
                
                cv2.circle(img, pose[human_id,  k, :2].tolist(), radius, color, thickness)
        
    return img

# Useful functions from the excercies

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





