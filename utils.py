import os
import cv2
import numpy as np
import pickle as pkl

def get_root_joint(root_candidates: list):
    """Generate root joint from candidates joints for all people.

    Args:
        root_candidates (list): A list of numpy.ndarray of shape (np, 3) as candidates with np as the number of people.

    Returns:
        numpy.ndarray: A (np, 3) root joint with np number of people.
    """
    root_candidates = np.array(root_candidates)
    root_joint = np.sum(root_candidates, axis=0)
    counts = np.sum((root_candidates != 0).max(-1), axis=0)
    counts = np.expand_dims(counts, axis=-1)
    counts = np.tile(counts, (1, 3))

    
    root_joint = np.divide(root_joint, counts, where=counts!=0)
    
    return root_joint

def cvt_root_relative(root_joint, pose: np.ndarray, eps: float=0.0001):
    """Convert absolute pose to root relative pose given a root joint.
                              (abs_pose - root)
             norm_pose = ---------------------------
                         (abs_pose + root + epsilon)
    Args:
        root_joint (numpy.ndarray): A (np, 3) numpy array with np as number of people.
        pose (np.ndarray): A (np, 18, 3) numpy array with np as number of people.
        eps (float, optional): A floating value for numerical stability. Defaults to 0.0001.

    Returns:
        numpy.ndarray: A (np, 18, 3) root relative pose with np number of people.
    """
    root_joint = np.expand_dims(root_joint, axis=1)
    root_joint = np.tile(root_joint, (1, pose.shape[1], 1))
    return ((pose - root_joint) / (root_joint+pose+eps))

def cvt_absolute_pose(root_joint, norm_pose: np.ndarray):
    """Convert root relative pose to absolute pose given a root joint.
                                (1 + norm_pose)
              abs_pose = root X ---------------
                                (1 - norm_pose)
                         
    Args:
        root_joint (numpy.ndarray): A (np, 3) numpy array with np as number of people.
        norm_pose (np.ndarray): A (np, 18, 3) numpy array with np as number of people.
                              
    Returns:
        numpy.ndarray: A (np, 18, 3) absolute pose with np number of people.
    """
    root_joint = np.expand_dims(root_joint, axis=1)
    root_joint = np.tile(root_joint, (1, norm_pose.shape[1], 1))
    return ((1 + norm_pose) * root_joint / (1 - norm_pose)).astype(int)

def pad_till_square(img):
    """Perform square padding on the image and bring it to its largest size.

    Args:
        img (numpy.ndarray): A (H, W, 3) numpy array.

    Returns:
        numpy.ndarray: Square padded image.
        numpy.ndarray: Square padding mask.
    """
    max_size = max(img.shape)
    target_shape = np.array((max_size, max_size, img.shape[2]))
    diff = target_shape - np.array(img.shape)
    padded = np.pad(img, ((0, diff[0]), (0, diff[1]), (0, 0)), mode='constant')
    mask = np.zeros(img.shape)
    mask =  np.pad(mask, ((0, diff[0]), (0, diff[1]), (0, 0)), mode='constant', constant_values=(1))
    return padded, mask

def pad(img, target_shape):
    """Perform padding on the image.

    Args:
        img (numpy.ndarray): A (H, W, 3) numpy array.
        target_shape (tuple, optional): Target shape after padding.

    Returns:
        numpy.ndarray: Padded image.
        numpy.ndarray: Padding mask.
    """
    assert target_shape[0] >= img.shape[0], "negative padding on x-axis not possible"
    assert target_shape[1] >= img.shape[1], "negative padding on y-axis not possible"

    diff = np.array(target_shape) - np.array(img.shape)
    padded = np.pad(img, ((0, diff[0]), (0, diff[1]), (0, 0)), mode='constant')
    mask = np.zeros(img.shape)
    mask =  np.pad(mask, ((0, diff[0]), (0, diff[1]), (0, 0)), mode='constant', constant_values=(1))
    return padded, mask

def rescale(position, src_shape, target_shape=(640, 480, 3)):
    
    factor = np.array(target_shape) / np.array(src_shape)
    factor = np.array([*factor[:2][::-1], factor[2]])
    factor = np.tile(np.expand_dims(factor, axis=(0)), (position.shape[0], 1))
    rescaled_position = position*factor
    
    return rescaled_position

def resize_rescale(img, mask, pose, target_shape=(640, 480, 3)):
    """Resize image, mask and rescale pose given a target shape.

    Args:
        img (numpy.ndarray): A (W, H, 3) numpy array.
        mask (numpy.ndarray): A (W, H, 3) numpy array.
        pose (numpy.ndarray): A (np, 18, 3) numpy array.
        target_shape (tuple, optional): Format (H, W, C) similar to cv2.resize(). Defaults to (640, 480, 3).

    Returns:
        numpy.ndarray: Resized image
        numpy.ndarray: Resized mask
        numpy.ndarray: A (np, 18, 3) rescaled pose with np number of people.
    """
    factor = np.array(target_shape) / np.array(img.shape)
    rescaled_img = cv2.resize(img, target_shape[:2][::-1])
    rescaled_pose = pose*np.tile(np.expand_dims([*factor[:2][::-1], factor[2]], axis=(0,1)), (1, 18, 1))
    rescaled_mask = cv2.resize(mask, target_shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    return rescaled_img, rescaled_mask, rescaled_pose.astype(int)

def sample(id, poses, img_paths, dir, imsize=(480, 480, 3), psize=None ):
    """Get a sample from a list of image paths and poses.

    Args:
        id (int): Frame id.
        poses (numpy.ndarray): A (np, nf, 18, 3) pose array with np ad the number of people, nf as the number of frames.  
        img_paths (list): A list of image paths belonging to a particular sequence
        dir (str): Image directory
        imsize (tuple, optional): Final padded and resized image shape. Defaults to (480, 480, 3).
        psize (tuple, optional): If none, then uses square padding using the largest dimension of the image otherwise uses the given shape for padding. Defaults to None.

    Returns:
        numpy.ndarray: Padded, and resized image frame. 
        numpy.ndarray: Corresponding resized padding mask. 
        numpy.ndarray: A (np, 3) rescaled root joint.
        numpy.ndarray: A (np, 18, 3) normalized pose relative to the root joint with np number of people.
        numpy.ndarray: A (np, 18, 3) rescaled absolute pose with np number of people.
    """
    pose = (poses[:, id, :, :]).astype(int)
    # pose = np.transpose(pose, (0, 2, 1))

    img_path = os.path.join(dir, img_paths[id])
    img = cv2.imread(img_path)
    
    # start = time.time()
    if psize is None: 
        img, pmask = pad_till_square(img)
    else:
        img, pmask = pad(img, psize)

    img, pmask, pose = resize_rescale(img, pmask, pose, imsize)

    # print("time took: ", (time.time() - start), flush=True)
    root_candidates = [pose[:, 8, :], pose[:, 11, :], pose[:, 5, :], pose[:, 2, :], pose[:, 1, :]]
    root_joint = get_root_joint(root_candidates)
    
    norm_pose = cvt_root_relative(root_joint, pose)
    abs_pose = cvt_absolute_pose(root_joint, norm_pose)

    return img, pmask, root_joint, norm_pose, abs_pose



def cvt_pkl2npz(datasetDir='D:\Saarland\HLCV\project\data\\3DPW\sequenceFiles', transformDir="D:\Saarland\HLCV\project\data\\3DPW\sequenceFilesTransformed"):
    for folder in os.listdir(datasetDir):
        folder_path = os.path.join(datasetDir, folder)
        outpath = os.path.join(transformDir, folder)
        files = os.listdir(folder_path)
        print(files)
        for file in files:
            file_path = os.path.join(folder_path, file)
            f = open(file_path,'rb')
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()

            np.savez(f"{os.path.join(outpath, file.split('.')[0])}.npz", **p)