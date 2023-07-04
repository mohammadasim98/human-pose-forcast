import os
import pickle as pkl
import numpy as np
import cv2
import json
from transforms import sample
from multiprocessing import Process
import multiprocessing
from glob import glob
import ffmpeg
import tfrecord

def write_sequence(seq_path):
    seq_path_splitted = seq_path.split("\\")
    datasetDir = os.path.join("..", "..", 'data', '3DPW')
    transformDir = os.path.join("..", "..", 'data', '3DPWPreprocessed',)
    img_dir = os.path.join(datasetDir, "imageFiles")
    
    sequence_file = seq_path_splitted[-1].split(".")[0]
    
    seq = np.load(seq_path, allow_pickle=True)
    
    poses = seq["poses2d"]
    trans = seq["trans"]
    jointPositions = seq["jointPositions"]
    cam_exts = seq["cam_poses"]
    cam_int = seq["cam_intrinsics"]
    
    img_sequence_dir = os.path.join(img_dir, sequence_file)    
    img_names = os.listdir(img_sequence_dir)
    
    print(f"Writing Sequence: {img_sequence_dir}")
    
    norm_poses = []
    abs_poses = []
    root_joints = []
    
    for id in range(poses.shape[1]):
        img, pmask, root_joint, norm_pose, abs_pose = sample(id, poses, img_names, img_sequence_dir, imsize=(384, 384, 3))
        if not os.path.exists(os.path.join(transformDir, "imageFiles", sequence_file)):
            os.mkdir(os.path.join(transformDir, "imageFiles", sequence_file))
            
        if not os.path.exists(os.path.join(transformDir, "paddingMaskFiles", sequence_file)):
            os.mkdir(os.path.join(transformDir, "paddingMaskFiles", sequence_file))
        
        new_img_path = os.path.join(transformDir, "imageFiles", sequence_file, img_names[id])
        new_mask_path = os.path.join(transformDir, "paddingMaskFiles", sequence_file, "mask.jpg")
        
        cv2.imwrite(new_img_path, img)
        
        root_joints.append(root_joint.tolist())
        norm_poses.append(norm_pose.tolist())
        abs_poses.append(abs_pose.tolist())
        

    cv2.imwrite(new_mask_path, pmask*255)

    norm_poses = np.transpose(np.array(norm_poses), (1, 0, 2, 3))
    abs_poses = np.transpose(np.array(abs_poses), (1, 0, 2, 3))
    root_joints = np.transpose(np.array(root_joints), (1, 0, 2))
    p = {
        "seq_name": seq["sequence"], 
        "norm_poses": norm_poses, 
        "abs_poses": abs_poses, 
        "root_joints": root_joints, 
        "intrinsics": cam_int, 
        "extrinsics": cam_exts, 
        "trans": trans, 
        "jointPositions": jointPositions, 
        "pmask": pmask, 
        "height": img.shape[0], 
        "width": img.shape[1]}

    np.savez(os.path.join(transformDir, "sequenceFiles", seq_path_splitted[4], sequence_file + ".npz"), **p)

def preprocess():
    datasetDir = os.path.join("..", "..", 'data', '3DPW')
    transformDir = os.path.join("..", "..", 'data', '3DPWPreprocessed')
    
    seq_dir = os.path.join(datasetDir, "sequenceFilesTransformed")
    sequence_folders = os.listdir(seq_dir)
    img_dir = os.path.join(datasetDir, "imageFiles")
    
    seq_paths = glob(os.path.join(seq_dir, "*", "*.npz"), )
    with multiprocessing.Pool() as pool:
        # call the function for each item in parallel
        pool.map(write_sequence, seq_paths)

def cvt_images2video():
    cwd = os.getcwd() 
    datasetDir = os.path.join(cwd, 'data', '3DPWPreprocessed') 
    sequence_path = os.path.join(datasetDir, "imageFiles")
    sequence_folders = os.listdir(sequence_path)
    sequence_list = [os.path.join(sequence_path, folder, "image_%05d.jpg") for folder in sequence_folders]
    
    out_path = os.path.join(datasetDir, "videos")
    for path, folder in zip(sequence_list, sequence_folders):
        (
            ffmpeg
            .input(path, framerate=30)
            .output(f'{os.path.join(out_path, folder)}.mp4')
            .run()
        )


if __name__ == '__main__':
    
    pass