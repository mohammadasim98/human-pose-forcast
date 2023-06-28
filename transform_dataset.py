import os
import pickle as pkl
import numpy as np
import cv2
import json
from utils import sample, cvt_pkl2npz
from viz_dataset import display
from multiprocessing import Process
import multiprocessing
from glob import glob

def write_sequence(seq_path):
    seq_path_splitted = seq_path.split("\\")
    datasetDir = os.path.join(".", 'data', '3DPW')
    transformDir = os.path.join('.', 'data', '3DPWPreprocessed',)
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
        
        # print("Root Joint Shape: ", root_joint.shape)
        # print("Normalized Pose Shape: ", norm_pose.shape)
        # print("Absolute Pose Shape: ", abs_pose.shape)
        # print("Padding Mask Shape: ", pmask.shape)
        # print("Padded Resized Image Shape: ", img.shape)
        # print("==========================================")
        
    

        # display(img, pmask, root_joint, abs_pose)
            
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break  
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
        "pmask": pmask, \
        "height": img.shape[0], 
        "width": img.shape[1]}
    # print("Root Joint Shape: ", root_joints.shape)
    # print("Normalized Pose Shape: ", norm_poses.shape)
    # print("Absolute Pose Shape: ", abs_poses.shape)
    np.savez(os.path.join(transformDir, "sequenceFiles", seq_path_splitted[4], sequence_file + ".npz"), **p)


if __name__ == '__main__':
    global img_dir, seq_folder, seq_dir
    
    datasetDir = os.path.join(".", 'data', '3DPW')
    transformDir = "D:\Saarland\HLCV\project\data\\3DPWPreprocessed"
    
    seq_dir = os.path.join(datasetDir, "sequenceFilesTransformed")
    sequence_folders = os.listdir(seq_dir)
    img_dir = os.path.join(datasetDir, "imageFiles")
    
    seq_paths = glob(os.path.join(seq_dir, "*", "*.npz"), )
    with multiprocessing.Pool() as pool:
        # call the function for each item in parallel
        pool.map(write_sequence, seq_paths)
    # for sequence_folder in sequence_folders:
    #     seq_folder = sequence_folder
    #     sequence_files = os.listdir(os.path.join(seq_dir, seq_folder)) 
           
    #     # for sequence_file in sequence_files:
    #     #     write_sequence(seq_dir, sequence_file, sequence_folder, img_dir)
        
                
                
    cv2.destroyAllWindows()
   
    
    


    
        