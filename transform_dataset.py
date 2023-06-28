import os
import pickle as pkl
import numpy as np
import cv2
import json
from utils import sample, cvt_pkl2npz
from viz_dataset import display






if __name__ == '__main__':
    
    datasetDir = 'D:\Saarland\HLCV\project\data\\3DPW'
    transformDir = "D:\Saarland\HLCV\project\data\\3DPWPreprocessed"
    
    seq_name = "outdoors_freestyle_00"
    seq_dir = os.path.join(datasetDir, "sequenceFilesTransformed")
    sequence_folders = os.listdir(seq_dir)
    img_dir = os.path.join(datasetDir, "imageFiles")
    
    for sequence_folder in sequence_folders:
        sequence_files = os.listdir(os.path.join(seq_dir, sequence_folder)) 
           
        for sequence_file in sequence_files:
            seq_path = os.path.join(seq_dir, sequence_folder, sequence_file)
            seq = np.load(seq_path, allow_pickle=True)
            ids = seq["img_frame_ids"]
            poses = seq["poses2d"]
            trans = seq["trans"]
            jointPositions = seq["jointPositions"]
            cam_exts = seq["cam_poses"]
            cam_int = seq["cam_intrinsics"]
            img_sequence_dir = os.path.join(img_dir, sequence_file.split(".")[0])    
            img_names = os.listdir(img_sequence_dir)
            print(img_sequence_dir)
            norm_poses = []
            abs_poses = []
            root_joints = []
            for id in range(poses.shape[1]):
                img, pmask, root_joint, norm_pose, abs_pose = sample(id, poses, img_names, img_sequence_dir, imsize=(480, 480, 3))
                if not os.path.exists(os.path.join(transformDir, "imageFiles", sequence_file.split(".")[0])):
                    os.mkdir(os.path.join(transformDir, "imageFiles", sequence_file.split(".")[0]))
                    
                if not os.path.exists(os.path.join(transformDir, "paddingMaskFiles", sequence_file.split(".")[0])):
                    os.mkdir(os.path.join(transformDir, "paddingMaskFiles", sequence_file.split(".")[0]))
                
                new_img_path = os.path.join(transformDir, "imageFiles", sequence_file.split(".")[0], img_names[id] + ".jpg")
                new_mask_path = os.path.join(transformDir, "paddingMaskFiles", sequence_file.split(".")[0], img_names[id] + ".jpg")
                
                cv2.imwrite(new_img_path, img)
                cv2.imwrite(new_mask_path, pmask*255)
                
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
                  
            norm_poses = np.transpose(np.array(norm_poses), (1, 0, 2, 3))
            abs_poses = np.transpose(np.array(abs_poses), (1, 0, 2, 3))
            root_joints = np.transpose(np.array(root_joints), (1, 0, 2))
            p = {"seq_name": seq["sequence"], "norm_poses": norm_poses, "abs_poses": abs_poses, "root_joints": root_joints, "intrinsics": cam_int, "extrinsics": cam_exts, "trans": trans, "jointPositions": jointPositions}
            print("Root Joint Shape: ", root_joints.shape)
            print("Normalized Pose Shape: ", norm_poses.shape)
            print("Absolute Pose Shape: ", abs_poses.shape)
            np.savez(os.path.join(transformDir, "sequenceFiles", sequence_folder, sequence_file), **p)
                
                
                
    cv2.destroyAllWindows()
   
    
    


    
        