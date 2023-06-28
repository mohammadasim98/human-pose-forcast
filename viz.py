import os
import pickle as pkl
import numpy as np
import cv2
import time
from utils import sample, rescale
from transformations import world2camera_transform




def annotate(img, pose=None, root_joint=None, world_origin=None):
    
    for human_id in range(pose.shape[0]):
        
        if root_joint is not None:
            cv2.putText(img, "ROOT", root_joint[human_id, :2].astype(int) + np.array([0, 20]), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 5, cv2.LINE_AA)
        
        if pose is not None:
        
            for k in range(1, pose.shape[1]):
                cv2.putText(img, str(k), pose[human_id,  k, :2] + np.array([0, 20]), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2, cv2.LINE_AA)
                
                cv2.circle(img, pose[human_id,  k, :2].tolist(), 5, [255, 0, 0], 5)
                
    if world_origin is not None:
        cv2.circle(img, world_origin[0, :2].tolist(), 5, [0, 0, 255], 5)
        
    return img
  
if __name__ == "__main__":
    seq_name = "courtyard_golf_00"
    seq_dir = "D:\Saarland\HLCV\project\data\\3DPW\sequenceFilesTransformed"
    img_dir = "D:\Saarland\HLCV\project\data\\3DPW\imageFiles"
    folder = "train"
    seq_path = os.path.join(seq_dir, folder, seq_name + ".npz")
    seq = np.load(seq_path, allow_pickle=True)
    
    ids = seq["img_frame_ids"]
    poses = seq["poses2d"]
    trans = seq["trans"]
    jointPositions = seq["jointPositions"]
    cam_ext = seq["cam_poses"]
    cam_int = seq["cam_intrinsics"]
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
