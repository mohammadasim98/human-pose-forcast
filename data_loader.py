import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import glob
import cv2

class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class ThreeDPWDataset(Dataset):
    def __init__(self, root_dir="D:/Saarland/HLCV/project/data\\3DPW", transform=None, batch_size = 32, window_size = 10, type="train"):
        
        # self.labels = []
        self.poses = []
        self.batch_size = batch_size
        self.window_size = window_size
        
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "imageFiles")
        self.annotation_dir = os.path.join(root_dir, "sequenceFilesTransformed", type)
        
        self.img_paths = []
        self.seq_lens = []
        for i, file_name in enumerate(os.listdir(self.annotation_dir)):
            file_path = os.path.join(self.annotation_dir, file_name)
            seq = np.load(file_path, allow_pickle=True)
            folder_path = os.path.join(self.img_dir, file_name.split(".")[0])

            self.seq_lens.append(seq["poses2d"].shape[1])
            for i in range(seq["poses2d"].shape[1]):
                self.poses.append(np.reshape(seq["poses2d"][0, i, :, :], (18, 3)))
                
            for img_path in os.listdir(folder_path):
                self.img_paths.append(os.path.join(folder_path, img_path))
        
        assert len(self.img_paths) == len(self.poses), " Counts are invalid"
        # for item in zip(self.img_paths, self.poses, self.seq_lens):
        #     assert len(item[0]) == item[1].shape[1] == item[2], "Sequence lengths are invalid"
        
        print(len(self.img_paths))
        print(len(self.poses))
        
        self.seq_endings = np.cumsum(self.seq_lens) - window_size
        print(self.seq_endings)
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.window_size
        if start_idx in self.seq_endings:
            start_idx += self.window_size
        imgs = []
        for i in range(start_idx, end_idx+1):
            imgs.append(cv2.resize(cv2.imread(self.img_paths[i]), ()))
        
        return np.array(imgs), label
    
    
if __name__ == "__main__":
    

    dataloader = ThreeDPWDataset()