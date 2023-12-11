### This file is adapted from https://github.com/AnyLoc/AnyLoc/blob/main/custom_datasets/gardens.py

import natsort
import os
import numpy as np
from PIL import Image


"""
Custom class representation for
Gardens point dataset
    From: https://zenodo.org/record/4590133#.ZAdKktJBxH5
    960 x 540 resolution
"""

# from utilities import CustomDataset

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")

# base_transform = T.Compose([
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# mixVPR_transform = T.Compose([
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#     T.Resize((320,320))
# ])

class Gardens():
    """
    Returns dataset class with images from database and queries for the gardens dataset.
    """
    def __init__(self,datasets_folder='./gardens',dataset_name="gardens",split="train",use_ang_positives=False,dist_thresh = 10,ang_thresh=20,use_mixVPR=False,use_SAM=False):
        # super().__init__()

        self.dataset_name = dataset_name
        self.datasets_folder = datasets_folder
        self.split = split
        self.use_mixVPR = use_mixVPR
        self.use_SAM = use_SAM
        self.db_paths = natsort.natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"day_right")))
        self.q_paths = natsort.natsorted(os.listdir(os.path.join(self.datasets_folder,self.dataset_name,"day_left")))

        self.db_abs_paths = [] # using day_right as database
        self.q_abs_paths_day_left = []
        self.q_abs_paths_night_right = []
        self.q_abs_paths = self.q_abs_paths_day_left + self.q_abs_paths_night_right

        for p in self.db_paths:
            self.db_abs_paths.append(os.path.join(self.datasets_folder,self.dataset_name,"day_right",p))

        for q in self.q_paths:
            self.q_abs_paths_night_right.append(os.path.join(self.datasets_folder,self.dataset_name,"night_right",q))
            self.q_abs_paths_day_left.append(os.path.join(self.datasets_folder,self.dataset_name,"day_left",q))

        self.q_abs_paths = self.q_abs_paths_day_left + self.q_abs_paths_night_right


        self.db_num = len(self.db_abs_paths)
        self.q_num = len(self.q_abs_paths)
        self.q_num_night_right = len(self.q_abs_paths_night_right)
        self.q_num_day_left = len(self.q_abs_paths_day_left)


        self.database_num = self.db_num
        self.queries_num = self.q_num

        self.gt_positives = np.load(os.path.join(self.datasets_folder,self.dataset_name,"gardens_gt.npy"),allow_pickle=True) # returns dictionary of gardens dataset ground truth values

        self.images_paths = list(self.db_abs_paths) + list(self.q_abs_paths)

        self.soft_positives_per_query = []

        for i in range(len(self.gt_positives)):
            self.soft_positives_per_query.append(self.gt_positives[i][1]) # corresponds to index wise soft positives

    def show_day_right_item(self, index):
        img = Image.open(self.db_abs_paths[index])
        return img

    def show_day_left_item(self, index):
        img = Image.open(self.q_abs_paths_day_left[index])
        return img

    def show_night_right_item(self, index):
        img = Image.open(self.q_abs_paths_night_right[index])
        return img
