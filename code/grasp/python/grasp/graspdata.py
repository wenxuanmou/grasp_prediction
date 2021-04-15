'''
facedata.py in DeepFaceAlignment

author  : wmou, tmarks, modified based on pointoctnet of cfeng
created : 2/12/18 4:20 AM
'''

import os
import sys
import argparse
import glog as logger
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as scio
from skimage import io
from matplotlib import pyplot as plt
import cv2
import pdb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True


class graspDataset(Dataset): 

    def __init__(self, root_dir, file_path, imSize = 256, left_right='left',fraction_jitter = 0.0, \
                 fraction_flip = 0.0, fraction_grayscale = 0.0, fraction_rotated = 0.0, max_rot_deg = 0, \
                 max_scale = 1, min_scale = 1, max_translation = 0, shuffle=False):
        self.imPath = np.load(file_path) #imPath
        self.root_dir = root_dir
        self.left_right=left_right

        self.fraction_rotated = fraction_rotated
        self.max_rot_deg = max_rot_deg
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.max_translation = max_translation # max_translation is a fraction of y_max- y_min
        self.fraction_flip = fraction_flip
        self.fraction_jitter = fraction_jitter
        self.fraction_grayscale = fraction_grayscale
        # set the image size
        self.imSize = imSize   # this can be used to change the size of the image before feeding to the network, but I did not use it here
        self.file_path=file_path


    def __len__(self):
        return len(self.imPath)

    def __getitem__(self, idx):

        im_org = io.imread(os.path.join(self.root_dir, self.imPath[idx]))  # read the image
        #print('im_org path ', os.path.join(self.root_dir, self.imPath[idx]))
        # split left and right images
        h, w, c = np.shape(im_org)
        im_left = im_org[:, :w // 2, :]
        im_right = im_org[:, w // 2:, :]

        label_path='/'.join(self.imPath[idx].split('/')[:-1])+'/grasp_label.npy'
        label=np.load(os.path.join(self.root_dir, label_path))
        if label==1:
            label = np.ones((1, 1), dtype=int)
        else:
            label = np.zeros((1, 1), dtype=int)

        if self.left_right=='right':
            im = im_right

            img = np.zeros([3,im.shape[0],im.shape[1]])
            img[0,:,:] = im[:,:,0]
            img[1,:,:] = im[:,:,1]
            img[2,:,:] = im[:,:,2]

            imNorm = np.zeros([3,im.shape[0],im.shape[1]])
            imNorm[0, :, :] = (img[0,:,:] - np.max(img[0,:,:]))/(np.max(img[0,:,:])-np.min(img[0,:,:])) -0.5
            imNorm[1, :, :] = (img[1,:,:] - np.max(img[1,:,:]))/(np.max(img[1,:,:])-np.min(img[1,:,:])) -0.5
            imNorm[2, :, :] = (img[2,:,:] - np.max(img[2,:,:]))/(np.max(img[2,:,:])-np.min(img[2,:,:])) -0.5
        if self.left_right=='left':
            im=im_left

            img = np.zeros([3,im.shape[0],im.shape[1]])
            img[0,:,:] = im[:,:,0]
            img[1,:,:] = im[:,:,1]
            img[2,:,:] = im[:,:,2]

            imNorm = np.zeros([3,im.shape[0],im.shape[1]])
            imNorm[0, :, :] = (img[0,:,:] - np.max(img[0,:,:]))/(np.max(img[0,:,:])-np.min(img[0,:,:])) -0.5
            imNorm[1, :, :] = (img[1,:,:] - np.max(img[1,:,:]))/(np.max(img[1,:,:])-np.min(img[1,:,:])) -0.5
            imNorm[2, :, :] = (img[2,:,:] - np.max(img[2,:,:]))/(np.max(img[2,:,:])-np.min(img[2,:,:])) -0.5
        if self.left_right =='both':
            im=im_org

            img = np.zeros([3,im.shape[0],im.shape[1]])
            img[0,:,:] = im[:,:,0]
            img[1,:,:] = im[:,:,1]
            img[2,:,:] = im[:,:,2]

            imNorm = np.zeros([3,im.shape[0],im.shape[1]])
            imNorm[0, :, :] = (img[0,:,:] - np.max(img[0,:,:]))/(np.max(img[0,:,:])-np.min(img[0,:,:])) -0.5
            imNorm[1, :, :] = (img[1,:,:] - np.max(img[1,:,:]))/(np.max(img[1,:,:])-np.min(img[1,:,:])) -0.5
            imNorm[2, :, :] = (img[2,:,:] - np.max(img[2,:,:]))/(np.max(img[2,:,:])-np.min(img[2,:,:])) -0.5

        if self.left_right =='both2':
            img_left = np.zeros([3,im_left.shape[0],im_left.shape[1]])
            img_left[0,:,:] = im_left[:,:,0]
            img_left[1,:,:] = im_left[:,:,1]
            img_left[2,:,:] = im_left[:,:,2]

            imNorm_left = np.zeros([3,im_left.shape[0],im_left.shape[1]])
            imNorm_left[0, :, :] = (img_left[0,:,:] - np.max(img_left[0,:,:]))/(np.max(img_left[0,:,:])-np.min(img_left[0,:,:])) -0.5
            imNorm_left[1, :, :] = (img_left[1,:,:] - np.max(img_left[1,:,:]))/(np.max(img_left[1,:,:])-np.min(img_left[1,:,:])) -0.5
            imNorm_left[2, :, :] = (img_left[2,:,:] - np.max(img_left[2,:,:]))/(np.max(img_left[2,:,:])-np.min(img_left[2,:,:])) -0.5


            img_right = np.zeros([3,im_right.shape[0],im_right.shape[1]])
            img_right[0,:,:] = im_right[:,:,0]
            img_right[1,:,:] = im_right[:,:,1]
            img_right[2,:,:] = im_right[:,:,2]

            imNorm_right = np.zeros([3,im_right.shape[0],im_right.shape[1]])
            imNorm_right[0, :, :] = (img_right[0,:,:] - np.max(img_right[0,:,:]))/(np.max(img_right[0,:,:])-np.min(img_right[0,:,:])) -0.5
            imNorm_right[1, :, :] = (img_right[1,:,:] - np.max(img_right[1,:,:]))/(np.max(img_right[1,:,:])-np.min(img_right[1,:,:])) -0.5
            imNorm_right[2, :, :] = (img_right[2,:,:] - np.max(img_right[2,:,:]))/(np.max(img_right[2,:,:])-np.min(img_right[2,:,:])) -0.5

            
            imNorm = np.concatenate((imNorm_left,img_right))

        

        return{
            #'im': im,                           # original image
            #'img': img.astype(np.float32),      #image in a size of 3 x imSize x imSize
            'imNorm': imNorm.astype(np.float32), # normalized image
            'label':np.transpose(label.astype(np.float32)),            
            'filename': self.imPath[idx]        # filename
            }


# script_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # the path of the running file /homes/wmou/Desktop/H/FaceAlignment/code/DeepFaceAlignment/python
# default_path = os.path.join(script_folder, 'grasp_list.npy') #  train_list_300WLP, list_300W_train.npy
# root_dir = os.path.dirname(os.path.dirname(script_folder)) +'/data'
        
# #print(root_dir)

# test=faceDataset(root_dir, file_path=default_path, imSize = 256, fraction_jitter = 1.0, fraction_flip=0.5,fraction_rotated = 0.4, max_rot_deg = 50,max_scale = 1.2, min_scale = 0.8, max_translation = 0.1)

# test.__getitem__(0)
# pdb.set_trace()



class DefaultTrainSet(graspDataset):
    def __init__(self, **kwargs):
        script_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # the path of the running file .../python
        default_path = os.path.join(script_folder, 'data_path/5folder/klist_train1.npy') # the list of the image paths
    
        root_dir = os.path.dirname(os.path.dirname(script_folder)) +'/data'
        super(DefaultTrainSetRGBAndGray, self).__init__(root_dir, file_path=default_path, left_right='both2',**kwargs)


class DefaultTestSet(graspDataset):

    def __init__(self, **kwargs):
        script_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_path = os.path.join(script_folder, 'data_path/5folder/klist1.npy')  
        root_dir = os.path.dirname(os.path.dirname(script_folder)) +'/data'
        super(DefaultTestSet, self).__init__(root_dir, file_path=default_path, left_right='both2',**kwargs)





