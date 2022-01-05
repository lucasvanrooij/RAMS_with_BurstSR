# Lint as: python
#
# Authors: Vittorio | Francesco
# Location: Turin, Biella, Ivrea
#
# This file is based on the work of Francisco Dorr - PROBA-V-3DWDSR (https://github.com/frandorr/PROBA-V-3DWDSR)

"""Some important functions for preprocessing the original probav-dataset"""
import cv2
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
from scipy.ndimage import shift
from skimage.transform import rescale
from skimage.registration import phase_cross_correlation
from burst_SR import BurstSRDataset
import torch

def load_dataset(base_dir, part, colored_image):
    """
    Load the original BurstSR dataset already splitted in train, validation and test
    
    Parameters
    ----------
    base_dir: str
        path to the original dataset folder
    part: str
        'train', 'val' or test string
    """
    imgsets = sorted(glob(base_dir+"/burstsr_dataset/"+part+"/*"))
    X = []; X_masks = []; y = []; y_masks = []
    burstSR = BurstSRDataset(root='D:/Desktop/Master/Y2Semester1/UC/Project/burstsr_dataset', split=part, colored_image=colored_image)
    if part == 'train':
        num_images = 10
    elif part == 'val':
        num_images = 10
    j=0
    HR_ALL = np.empty((num_images,640,640,1),dtype="uint16")
    for i in tqdm(range(num_images)):
        # LRs = sorted(glob(imgset+"/samsung*"))
        burst, frame_gt, meta_info_burst, meta_info_gt = burstSR.__getitem__(i)
        # QMs = sorted(glob(imgset+"/QM*.png"))
        T = len(burst)
        LR = np.empty((80,80,T),dtype="uint16")
        QM = np.ones((80,80,T),dtype="bool")
        GT = np.ones((640,640,1),dtype="bool")
        for i,img in enumerate(burst):
            # LR[...,i] = cv2.imread(img,cv2.IMREAD_UNCHANGED)
            LR[...,i] = img*255
        # for i,img in enumerate(burst):
        #     # QM[...,i] = cv2.imread(img,cv2.IMREAD_UNCHANGED).astype("bool")
        #     QM[...,i] = img
        X.append(LR)
        X_masks.append(QM)
        if part != "test":
            #HR[] = frame_gt[...,None]
            HR = np.empty((640,640,1),dtype="uint16")
            frame_gt=frame_gt*255
            test = frame_gt[...,None]
            HR = test[:,:,:]
            HR_ALL[j] = HR
            #print(HR_ALL[j])
            j+=1
            y_masks.append(GT)

    if part != "test":
        return X,X_masks,HR_ALL,np.array(y_masks)
    else:
        return X,X_masks

def register_dataset(X, masks):
    """
    Register the input tensor X of shape (B, H, W, T) with respect to the image with the best quality map
    
    Parameters
    ----------
    X: numpy array
        tensor X to register
    masks: numpy array
        tensor with the quality maps of X
    """
    X_reg = []
    masks_reg = []
    
    for i in tqdm(range(len(X))):
        img_reg,m_reg = register_imgset(X[i], masks[i])
        X_reg.append(img_reg)
        masks_reg.append(m_reg)
    
    return X_reg,masks_reg