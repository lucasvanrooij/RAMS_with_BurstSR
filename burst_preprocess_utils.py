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
    burstSR = BurstSRDataset(root='C:/Users/Default.DESKTOP-FRPMN5G/Documents/Master/UC/project/burstsr_dataset', split=part, colored_image=colored_image)
    if part == 'train':
        num_images = 2000
    elif part == 'val':
        num_images = 882
    for i in tqdm(range(num_images)):
        # LRs = sorted(glob(imgset+"/samsung*"))
        burst, frame_gt, meta_info_burst, meta_info_gt = burstSR.__getitem__(i)
        # QMs = sorted(glob(imgset+"/QM*.png"))
        T = len(burst)
        LR = np.empty((80,80,T),dtype="uint16")
        QM = np.empty((80,80,T),dtype="bool")
        
        for i,img in enumerate(burst):
            # LR[...,i] = cv2.imread(img,cv2.IMREAD_UNCHANGED)
            LR[...,i] = img
        for i,img in enumerate(burst):
            # QM[...,i] = cv2.imread(img,cv2.IMREAD_UNCHANGED).astype("bool")
            QM[...,i] = img
        X.append(LR)
        X_masks.append(QM)
        if part != "test":
            to_append = frame_gt[...,None]
            y.append(to_append)
            y_masks.append(to_append)

    if part != "test":
        return X,X_masks,np.array(y),np.array(y_masks)
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