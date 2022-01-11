import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.preprocessing import gen_sub, bicubic
from utils.loss import l1_loss, psnr, ssim
from utils.network import RAMS
from utils.training import Trainer
from skimage import io
from zipfile import ZipFile
from utils.prediction import ensemble, unensemble, shuffle_last_axis, predict_tensor, predict_tensor_permute, savePredictions, savePredictionsPermut

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)


#-------------
# General Settings
#-------------
PATH_DATASET = 'dataset' # pre-processed dataset path
name_net = 'RAMS_w_BURSTSR' # name of the network
LR_SIZE = 128 # pathces dimension
SCALE = 3 # upscale of the proba-v dataset is 3
HR_SIZE = LR_SIZE * SCALE # upscale of the dataset is 3
OVERLAP = 32 # overlap between pathces
CLEAN_PATH_PX = 0.0 # percentage of clean pixels to accept a patch
band = 'burst' # choose the band for the training
checkpoint_dir = 'ckpt/burst_RAMS_w_BURSTSR_retrain' # weights path
log_dir = 'logs' # tensorboard logs path
submission_dir = 'submission' # submission dir

#-------------
# Network Settings
#-------------
FILTERS = 32 # features map in the network
KERNEL_SIZE = 3 # convolutional kernel size dimension (either 3D and 2D)
CHANNELS = 9 # number of temporal steps
R = 8 # attention compression
N = 12 # number of residual feature attention blocks
lr = 1e-4 # learning rate (Nadam optimizer)
BATCH_SIZE = 4 # batch size
EPOCHS_N = 300 # number of epochs

# create logs folder
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    
dataset_size = 100
X_train = np.load(os.path.join(PATH_DATASET, 'X_burst_train.npy'))[0:dataset_size]
y_train = np.load(os.path.join(PATH_DATASET, 'y_burst_train.npy'),allow_pickle=True)[0:dataset_size]
y_train_mask = np.load(os.path.join(PATH_DATASET, 'y_burst_train_masks.npy'),allow_pickle=True)[0:dataset_size]

# load validation dataset
X_val = np.load(os.path.join(PATH_DATASET, 'X_burst_val.npy'))[0:dataset_size]
y_val = np.load(os.path.join(PATH_DATASET, 'y_burst_val.npy'), allow_pickle=True)[0:dataset_size]
y_val_mask = np.load(os.path.join(PATH_DATASET, 'y_burst_val_masks.npy'), allow_pickle=True)[0:dataset_size]


# print loaded dataset info
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('y_train_mask: ', y_train_mask.shape)


print('X_val: ', X_val.shape)
print('y_val: ', y_val.shape)
print('y_val_mask: ', y_val_mask.shape)

# create patches for LR images
d = LR_SIZE  # 32x32 patches
s = OVERLAP  # overlapping patches
# Ex: n = (128-d)/s+1 = 7 -> 49 sub images from each image 
X_train_patches = gen_sub(X_train,d,s)
X_val_patches = gen_sub(X_val,d,s)


# create patches for HR images and masks
d = HR_SIZE  # 96x96 patches
s = OVERLAP * SCALE  # overlapping patches
# Ex: n = (384-d)/s+1 = 7 -> 49 sub images from each image

y_train_patches = gen_sub(y_train,d,s)
y_train_mask_patches = gen_sub(y_train_mask,d,s)


y_val_patches = gen_sub(y_val,d,s)
y_val_mask_patches = gen_sub(y_val_mask,d,s)

# free up memory
del X_train, y_train, y_train_mask

del X_val, y_val, y_val_mask

patches_to_remove_train = [i for i,m in enumerate(y_train_mask_patches) if np.count_nonzero(m)/(HR_SIZE*HR_SIZE) < CLEAN_PATH_PX]
patches_to_remove_val = [i for i,m in enumerate(y_val_mask_patches) if np.count_nonzero(m)/(HR_SIZE*HR_SIZE) < CLEAN_PATH_PX]

X_train_patches = np.delete(X_train_patches,patches_to_remove_train,axis=0)
y_train_patches =  np.delete(y_train_patches,patches_to_remove_train,axis=0)
y_train_mask_patches =  np.delete(y_train_mask_patches,patches_to_remove_train,axis=0)

X_val_patches = np.delete(X_val_patches,patches_to_remove_val,axis=0)
y_val_patches =  np.delete(y_val_patches,patches_to_remove_val,axis=0)
y_val_mask_patches =  np.delete(y_val_mask_patches,patches_to_remove_val,axis=0)

rams_network = RAMS(scale=SCALE, filters=FILTERS, kernel_size=KERNEL_SIZE, channels=CHANNELS, r=R, N=N)

trainer_rams = Trainer(rams_network, band, HR_SIZE, name_net, loss=l1_loss, metric=psnr, optimizer=tf.keras.optimizers.Nadam(learning_rate=lr), checkpoint_dir=os.path.join(checkpoint_dir), log_dir=log_dir)
                      
trainer_rams.fit(X_train_patches, [y_train_patches.astype('float32'), y_train_mask_patches], initial_epoch = 0, batch_size=BATCH_SIZE, evaluate_every=400, data_aug = True, epochs=EPOCHS_N, validation_data=(X_val_patches, [y_val_patches.astype('float32'), y_val_mask_patches])) 

index = 2 # choose an image from validation set

x_pred = predict_tensor(rams_network, X_val[index:index+1])

fig, ax = plt.subplots(2,2, figsize=(15,15))
ax[0,0].imshow(X_val[index,:,:,0], cmap = 'gray')
ax[0,0].set_title('LR')
ax[0,1].imshow(bicubic(np.mean(X_val[index:index+1], axis=-1)[...,None])[0,:,:,0], cmap ='gray')
ax[0,1].set_title('Bicubic')
ax[1,0].imshow(x_pred[0,:,:,0], cmap ='gray')
ax[1,0].set_title('Prediction')
ax[1,1].imshow(y_val[index,:,:,0], cmap = 'gray')
ax[1,1].set_title('HR')
fig.show()
