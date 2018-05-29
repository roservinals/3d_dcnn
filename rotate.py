import numpy as np
import h5py
import tensorflow as tf
import random
from random import shuffle
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import nrrd
import matplotlib.pyplot as plt
import random
from itertools import product, combinations
from math import fabs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import tensorflow as tf
from PIL import Image
import scipy.io
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom
import sampleVol
from sampleVol import sampling

def rotateVol(image):
    # Random rotation 
    angle1=random.randint(0,36)
    angle1=angle1*10#*math.pi/180
    # c1=cos(angle1)
    # s1=sin(angle1)
    angle2=random.randint(0,36)
    angle2=angle2*10#*math.pi/180
    # c2=cos(angle2)
    # s2=sin(angle2)
    angle3=random.randint(0,36)
    angle3=angle3*10#*math.pi/180
    # c3=cos(angle3)
    # s3=sin(angle3)
    # R=[[c2*c3, -c2*c3, s2],
    #                  [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
    #                  [s1*s3-c1*c3*c2, c3*s1+c1*s2*s3, c1*c2]]
    cube=np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    cube1=np.zeros((image.shape[0],image.shape[1],image.shape[2]/2))
    cube2=np.zeros((image.shape[0],image.shape[1],image.shape[2]/2))

    # SEPARATE IMAGE AND MASK
    im=image[:,:,0:(image.shape[2]/2)]
    mask=image[:,:,(image.shape[2]/2):image.shape[2]]

    # IMAGE
    #  x rotation
    for i in range(0,im.shape[0]):
        layer = im[i,:,:]
        cube1[i,:,:] = (Image.fromarray(layer).rotate(angle1, resample=Image.BICUBIC))
    # Y rotation
    for i in range(0,im.shape[1]):
        layer = im[:,i,:]
        cube1[:,i,:] = (Image.fromarray(layer).rotate(angle2, resample=Image.BICUBIC))
     # Z rotation
    for i in range(0,im.shape[2]):
        layer = im[:,:,i]
        cube1[:,:,i] = (Image.fromarray(layer).rotate(angle3, resample=Image.BICUBIC))
    ## MASK
    for i in range(0,mask.shape[0]):
        layer = mask[i,:,:]
        cube2[i,:,:] = (Image.fromarray(layer).rotate(angle1, resample=Image.BICUBIC))
    # Y rotation
    for i in range(0,mask.shape[1]):
        layer = mask[:,i,:]
        cube2[:,i,:] = (Image.fromarray(layer).rotate(angle2, resample=Image.BICUBIC))
    # Z rotation
    for i in range(0,mask.shape[2]):
        layer = mask[:,:,i]
        cube2[:,:,i] = (Image.fromarray(layer).rotate(angle3, resample=Image.BICUBIC))
    cube[:,:,0:im.shape[2]]=cube1
    cube[:,:,im.shape[2]:image.shape[2]]=cube2
    return cube

def scaleVol(image):
    # Random rotation 
    zoom=random.uniform(1.5,3)
    cube=np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    for i in range(0,image.shape[2]):
        layer = image[:,:,i]
        cube[:,:,i] = clipped_zoom(layer,zoom)
    return cube

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


######################## TRAINING ####################

################## ROTATION and SCALING DB ###################
rot_samplesList_benign=list()
rot_samplesList_malign=list()
rot_namesList_benign=list()
rot_namesList_malign=list()
rot_numSamplesListTrain_benign=list()
rot_numSamplesListTrain_malign=list()

sc_samplesList_benign=list()
sc_samplesList_malign=list()
sc_namesList_benign=list()
sc_namesList_malign=list()
sc_numSamplesListTrain_benign=list()

sc_numSamplesListTrain_malign=list()
trainLabels = pd.read_excel('labels_modified.xls', sheet_name='y_train')
trainLabels.as_matrix()
size=np.shape(trainLabels)
label=np.zeros((size[0],1))

numRotations=2
numScaling=2
for i in range(0,size[0]):#
    print str(i)+'/'+str(size[0])
    name=trainLabels.iloc[i,0]
    label[i]=trainLabels.iloc[i,1]
    folder='volumes_bounded/'+name
    d= scipy.io.loadmat(folder+'/scan.mat') # read the data back from file
    data=d["volumeCT"]
    dL=scipy.io.loadmat(folder+'/mask.mat')
    dataLabel=dL["volumeROI"]
    folder2='DB/'+name

    # First sample original image to two db
    x=sampling(data,dataLabel)
    sz=np.shape(x)
    count_rot=0
    count_sc=0
    for j in range(0,sz[0]):
        if(label[i]==0):
            rot_samplesList_benign.append(x[j][:][:][:])
            rot_namesList_benign.append(name+'_'+str(count_rot))
            rot_numSamplesListTrain_benign.append(data.shape[0])
            sc_samplesList_benign.append(x[j][:][:][:])
            sc_namesList_benign.append(name+'_'+str(count_sc))
            sc_numSamplesListTrain_benign.append(data.shape[0])
        else:
            rot_samplesList_malign.append(x[j][:][:][:])
            rot_namesList_malign.append(name+'_'+str(count_rot))
            rot_numSamplesListTrain_malign.append(data.shape[0])
            sc_samplesList_malign.append(x[j][:][:][:])
            sc_namesList_malign.append(name+'_'+str(count_sc))
            sc_numSamplesListTrain_malign.append(data.shape[0])
        count_rot+=1
        count_sc+=1
    # ROTATE
    dataWithMask=np.zeros((data.shape[0],data.shape[1],data.shape[2]*2))
    dataWithMask[:,:,0:data.shape[2]]=data
    dataWithMask[:,:,data.shape[2]:2*data.shape[2]]=dataLabel
    for r in range(0,numRotations):
        rotatedImage=rotateVol(dataWithMask)
        x=sampling(rotatedImage[:,:,0:data.shape[2]],rotatedImage[:,:,data.shape[2]:data.shape[2]*2])
        sz=np.shape(x)
        for j in range(0,sz[0]):
            if(label[i]==0):
                rot_samplesList_benign.append(x[j][:][:][:])
                rot_namesList_benign.append(name+'_'+str(count_rot))
            else:
                rot_samplesList_malign.append(x[j][:][:][:])
                rot_namesList_malign.append(name+'_'+str(count_rot))
            count_rot+=1
    for s in range(0,numScaling):
        scaledImage=scaleVol(dataWithMask)
        x=sampling(scaledImage[:,:,0:data.shape[2]],scaledImage[:,:,data.shape[2]:data.shape[2]*2])
        sz=np.shape(x)
        for j in range(0,sz[0]):
            if(label[i]==0):
                sc_samplesList_benign.append(x[j][:][:][:])
                sc_namesList_benign.append(name+'_'+str(count_sc))
            else:
                sc_samplesList_malign.append(x[j][:][:][:])
                sc_namesList_malign.append(name+'_'+str(count_sc))
        count_sc+=1

smt=np.shape(rot_samplesList_benign)
num_benign=smt[0]
smt=np.shape(rot_samplesList_malign)
num_malign=smt[0]
rot_num_train=min(num_benign,num_malign)


rot_train=list()
rot_names_train=list()
rot_labels_train=list()
rdn=random.sample(range(0, num_benign), rot_num_train)
rdn2=random.sample(range(0, num_malign), rot_num_train)
for i in range(0,rot_num_train):
    idx_b=rdn[i]
    rot_train.append(rot_samplesList_benign[idx_b])
    rot_labels_train.append(0)
    rot_names_train.append(rot_namesList_benign[idx_b])
    idx_m=rdn2[i]
    rot_names_train.append(rot_namesList_malign[idx_m])
    rot_train.append(rot_samplesList_malign[idx_m])
    rot_labels_train.append(1)
 
for i in range(0,rot_num_train*2):
    h5_t = os.path.join('rot_train/', 'rottrain_'+rot_names_train[i]+'_data.h5')
    with h5py.File('rottrain_'+rot_names_train[i]+'_data.h5', 'w') as f:
        f['data'] = rot_train[i]
        f['label'] = rot_labels_train[i]  

smt=np.shape(sc_samplesList_benign)
num_benign=smt[0]
smt=np.shape(sc_samplesList_malign)
num_malign=smt[0]
sc_num_train=min(num_benign,num_malign)
sc_train=list()
sc_names_train=list()
sc_labels_train=list()
rdn=random.sample(range(0, num_benign), sc_num_train)
rdn2=random.sample(range(0, num_malign), sc_num_train)
for i in range(0,sc_num_train):
    idx_b=rdn[i]
    sc_train.append(sc_samplesList_benign[idx_b])
    sc_labels_train.append(0)
    sc_names_train.append(sc_namesList_benign[idx_b])
    idx_m=rdn2[i]
    sc_names_train.append(sc_namesList_malign[idx_m])
    sc_train.append(sc_samplesList_malign[idx_m])
    sc_labels_train.append(1)
    
for i in range(0,sc_num_train*2):
    h5_t = os.path.join('sc_train/', 'sctrain_'+sc_names_train[i]+'_data.h5')
    with h5py.File('sctrain_'+sc_names_train[i]+'_data.h5', 'w') as f:
        f['data'] = sc_train[i]
        f['label'] = sc_labels_train[i]  


################## TEST #####################
samplesListTest=list()
numSamplesListTest=list()
numSamplesListTest_benign=list()
numSamplesListTest_malign=list()
testLabels = pd.read_excel('labels_modified.xls', sheet_name='y_test')
testLabels.as_matrix()
size=np.shape(testLabels)
label=np.zeros((size[0],1))
for i in range(0,size[0]):
    print str(i)+'/'+str(size[0])
    name=testLabels.iloc[i,0]
    label[i]=testLabels.iloc[i,1]
    folder='volumes_bounded/'+name
    d= scipy.io.loadmat(folder+'/scan.mat') # read the data back from file
    data=d["volumeCT"]
    dL=scipy.io.loadmat(folder+'/mask.mat')
    dataLabel=dL["volumeROI"]
    folder2='DB/'+name
    x=sampling(data,dataLabel)
    samplesListTest.append(x)
    sz=np.shape(x)
    numSamplesListTest.append(sz[0])

np.save('samplesListTest',samplesListTest)
namesListTest=list()
labelsListTest=list()

testLabels = pd.read_excel('labels_modified.xls', sheet_name='y_test')
testLabels.as_matrix()
size=np.shape(testLabels)
test=list()
for i in range(0,size[0]):
    name=testLabels.iloc[i,0]
    label=testLabels.iloc[i,1]
    x=samplesListTest[i]
    sz=np.shape(x)
    for j in range(0,sz[0]):
        y=x[j]
        test.append(y)
        namesListTest.append(name+'_'+str(j))
        labelsListTest.append(label)

num_test=np.shape(labelsListTest)
for i in range(0,num_test[0]):
    h5_test = os.path.join('test/', 'test_'+namesListTest[i]+'_data.h5')
    with h5py.File('test_'+namesListTest[i]+'_data.h5', 'w') as f:
        f['data'] = test[i]
        f['label'] = labelsListTest[i]


# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# for i in range(0,dataLabel.shape[0]):
#     for j in range(0,dataLabel.shape[1]):
#         for k in range(0,dataLabel.shape[2]):
#             if dataLabel[i,j,k]==1:
#                 ax.scatter3D(i,j,k,marker='s',c='k')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax2 = fig.add_subplot(122, projection='3d')
# for i in range(0,x.shape[0]):
#     for j in range(0,x.shape[1]):
#         for k in range(0,x.shape[2]):
#             if x[i,j,k]==1:
#                 ax2.scatter3D(i,j,k,marker='s',c='k')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# plt.show()
