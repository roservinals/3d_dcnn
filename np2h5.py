import numpy as np
import h5py
import tensorflow as tf
import random
from random import shuffle
import pandas as pd
import os


samplesList_benign=list()
samplesList_malign=list()
namesList_benign=list()
namesList_malign=list()

samplesList_malign=np.load('samplesList_malign.npy')
samplesList_benign=np.load('samplesList_benign.npy')
namesList_malign=np.load('namesList_malign.npy')
namesList_malign.tolist()
namesList_benign=np.load('namesList_benign.npy')
namesList_benign.tolist()
smt=np.shape(samplesList_benign)
num_benign=smt[0]
smt=np.shape(samplesList_malign)
num_malign=smt[0]
num_train=num_benign

train=list()
names_train=list()
#train=samplesList_benign+samplesList_malign[0:num_benign] #same number of samples
#names_train=[]
#nmm=namesList_malign[0:num_benign]
#namesList_benign.extend(nmm)
#print names_train

labels_train=list()
rdn=random.sample(range(0, num_benign), num_train)
rdn2=random.sample(range(0, num_malign), num_train)
for i in range(0,num_train):
    idx_b=rdn[i]
    train.append(samplesList_benign[idx_b])
    labels_train.append(0)
    names_train.append(namesList_benign[idx_b])
    idx_m=rdn2[i]
    names_train.append(namesList_malign[idx_m])
    train.append(samplesList_malign[idx_m])
    labels_train.append(1)

samplesListTest=np.load('samplesListTest.npy')
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
        namesListTest.append(name)
        labelsListTest.append(label)

h5_train = os.path.join(DIR, 'train_data.h5')
h5_test = os.path.join(DIR, 'test_data.h5')

with h5py.File('train_data.h5', 'w') as f:
    f['data'] = train
    f['label'] = labels_train
    #f['name'] = names_train

with h5py.File('test_data.h5', 'w') as f:
    f['data'] = test
    f['label'] = labelsListTest
    #f['name'] = namesListTest