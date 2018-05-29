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
import pandas as pd
import tensorflow as tf
from PIL import Image
import scipy.io

MIN_DISTANCE_X=16
MIN_DISTANCE_Y=16
MIN_DISTANCE_Z=16
MAX_ITERATIONS=1000
NODULE_PERCENTAGE=15
MAX_SAMPLES=15


def box_overlap(point1,point2):
    '''Cond1.  If A's left face is to the right of the B's right face,
               -  then A is Totally to right Of B
                  CubeA.X2 < CubeB.X1
    Cond2.  If A's right face is to the left of the B's left face,
               -  then A is Totally to left Of B
                  CubeB.X2 < CubeA.X1
    Cond3.  If A's top face is below B's bottom face,
               -  then A is Totally below B
                  CubeA.Z2 < CubeB.Z1
    Cond4.  If A's bottom face is above B's top face,
               -  then A is Totally above B
                  CubeB.Z2 < CubeA.Z1
    Cond5.  If A's front face is behind B's back face,
               -  then A is Totally behind B
                  CubeB.Y2 < CubeA.Y1
    Cond6.  If A's left face is to the left of B's right face,
               -  then A is Totally to the right of B
                  CubeB.X2 < CubeA.X1'''
    x_cube=(int)(MIN_DISTANCE_X/4)
    y_cube=(int)(MIN_DISTANCE_Y/4)
    z_cube=(int)(MIN_DISTANCE_Z/4)
    cubeA_x1=point1[0]-x_cube
    cubeA_x2=point1[0]+x_cube
    cubeA_y1=point1[1]-y_cube
    cubeA_y2=point1[1]+y_cube
    cubeA_z1=point1[2]-z_cube
    cubeA_z2=point1[2]+z_cube
    cubeB_x1=point2[0]-x_cube
    cubeB_x2=point2[0]+x_cube
    cubeB_y1=point2[1]-y_cube
    cubeB_y2=point2[1]+y_cube
    cubeB_z1=point2[2]-z_cube
    cubeB_z2=point2[2]+z_cube

    cond1=cubeA_x2<cubeB_x1
    cond2=cubeB_x2<cubeA_x1
    cond3=cubeA_z2<cubeB_z1
    cond4=cubeB_z2<cubeA_z1
    cond5=cubeB_y2<cubeA_y1
    cond6=cubeB_x2<cubeA_x1
    return (cond1 or cond2 or cond3 or cond4 or cond5 or cond6) # if no overlap, returns true

def sampling(image,readdata):
    points=list()
    num_slices=readdata.shape[2]
    num_non_zeros=np.count_nonzero(readdata)
    image_size=[readdata.shape[0],readdata.shape[1],readdata.shape[2]]

    x_cube=(int)(MIN_DISTANCE_X/2)
    y_cube=(int)(MIN_DISTANCE_Y/2)
    z_cube=(int)(MIN_DISTANCE_Z/2)
    box_size=MIN_DISTANCE_X*MIN_DISTANCE_Y*MIN_DISTANCE_Z
    non_zero_indices=np.nonzero(readdata)
    num_samples=0
    samples=list()
    # First point

    # Two conditions:
    # 1. Contains 75% of nodule
    # 2. Distance between points

    # Obtain pixels around point
    condition=True
    count=0
    while condition:
        count+=1
        if(num_non_zeros==0):
            condition=False
            break
        n=random.randint(0,num_non_zeros-1)
        point=[non_zero_indices[0][n],non_zero_indices[1][n],non_zero_indices[2][n]]
        subimage_nodule=0
        for i in range(-x_cube,x_cube):
            index_x=point[0]+i
            for j in range(-y_cube,y_cube):
                index_y=point[1]+j
                for k in range(-z_cube,z_cube+1):
                    index_z=point[2]+k
                    if((index_x>=0 and index_x<readdata.shape[0])and(index_y>=0 and index_y<readdata.shape[1])and(index_z>=0 and index_z<readdata.shape[2])):
                        if(readdata[index_x][index_y][index_z]==1):
                            subimage_nodule+=1
        if(subimage_nodule*100/box_size>=NODULE_PERCENTAGE):
            condition=False
            points.append(point)
            num_samples+=1
            tmp_image=np.zeros((MIN_DISTANCE_X,MIN_DISTANCE_Y,MIN_DISTANCE_Z*2))
            for i in range(-x_cube,x_cube):
                index_x=point[0]+i
                for j in range(-y_cube,y_cube):
                    index_y=point[1]+j
                    for k in range(-z_cube,z_cube):
                        index_z=point[2]+k
                        # MIRRORING
                        if(index_x<0):
                            index_x=-index_x
                        if(index_x>=readdata.shape[0]):
                            index_x=readdata.shape[0]-1-(index_x-readdata.shape[0])
                        if(index_y<0):
                            index_y=-index_y
                        if(index_y>=readdata.shape[1]):
                            index_y=readdata.shape[1]-1-(index_y-readdata.shape[1])
                        if(index_z<0):
                            index_z=-index_z
                        if(index_z>=readdata.shape[2]):
                            index_z=readdata.shape[2]-1-(index_z-readdata.shape[2])
                        tmp_image[i+x_cube,j+y_cube,MIN_DISTANCE_Z+k+z_cube]=readdata[index_x,index_y,index_z]
                        tmp_image[i+x_cube,j+y_cube,k+z_cube]=image[index_x,index_y,index_z]
            samples.append(tmp_image)
        #this point is  not elegible anymore:
        a=non_zero_indices[0]
        b=non_zero_indices[1]
        c=non_zero_indices[2]
        a=np.delete(a,n)
        b=np.delete(b,n)
        c=np.delete(c,n)
        non_zero_indices=[a,b,c]
        num_non_zeros-=1

        if(count>MAX_ITERATIONS):
            condition=False
            break
        if(num_non_zeros==0):
            condition=False
            break

    condition=True
    while condition and num_non_zeros>0:
        count+=1
        n=random.randint(0,num_non_zeros-1)
        point=[non_zero_indices[0][n],non_zero_indices[1][n],non_zero_indices[2][n]]
        # First condition
        c1=True
        for r in range(0,len(points)):
            if(box_overlap(point,points[r])==0):
                c1=False
                #this point is  not elegible anymore:
                break
        if(c1):
            subimage_nodule=0
            for i in range(0,MIN_DISTANCE_X):
                index_x=point[0]-i-x_cube
                for j in range(0,MIN_DISTANCE_Y):
                    index_y=point[1]-j-y_cube
                    for k in range(0,MIN_DISTANCE_Z):
                        index_z=point[2]-k-z_cube
                        if((index_x>=0 and index_x<readdata.shape[0])and(index_y>=0 and index_y<readdata.shape[1])and(index_z>=0 and index_z<readdata.shape[2])):
                            if(readdata[index_x][index_y][index_z]==1):
                                subimage_nodule+=1
            # second condition: 75% nodule
            if(subimage_nodule*100/box_size>=NODULE_PERCENTAGE):
                points.append(point)
                num_samples+=1
                tmp_image=np.zeros((MIN_DISTANCE_X,MIN_DISTANCE_Y,MIN_DISTANCE_Z*2))
                for i in range(-x_cube,x_cube):
                    index_x=point[0]+i
                    for j in range(-y_cube,y_cube):
                        index_y=point[1]+j
                        for k in range(-z_cube,z_cube):
                            index_z=point[2]+k
                            # MIRRORING
                            if(index_x<0):
                                index_x=-index_x
                            if(index_x>=readdata.shape[0]):
                                index_x=readdata.shape[0]-1-(index_x-readdata.shape[0])
                            if(index_y<0):
                                index_y=-index_y
                            if(index_y>=readdata.shape[1]):
                                index_y=readdata.shape[1]-1-(index_y-readdata.shape[1])
                            if(index_z<0):
                                index_z=-index_z
                            if(index_z>=readdata.shape[2]):
                                index_z=readdata.shape[2]-1-(index_z-readdata.shape[2])
                            tmp_image[i+x_cube,j+y_cube,MIN_DISTANCE_Z+k+z_cube]=readdata[index_x,index_y,index_z]
                            tmp_image[i+x_cube,j+y_cube,k+z_cube]=image[index_x,index_y,index_z]
                samples.append(tmp_image)
        #this point is  not elegible anymore:
        a=non_zero_indices[0]
        b=non_zero_indices[1]
        c=non_zero_indices[2]
        a=np.delete(a,n)
        b=np.delete(b,n)
        c=np.delete(c,n)
        non_zero_indices=[a,b,c]
        num_non_zeros-=1
        if(count>MAX_ITERATIONS):
            condition=False
            break
        if(num_non_zeros==0):
            condition=False
            break
        if(num_samples>=MAX_SAMPLES):
            condition=False
            break
    return samples

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
