#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
# import fnmatch
# import parmap
import pickle
import glob
import pandas as pd
from obspy import read,UTCDateTime,Trace,Stream
from sklearn.model_selection import train_test_split, KFold
import tqdm    #연속적인 작업의 진행률을 시각적으로 표현    for i in tqdm.tqdm(list):같은 형식
#import multiprocessing
import math
import time
import datetime


n = 20    #label(거리) 범위의 최댓값
step = 1
xlist = list(range(0,n+1,step))    #label 거리 값으로 가능한 값

labeldata = [5, 10, 15]    #파일 이름에 저장된 label 값(=mean)
sigma = 1   #만들려는 정규분포의 표준편차
labellist = []    #최종 label(정규분포의 확률밀도함수를 가지는 리스트를 하나의 원소로 가지는 리스트)
for i in labeldata:
    distribution = []    #label list의 원소 하나
    for x in xlist:
        y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-i)**2 / (2 * sigma**2))
        distribution.append(y)
    labellist.append(distribution)
labellist = tf.convert_to_tensor(labellist, dtype=np.float32)
print(tf.size(labellist))
list1 = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
list2 = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
list3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
biglist = []
biglist.append(list1)
biglist.append(list2)
biglist.append(list3)
biglist = tf.convert_to_tensor(biglist, dtype=np.float32)

cos_similarity = tf.keras.losses.cosine_similarity(y_true=biglist, y_pred=labellist,axis=-1)
print(cos_similarity)
cos_distance = tf.add(cos_similarity,1)
print(cos_distance)
# %%
