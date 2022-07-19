
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
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
import scipy.stats as stats
import shutil
import pathlib

bs='\\'
slash='/'
dot='.'
nextline = '\n'

currentdir = os.getcwd()
checkpt_dir_name = 'checkpoints'
modelname = 'dist_cnn_seq_cosine_distance_avg_change'
textdir = r'/home/bssuh/program/Test/modelinfo.txt'
modeltopdir = r'/home/bssuh/program/model'
plotdir = r'/home/bssuh/program/model/dist_cnn_seq_cosine_distance_avg_change/modelplot.png'
textappenddir = r'/home/bssuh/program/Test/modeltext.txt'

losslist = []
trainacclist=[]
testacclist=[]
epoch = 113
with open(textappenddir,'w') as t:
    t.close()
with open(textappenddir,'a') as t:
    with open(textdir,'r') as f:
        for line in f:
            print(line)
            partlist=line.split('=')
            losspart = partlist[1]
            trainpart = partlist[2]
            testpart = partlist[3]

            loss = losspart.split(' train')[0]
            train = trainpart.split(' test')[0]
            test = testpart.split('\n')[0]

            appendline = f'{loss} {train} {test}\n'
            print(appendline)
            t.write(appendline)
t.close()



""" epochlist=list(range(1,epoch+1))
plt.subplot(2, 1, 1)               
plt.plot(epochlist, losslist, color='blue')
plt.title('Loss')
plt.ylim(0,2)
plt.ylabel('Loss')

plt.subplot(2, 1, 2)                
plt.plot(epochlist,trainacclist,'r-',label='train_acc')
plt.plot(epochlist,testacclist,'b-',label='test_acc')
plt.title('Accuracy')
plt.ylim(0,1)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

plt.savefig(plotdir,facecolor='#eeeeee')
         """
