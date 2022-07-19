# -*- coding: utf-8 -*-
#%%
#CNN의 전체 과정
""" 
대략적인 과정
1. set hyper parameters(learning rate, training epochs, batch size, etc)
2. make data pipelining(use tf.data)    #dataset을 로드하고 설정한 batch size 만큼 데이터를 가져와서 network에 전달
3. build neural network model(use tf.keras.sequential API)
4. define loss function(cross entropy)    #classification 문제이므로 cross entropy
5. calculate gradient(use tf.GradientTape)    #weight에 대한 gradient 계산
6. select optimizer(Adam optimizer)    #계산한 gradient로부터 weight를 업데이트
7. define metric for model performance(accuracy)    #학습모델의 성능을 판단하는 지표 설정
8. (optional) make checkpoing for saving    #학습 결과를 임시로 저장(선택)
9. train and validate neural network model
"""
#서버 접속 상태로 top 명령어로 CPU 모니터링!!
# 0.import libraries
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
import scipy.stats as stats
import shutil
import pathlib
import random
#print(f"Tensorflow {tf.__version__}")  #Tensorflow 2.4.1(20211223)

#data processing time measure start
processstart = time.time()

# core number count(activate import multiprocessing)
#cores = multiprocessing.cpu_count()    #40

#0 special character definition
bs='\\'
slash='/'
dot='.'
nextline = '\n'

#0 directory of data
data_dir=r'/home/bssuh/jwhan/npy_data_test'    #/home/bssuh/jwhan/npy_data_test
testdir=r'/home/bssuh/program/Test'

# 1.hyper parameters setting
learning_rate = 0.0001
training_epochs = 20
batch_size = 10

tf.random.set_seed(957)

# 1.create checkpoint directory
currentdir = os.getcwd()
checkpt_dir_name = 'checkpoints'
modelname = 'dist_cnn_cosine_distance_20220111'

modeltopdir = r'/home/bssuh/program/model'
resultmodeldir = os.path.join(modeltopdir, modelname)    #/home/bssuh/program/model/dist_cnn_seq
checkpointdir = os.path.join(resultmodeldir, checkpt_dir_name)    #/home/bssuh/program/model/dist_cnn_seq/checkpoints
test_txtdir = os.path.join(testdir,'shuffle_test.txt',)

# GPU setting(watch -n 0.1 nvidia-smi 로 GPU monitoring!!)
#GPU로 돌릴 때는 VSCODE 터미널로 돌리지 말고 WSL 따로 열어서 돌릴 것!!!!!!!!!!
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    # 텐서플로가 첫 번째 GPU만 사용하도록 제한
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print('# GPU selected!')
    except RuntimeError as e:
        # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
        print(e)

# 2.load data        #INPUT=Width*Height*channel
def load_data(npy):    
    st=np.load(npy)
    a=st.reshape(6000,3)    # 6000*3
    b=np.expand_dims(a,axis=0)    # ??*6000*3
    c=np.expand_dims(b,axis=0)    # ??*??*6000*3(ZNE)    batch*height*width*channel
    return c

## Load pickle
with open(r'/home/bssuh/program/eqdata.pkl','rb') as fh1:    #pickle file directory
    dataload=pickle.load(fh1)

samplingnum=1000
samp_npylist=[]
samp_labellist=[]

datanum=len(dataload)
datanumlist=list(range(0,datanum,1))
randomsamplelist=random.sample(datanumlist,samplingnum)

npylist = sorted(glob.glob(data_dir+'/*.npy'))
for i in randomsamplelist:
    sample_npy=npylist[i]
    samp_npylist.append(sample_npy)

for i in samp_npylist:
    fname = i.split(slash)[-1]    #npy_data_test/2020364_212732.KS.YOCB.HH_42.18_.npy
    labelfloat = float(fname.split('_')[2])    # 42.18(float)
    labelround = float(round(labelfloat))    #42(float)
    samp_labellist.append(labelround)    
datalist=[]
for i in randomsamplelist:
    data=dataload[i]
    datalist.append(data)

labels = np.array(samp_labellist)
npys = np.vstack(datalist)

# 2. dataframe to float32 type
npys = npys.astype(np.float32)
labels = labels.astype(np.float32)
totaltest = labels
# 2.label to categorical(one hot encoding to 0~100)
labels=to_categorical(labels,101)

""" # 2-2.label to normal distribution(0~100 step=1)
def N_distribute(n,step,meanlist,sigma):
    xlist = list(range(0,n+1,step))    #label 거리 값으로 가능한 값
    label_final = []    #최종 label(정규분포의 확률밀도함수를 가지는 리스트를 하나의 원소로 가지는 리스트)
    for i in meanlist:
        distribution = []    #label list의 원소 하나
        for x in xlist:
            y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-i)**2 / (2 * sigma**2))
            distribution.append(y)
        label_final.append(distribution)
    return label_final

dist_max = 100    #label(거리) 범위의 최댓값
dist_step = 1
distribute_sigma = 1   #만들려는 정규분포의 표준편차

labels = N_distribute(dist_max,dist_step,labels,distribute_sigma)    # labels : numpy.ndarray
labels = list(np.array(labels,dtype='float32')) """

# 2. split train, test, validation dataset
traindata, testdata, trainlabel, testlabel = train_test_split(npys, labels, test_size=0.2, shuffle=True, random_state=42)
#traindata, valdata, trainlabel, vallabel = train_test_split(restdata, restlabel, test_size=0.25, shuffle=True, random_state=42)
# train 81149(0.8), test 20288(0.2)

########shuffle 확인
def label_test(label):
    labelmax_test = tf.math.argmax(input=label,axis=1,output_type=tf.dtypes.int32)
    return labelmax_test

traintest = label_test(trainlabel)
testtest = label_test(testlabel)
plt.subplot(2, 1, 1)
plt.hist(totaltest,bins=list(range(0,101,1)),label='total')
plt.hist(traintest,list(range(0,101,1)),label='train')
plt.hist(testtest,list(range(0,101,1)),label='test')
plt.xlim([0, 100])
plt.ylim([0,30])
plt.legend(loc='best')

x1=list(range(1,len(totaltest)+1))
x2=list(range(1,len(traintest)+1))
x3=list(range(1,len(testtest)+1))
y1=totaltest
y2=traintest
y3=testtest

plt.subplot(2, 1, 2)
plt.violinplot([totaltest,traintest,testtest],positions=[1,2,3],showmeans=True,showmedians=True)
plt.ylim([0,100])
plt.xticks([1,2,3],label=['total','train','test'])


plt.show()
plt.savefig(r'/home/bssuh/program/Test/plot/train_test_split_1000sample.png',facecolor='#eeeeee')

traintest=traintest.numpy()
testtest=testtest.numpy()
with open(test_txtdir, 'w') as f:
    samp_labelstr=', '.join(map(str, samp_labellist))
    f.write(samp_labelstr+nextline)
    f.close()
with open(test_txtdir,'a') as f:
    f.write(nextline)
    trainteststr=', '.join(map(str, traintest))
    f.write(trainteststr+nextline)
    f.write(nextline)
    testteststr=', '.join(map(str, testtest))
    f.write(testteststr+nextline)
    f.write(nextline)

###########
# 2.batch size로 각각의 dataset 분할
traindataset = tf.data.Dataset.from_tensor_slices((traindata, trainlabel)).shuffle(buffer_size=100000).batch(batch_size)  #학습 데이터는 섞어서(다음 원소가 일정하게 선택되는 고정된 값 buffer_size는 충분히 크게)
testdataset = tf.data.Dataset.from_tensor_slices((testdata, testlabel)).batch(batch_size)    #data를 batch size만큼 잘라서 전달
#valdataset = tf.data.Dataset.from_tensor_slices((valdata, vallabel)).batch(batch_size) 
with open(test_txtdir, 'a') as f:
    f.write(traindataset)
    f.write(nextline)
    f.write(testdataset)
    f.write(nextline)
    f.close()
    
#여기까지가 data processing time measuring
processend = time.time()
processsec = processend - processstart
processtime = str(datetime.timedelta(seconds=processsec)).split(".")[0]
print(f"Data processing took {processtime}")


""" # model loss, accuracy plot
epochlist=list(range(1,training_epochs+1))
plt.subplot(3, 1, 1)               
plt.plot(epochlist, losslist, color='blue')
plt.title('Loss')
plt.ylim(0,2)
plt.ylabel('Loss')

plt.subplot(3, 1, 2)               
plt.plot(epochlist, train_acc_list,'r-',label='train_acc')
plt.plot(epochlist,test_acc_list,'b-',label='test_acc')
plt.title('Accuracy')
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.subplot(3, 1, 3)                
plt.plot(epochlist, train_mae_list,'r-',label='train_mae')
plt.plot(epochlist,test_mae_list,'b-',label='test_mae')
plt.title('Mean Absolute Error')
plt.ylim(0,)
plt.xlabel('Epochs')
plt.ylabel('MAE(km)')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
modelplotdir = os.path.join(resultmodeldir, "modelplot.png")    #/home/bssuh/program/model/dist_cnn_seq/modelplot.png
plt.savefig(modelplotdir,facecolor='#eeeeee')

f.write(f"Learning Finished! It took {modeltime}.")
f.close()
print(f"Learning Finished! Learning time: {modeltime}")
 """

#개선점 train acc에 비해 test acc가 너무 낮음(데이터 갯수는 충분한데)
#일단 label을 one hot encoding->normal distribution으로 바꿔서 fix된 정답이 아니라 probability로 제공
#그래도 안되면 layer 깊이 줄이고
#K-fold Cross Evaluation은 후순위로 고려, 일단은 모델 구조부터 생각해서 test accuracy 향상