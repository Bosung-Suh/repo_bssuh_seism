# -*- coding: utf-8 -*-

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

#For Functional API(NOT Sequential API)
from tensorflow.keras.layers import Dense, Input, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization,Add,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, concatenate
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K

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

# 1.hyper parameters setting
learning_rate = 0.01
training_epochs = 20
batch_size = 10

tf.random.set_seed(957)

# 1.create checkpoint directory
currentdir = os.getcwd()
checkpt_dir_name = 'checkpoints'
modelname = 'dist_Unet_CD_OH_100m_20220219_lr0_01'

modeltopdir = r'/home/bssuh/repo_bssuh_seism/program/model'
resultmodeldir = os.path.join(modeltopdir, modelname)    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq
checkpointdir = os.path.join(resultmodeldir, checkpt_dir_name)    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq/checkpoints

#0 model folder reset
if os.path.exists(resultmodeldir):
    dirlist = os.listdir(resultmodeldir)
    print('*'*50)
    print('* List of directories and files to be DELETED! listed in abspath.')
    print('* WARNING! rmtree method deletes everything in the folder!')
    print('* UNABLE TO RECOVER after delete. Backup before DELETE!\n')
    print(resultmodeldir,dirlist,sep='\n')
    deleteKey = input('%s Are you sure to format the output folder? [Y/N]' %os.path.abspath(resultmodeldir))
    
    if deleteKey == 'y':
        try:
            shutil.rmtree(resultmodeldir)
            print("Output folder RESET")
        except Exception as e:
            print(e)
    else:
        print('STOP')
        exit(0)
else:
    print('Create New model folder')

#create model folder, checkpoint folder
os.makedirs(resultmodeldir, exist_ok=True)
os.makedirs(checkpointdir, exist_ok=True) 

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

# multiprocessing(server core=40)
#cores = 10
#dataload = parmap.map(load_data,sorted(glob.glob(data_dir+'/*.npy')),pm_pbar=True,pm_processes=cores,pm_chunksize=1)    #pm_pbar:process bar 표시

# pickle module(파이썬 객체 자체를 파일로 저장. 불러오기만 하면 되므로 속도 빨라짐)
## Save pickle
""" with open(r'/home/bssuh/repo_bssuh_seism/program/eqdata.pkl','wb') as fh1:    #pickle file directory
    pickle.dump(dataload,fh1) #여기 아래에 quit()으로 일단 pkl 파일 작성
quit() """
## Load pickle
with open(r'/home/bssuh/repo_bssuh_seism/program/eqdata.pkl','rb') as fh1:    #pickle file directory
    dataload=pickle.load(fh1)

# 2.datasets
npylist = sorted(glob.glob(data_dir+'/*.npy'))    #/home/bssuh/jwhan/npy_data_test/2020364_212732.KS.YOCB.HH_42.18_.npy
labellist = []
for i in tqdm.tqdm(npylist):
    fname = i.split(slash)[-1]    #npy_data_test/2020364_212732.KS.YOCB.HH_42.18_.npy
    labelfloat = float(fname.split('_')[2])    # 42.18(float)
    labelround = float(10*round(labelfloat,1))    #42.2(float) -> 422번째(float)
    labellist.append(labelround)
print(f"Total data number is {len(labellist)}")    # Total data number is 101437

labels = np.array(labellist)
npys = np.vstack(dataload)

# 2. dataframe to float32 type
npys = npys.astype(np.float32)
labels = labels.astype(np.float32)

# 2.label to categorical(one hot encoding to 0~100km(by 100m))
labels=to_categorical(labels,1001)

""" # 2-2.label to normal distribution(0~1000(100km by 0.1km) step=1)
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

dist_max = 1000    #label(거리) 범위의 최댓값
dist_step = 1
distribute_sigma = 1   #만들려는 정규분포의 표준편차

labels = N_distribute(dist_max,dist_step,labels,distribute_sigma)    # labels : numpy.ndarray
labels = list(np.array(labels,dtype='float32')) """
""" label1 = labels[1]
label2 = labels[2]
plt.plot(label1,'ro-',label2,'bs-') #list1,2를 y값으로
plt.axis([0,1000,0,1]) #x축의 최소, 최대, y축의 최소, 최대
plt.xlabel('index')
plt.ylabel('probabiliry')
plt.legend(['label1','label2'])

plt.show()
plt.savefig(r'/home/bssuh/repo_bssuh_seism/program/Test/plot/one_hot.png',facecolor='#eeeeee') """

# 2. split train, test, validation dataset
traindata, testdata, trainlabel, testlabel = train_test_split(npys, labels, test_size=0.2, shuffle=True, random_state=42)
#traindata, valdata, trainlabel, vallabel = train_test_split(restdata, restlabel, test_size=0.25, shuffle=True, random_state=42)
# train 81149(0.8), test 20288(0.2)

# 2.batch size로 각각의 dataset 분할
traindataset = tf.data.Dataset.from_tensor_slices((traindata, trainlabel)).shuffle(buffer_size=100000).batch(batch_size)  #학습 데이터는 섞어서(다음 원소가 일정하게 선택되는 고정된 값 buffer_size는 충분히 크게)
testdataset = tf.data.Dataset.from_tensor_slices((testdata, testlabel)).batch(batch_size)    #data를 batch size만큼 잘라서 전달
#valdataset = tf.data.Dataset.from_tensor_slices((valdata, vallabel)).batch(batch_size) 

#여기까지가 data processing time measuring
processend = time.time()
processsec = processend - processstart
processtime = str(datetime.timedelta(seconds=processsec)).split(".")[0]
print(f"Data processing took {processtime}")

# 3.model define    Convolution(Fully Connected layer) - Batch Normalization - Activation - Dropout - Pooling 순서 권장
#Functional API(Input() 함수에 입력의 크기를 정의, 이전층을 다음층 함수의 입력으로 사용, Model() 함수에 입력과 출력을 정의)
#input이 1 X 6000 X 3channel 이므로 conv2D, UpSampling2D 필터 크기(kernel_size)를 1 X *로 설정해야 Size가 맞아서 Error가 발생하지 않음)
def unet(pretrained_weights = None,input_size = (1,6000,3)):    #UNet은 Functional API로 네트워크 구성
    inputs = Input(input_size)
    #Contracting Path 1(3x3 convolution&ReLU 2회 + 2x2 max-pooling) Down-sampling 1번마다 채널 수가 2배로 늘어남
    conv1 = Conv2D(filters=64, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(filters=64, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(padding = 'same')(conv1)
    #Contracting Path 2
    conv2 = Conv2D(filters=128, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(filters=128, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(padding = 'same')(conv2)
    #Contracting Path 3
    conv3 = Conv2D(filters=256, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(filters=256, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(padding = 'same')(conv3)
    #Contracting Path 4
    conv4 = Conv2D(filters=512, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(filters=512, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.4)(conv4)
    pool4 = MaxPooling2D(padding = 'same')(drop4)
    #Bottle Neck(전환 구간: 3X3 convolution&ReLU 2회 + Dropout) 모델 일반화&노이즈에 견고해짐(robust)
    conv5 = Conv2D(filters=1024, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(filters=1024, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.4)(conv5)
    #Expanding Path 1(2x2 UP-convolution&ReLU + 3x3 convolution&ReLU 2회) Up-sampling 1번마다 채널 수가 반으로 감소
    up6 = Conv2D(filters=512, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)    #Up-Conv 된 특징맵은 Contracting path의 테두리가 Cropped된 특징맵과 concatenation 됨(Copy&Crop)
    conv6 = Conv2D(filters=512, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(filters=512, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #Expanding Path 2
    up7 = Conv2D(filters=256, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(filters=256, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(filters=256, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #Expanding Path 3
    up8 = Conv2D(filters=128, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(filters=128, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(filters=128, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #Expanding Path 4
    up9 = Conv2D(filters=64, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(filters=64, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(filters=64, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(filters=2, kernel_size=(1,4), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    flatten = Flatten()(conv9)    #fully connected layer 진입을 위해 결과값을 일렬로 펴줌
    dense1 = Dense(1024, activation='relu')(flatten)   #dense layer
    drop = Dropout(0.4)(dense1)    #dense layer parameter 수가 많으므로 dropout 적용
    dense2 = Dense(1001)(drop)
    
    model = Model(inputs, dense2)
     
    return model

# 3.print and save model summary
model = unet()
model.summary()    #model에 대한 정보 표출
modelinfodir = os.path.join(resultmodeldir, "modelinfo.txt")    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq/modelinfo.txt
with open(modelinfodir, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + nextline))
print("model info save COMPLETE!")    
f = open(modelinfodir, 'a')
f.write("model info save COMPLETE!\n")

# 4.loss function               #결과 확인해보고 나중에 최대 우도 추정법(MLE;Maximum Likelihood Estimation)으로 바꾸는 것 고려
def loss_fn(model, images, labels):    #labels=정답
    logits = model(images, training=True)    #training True로 설정하면 model의 dropout layer에서 dropout이 적용
    cos_similarity = tf.keras.losses.cosine_similarity(y_true=labels, y_pred=logits,axis=-1)    #cosine similarity[-1,1] 계산(벡터 방향 같으면 +1, 반대면 -1, 직각이면 0)
    cos_distance = tf.add(cos_similarity,1)    #cosine distance[0,2] 계산(벡터 방향 같으면 0, 반대면 +2, 직각이면 +1)
    loss = tf.reduce_mean(cos_distance)     #cosine distance의 평균을 loss로 반환(벡터 방향이 같을수록 loss 값이 작아짐)
    return loss

# 5.calculate gradient
@tf.function
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)    #back propagation(model을 거꾸로 거슬러 올라가며 gradient 계산)

# 7.calculate model's accuracy    ####ND
#@tf.function      
# @tf.function이 활성화 되어 있으면 그래프 모드가 활성화 되어 그래프의 생성/실행이 분리됨. 
# 속도는 빠르지만 함수 내의 값을 바로 계산할 수 없어 tensor to numpy 연산에서 에러 발생. 
# 따라서 해당 annotation을 해제하여 eager mode에서 함수 생성/실행
def evaluate(model, images, labels):
    logits = model(images, training=False)    #training이 아니므로 기능 끄기
    logitmax = tf.argmax(input=logits,axis=1,output_type=tf.dtypes.int32)
    labelmax = tf.argmax(input=labels,axis=1,output_type=tf.dtypes.int32)
    errordistance = tf.abs(tf.subtract(labelmax,logitmax))     #정답(label)과 계산값(logit) 사이 거리의 절댓값
    errordistance_float = tf.cast(errordistance,dtype=tf.float32).numpy()
    mean_abs_error = tf.reduce_mean(errordistance_float)*0.1    #0.1km단위로 계산하므로 예측결과를 확인할 때 km 단위로 환산
    return mean_abs_error

# 6.AdamOptimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,loss=loss_fn,metrics=[evaluate])

# 8.create checkpoint for save
checkpoint = tf.train.Checkpoint(cnn=model)

# 9.training 함수 정의
@tf.function
def train(model, images, labels):
    grads = grad(model, images, labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# 9.model 학습(modelling time measure 포함)
f.write(f"Total data number is {len(labellist)}, Data processing took {processtime}.\n")
f.write(f"learning rate = {learning_rate} epoch = {training_epochs}\n")
print('Learning started. It takes sometime.')
f.write('Learning started. It takes sometime.\n')
modelstart = time.time()
losslist = []
train_mae_list = []
test_mae_list = []

for epoch in range(training_epochs):    # 1 training epoch가 1 cycle
    sum_loss = 0.
    sum_train_mae = 0
    sum_test_mae = 0
    train_step = 0
    test_step = 0  
        
    for images, labels in traindataset:    # training epoch 내에서 1 batch size가 1 cycle
        train(model, images, labels)
        loss = loss_fn(model, images, labels)
        mae_train = evaluate(model, images, labels)    #학습 중간에 loss, acc 계산
        sum_loss += loss
        sum_train_mae += mae_train
        train_step += 1
    avg_loss = sum_loss / train_step
    avg_train_mae = sum_train_mae / train_step

    losslist.append(avg_loss)
    train_mae_list.append(avg_train_mae)    
    
    for images, labels in testdataset:      # 1 epoch가 끝나면 test dataset을 넣고 모델 검증 
        mae_test = evaluate(model, images, labels)           
        sum_test_mae += mae_test           
        test_step += 1    
    avg_test_mae = sum_test_mae / test_step    

    test_mae_list.append(avg_test_mae)    
        
    epochresult = f"Epoch: {epoch + 1} loss = {avg_loss:.8f} train MAE = {avg_train_mae:.4f}km test MAE = {avg_test_mae:.4f}km"
    print(epochresult)    #epoch 끝날 때마다 training, test 정확도 출력
    f.write(epochresult + nextline)

    checkpoint.save(file_prefix=os.path.join(checkpointdir, "checkpoint"))    #epoch 끝날 때마다 checkpoint save
modelend = time.time()
modelsec = modelend - modelstart
modeltime = str(datetime.timedelta(seconds=modelsec)).split(".")[0]

# model loss, accuracy plot
epochlist=list(range(1,training_epochs+1))
plt1 = plt.subplot(2, 1, 1)               
plt.plot(epochlist, losslist, color='blue')
plt.title(modelname)
plt.ylim(0,2)
plt.ylabel('Loss')
plt.xticks(epochlist)

plt2 = plt.subplot(2, 1, 2, sharex=plt1)                
plt.plot(epochlist, train_mae_list,'r-',label='train_mae')
plt.plot(epochlist,test_mae_list,'b-',label='test_mae')
plt.ylim(0,)
plt.xlabel('Epochs')
plt.ylabel('MAE(km)')
plt.xticks(epochlist)
plt.legend(loc='best')

plt.tight_layout()
plt.show()
modelplotdir = os.path.join(resultmodeldir, "modelplot.png")    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq/modelplot.png
plt.savefig(modelplotdir,facecolor='#eeeeee')

f.write(f"Learning Finished! It took {modeltime}.")
f.close()
print(f"Learning Finished! Learning time: {modeltime}")


#개선점 train acc에 비해 test acc가 너무 낮음(데이터 갯수는 충분한데)
#일단 label을 one hot encoding->normal distribution으로 바꿔서 fix된 정답이 아니라 probability로 제공
#그래도 안되면 layer 깊이 줄이고
#K-fold Cross Evaluation은 후순위로 고려, 일단은 모델 구조부터 생각해서 test accuracy 향상