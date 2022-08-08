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


#print(f"Tensorflow {tf.__version__}")  #Tensorflow 2.4.1(20211223)

#그냥 쓰면 tf가 서버 전체 CPU를 써버리니 core limit 코드 필요!!!!!!!!!!
#tf.config.threading.set_intra_op_parallelism_threads(1)
#tf.config.threading.set_inter_op_parallelism_threads(1)    #분산작업 사이, 내부에서 사용하는 코어를 1개(100%)로 제한
#이거 써도 20~30 core까지 올라가는데 순간적으로 올라가는 거는 상관없으니 그냥 GPU로 돌릴 것

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
learning_rate_initial = 0.0001
training_epochs = 100
batch_size = 10

tf.random.set_seed(777)

# 1.create checkpoint directory
currentdir = os.getcwd()
checkpt_dir_name = 'checkpoints'
modelname = 'dist_cnn_seq_norm'

modeltopdir = r'/home/bssuh/repo_bssuh_seism/program/model'
resultmodeldir = os.path.join(modeltopdir, modelname)    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq
checkpointdir = os.path.join(resultmodeldir, checkpt_dir_name)    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq/checkpoints

os.makedirs(resultmodeldir, exist_ok=True)
os.makedirs(checkpointdir, exist_ok=True)   

# GPU setting(watch -n 0.1 nvidia-smi 로 GPU monitoring!!)
#GPU로 돌릴 때는 가상환경 "eqt4"에서, VSCODE 터미널로 돌리지 말고 WSL 따로 열어서 돌릴 것!!!!!!!!!!
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
    labellist.append(labelfloat)
print(f"Total data number is {len(labellist)}")    # Total data number is 101437

labels = np.array(labellist)
npys = np.vstack(dataload)

# 2. split train, test, validation dataset
restdata, testdata, restlabel, testlabel = train_test_split(npys, labels, test_size=0.2, shuffle=True, random_state=42)
traindata, valdata, trainlabel, vallabel = train_test_split(restdata, restlabel, test_size=0.25, shuffle=True, random_state=42)
# train 60861(0.6), test 20288(0.2), val 20288(0.2)

# 2. dataframe to float32 type
traindata = traindata.astype(np.float32)
testdata = testdata.astype(np.float32)
valdata = valdata.astype(np.float32)
trainlabel = trainlabel.astype(np.float32)
testlabel = testlabel.astype(np.float32)
vallabel = vallabel.astype(np.float32)

""" # 2-1.label to categorical(one hot encoding to 0~100)
trainlabel=to_categorical(trainlabel,101)
testlabel=to_categorical(testlabel,101)
vallabel=to_categorical(vallabel,101) """

# 2-2.label to normal distribution(0~100 step=1)
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

trainlabel = N_distribute(dist_max,dist_step,trainlabel,distribute_sigma)
testlabel = N_distribute(dist_max,dist_step,testlabel,distribute_sigma)
vallabel = N_distribute(dist_max,dist_step,vallabel,distribute_sigma)

# 2.batch size로 각각의 dataset 분할
traindataset = tf.data.Dataset.from_tensor_slices((traindata, trainlabel)).shuffle(buffer_size=100000).batch(batch_size)  #학습 데이터는 섞어서(다음 원소가 일정하게 선택되는 고정된 값 buffer_size는 충분히 크게)
testdataset = tf.data.Dataset.from_tensor_slices((testdata, testlabel)).batch(batch_size)    #data를 batch size만큼 잘라서 전달
valdataset = tf.data.Dataset.from_tensor_slices((valdata, vallabel)).batch(batch_size) 

#여기까지가 data processing time measuring
processend = time.time()
processsec = processend - processstart
processtime = str(datetime.timedelta(seconds=processsec)).split(".")[0]
print(f"Data processing took {processtime}")

# 3.model define
def create_model():    #함수 형태(sequential)로 네트워크 구성
    model = keras.Sequential()    #API 선언
    #First Conv2d+Maxpool
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(1,4), activation=tf.nn.relu, padding='SAME', 
                                  input_shape=(1,6000,3)))    #첫번째 layer에만 input shape 입력(height*width*channel)
    model.add(keras.layers.MaxPool2D(padding='SAME'))    #size 2*2, stride 2는 default이므로 생략
    #Second Conv2d+Maxpool
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(1,4), activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    #Third Conv2d+Maxpool
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(1,4), activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    #Fourth Conv2d+Maxpool
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(1,4), activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    #Flatten&Fully Connected Layer
    model.add(keras.layers.Flatten())    #fully connected layer 진입을 위해 결과값을 일렬로 펴줌
    model.add(keras.layers.Dense(1024, activation=tf.nn.relu))   #dense layer
    model.add(keras.layers.Dropout(0.4))    #dense layer parameter 수가 많으므로 dropout 적용
    model.add(keras.layers.Dense(101))    #layer를 순서대로 하나씩 추가
    return model

# 3.print and save model summary
model = create_model()
model.summary()    #model에 대한 정보 표출
modelinfodir = os.path.join(resultmodeldir, "modelinfo.txt")    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq/modelinfo.txt
with open(modelinfodir, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + nextline))
print("model info save COMPLETE!") 
f.close()   
f = open(modelinfodir, 'a')
f.write("model info save COMPLETE!\n")

# 4.loss function
#@tf.function
#    WARNING:tensorflow:Entity <function loss_fn at 0x000002A0B6640EA0> could not be transformed and will be staged without change. 
#    Error details can be found in the logs when running with the env variable AUTOGRAPH_VERBOSITY >= 1. 
#    Please report this to the AutoGraph team. 
#    Cause: Unexpected error transforming <function loss_fn at 0x000002A0B6640EA0>. 
#    If you believe this is due to a bug, please set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output when filing the bug report. 
#    Caused by: Bad argument number for Name: 3, expecting 4
#라고 뜨는 경우 있음. @tf.function 지우고 돌리면 Warning 없이 작동할 수도?
def loss_fn(model, images, labels):    #labels=정답
    logits = model(images, training=True)    #training True로 설정하면 model의 dropout layer에서 dropout이 적용
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(    #softmax도 결과 확률 총합을 1로 만들고, from_logits=True도 경과 총합을 1로 만드므로 둘 중 하나만 활성화(여기서는 softmax 없으므로 True로 설정)
        y_pred=logits, y_true=labels, from_logits=True))    #softmax cross entropy로 loss function 계산
    return loss   

# 5.calculate gradient
@tf.function
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)    #back propagation(model을 거꾸로 거슬러 올라가며 gradient 계산)

# 7.calculate model's accuracy
@tf.function
def evaluate(model, images, labels):
    logits = model(images, training=False)    #training이 아니므로 기능 끄기
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))    #정답(label)과 계산값(logit)을 비교
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

#learning rate decay
""" lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate_initial,
                                                            decay_steps=7,
                                                            decay_rate=0.8,
                                                            staircase=True)
 """
# 6.AdamOptimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_initial)

# 8.create checkpoint for save
checkpoint = tf.train.Checkpoint(cnn=model)

# 9.training 함수 정의
@tf.function
def train(model, images, labels):
    grads = grad(model, images, labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# 9.model 학습(modelling time measure 포함)
f.write(f"Total data number is {len(labellist)}, Data processing took {processtime}\n")
print('Learning started. It takes sometime.')
f.write('Learning started. It takes sometime.\n')
f.close()
modelstart = time.time()
for epoch in range(training_epochs):    # 1 training epoch가 1 cycle
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0    
    
    f = open(modelinfodir, 'a')

    for images, labels in traindataset:    # training epoch 내에서 1 batch size가 1 cycle
        train(model, images, labels)
        #grads = grad(model, images, labels)                
        #optimizer.apply_gradients(zip(grads, model.variables))    #training 함수 정의 부분에서 정의한 부분
        loss = loss_fn(model, images, labels)
        acc = evaluate(model, images, labels)    #학습 중간에 loss, acc 계산
        avg_loss = avg_loss + loss
        avg_train_acc = avg_train_acc + acc    #평균 계산
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step
    
    for images, labels in testdataset:      # 1 epoch가 끝나면 test dataset을 넣고 모델 검증 
        acc = evaluate(model, images, labels)        
        avg_test_acc = avg_test_acc + acc
        test_step += 1    
    avg_test_acc = avg_test_acc / test_step    

    epochresult = f"Epoch: {epoch + 1} loss = {avg_loss:.8f} train accuracy = {avg_train_acc:.6f} test accuracy = {avg_test_acc:.6f}"
    f.write(epochresult + nextline)
    f.close()
    print(epochresult)    #epoch 끝날 때마다 training, test 정확도 출력

    checkpoint.save(file_prefix=checkpointdir)    #epoch 끝날 때마다 checkpoint save
modelend = time.time()
modelsec = modelend - modelstart
modeltime = str(datetime.timedelta(seconds=modelsec)).split(".")[0]
f = open(modelinfodir, 'a')
f.write(f"Learning Finished! It took {modeltime}.\n")
f.close()
print(f"Learning Finished! Learning time: {modeltime}")


#개선점 train acc에 비해 test acc가 너무 낮음(데이터 갯수는 충분한데)
#일단 label을 one hot encoding->normal distribution으로 바꿔서 fix된 정답이 아니라 probability로 제공
#그래도 안되면 layer 깊이 줄이고
#K-fold Cross Evaluation은 후순위로 고려, 일단은 모델 구조부터 생각해서 test accuracy 향상