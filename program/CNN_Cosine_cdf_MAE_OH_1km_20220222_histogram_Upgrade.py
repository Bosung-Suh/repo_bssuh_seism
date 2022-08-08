# -*- coding: utf-8 -*-
#https://gooopy.tistory.com/86?category=876252 참고
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
learning_rate = 0.0001
training_epochs = 8
batch_size = 10

tf.random.set_seed(957)

# 1.create checkpoint directory
currentdir = os.getcwd()
checkpt_dir_name = 'checkpoints'
modelname = 'dist_cnn_CD_OH_20220302_histogram'

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
    labelround = float(round(labelfloat))    #42(float)
    labellist.append(labelround)
print(f"Total data number is {len(labellist)}")    # Total data number is 101437

labels = np.array(labellist)
npys = np.vstack(dataload)

# 2. dataframe to float32 type
npys = npys.astype(np.float32)
labels = labels.astype(np.float32)

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
""" label1 = labels[1]
label2 = labels[2] """
""" plt.plot(label1,'ro-',label2,'bs-') #list1,2를 y값으로
plt.axis([0,100,0,1]) #x축의 최소, 최대, y축의 최소, 최대
plt.xlabel('index')
plt.ylabel('probabiliry')
plt.legend(['label1','label2'])

plt.show()
plt.savefig(r'/home/bssuh/repo_bssuh_seism/program/Test/plot/one_hot.png',facecolor='#eeeeee') """

# 2. split train, test, validation dataset(train은 학습, test는 최종 평가, validation은 학습 도중 평가)
traindata, restdata, trainlabel, restlabel = train_test_split(npys, labels, test_size=0.2, shuffle=True, random_state=42)
valdata, testdata, vallabel, testlabel = train_test_split(restdata, restlabel, test_size=0.5, shuffle=True, random_state=42)
#Total 101437= train 81149(0.8)+val 10144(0.1)+test 10144(0.1)

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
#순차 모델(sequential)로 네트워크 구성(레이어가 선형으로 연결되며, 단일 텐서 입력, 단일 출력에 사용. 다중 입출력이나 레이어를 공유하는 경우에는 사용 안 함)
def create_model():    
    model = keras.Sequential()    #API 선언(.add를 사용해 layer를 순서대로 하나씩 삽입)
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
    model.add(keras.layers.Dense(1024, activation=tf.nn.relu))   #dense layer(완전 연결 계층)
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
f = open(modelinfodir, 'a')
f.write("model info save COMPLETE!\n")

# 4.loss function               #결과 확인해보고 나중에 최대 우도 추정법(MLE;Maximum Likelihood Estimation)으로 바꾸는 것 고려
def loss_fn(model, images, labels):    #labels=정답
    logits = model(images, training=True)    #training True로 설정하면 model의 dropout layer에서 dropout이 적용
    cos_similarity = tf.keras.losses.cosine_similarity(y_true=labels, y_pred=logits,axis=-1)    #(-1)*cosine similarity[-1,1] 계산(벡터 방향 같으면 -1, 반대면 +1, 직각이면 0)
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
def evaluate(model, images, labels, errorlist, batchlabellist):
    logits = model(images, training=False)    #training이 아니므로 기능 끄기
    logitmax = tf.argmax(input=logits,axis=1,output_type=tf.dtypes.int32)
    labelmax = tf.argmax(input=labels,axis=1,output_type=tf.dtypes.int32)
    batchlabel = tf.cast(labelmax,dtype=tf.int32).numpy()
    batchlabellist.extend(batchlabel)
    error = tf.subtract(labelmax,logitmax)     #정답(label)과 계산값(logit) 사이 거리의 절댓값
    error = tf.cast(error,dtype=tf.int32).numpy()
    errorlist.extend(error)    #기존 error 리스트에 이번 batch에서 계산된 error들 추가
    errordistance = tf.abs(error)
    mean_abs_error = tf.reduce_mean(errordistance)
    return mean_abs_error, errorlist, batchlabellist

# 6.AdamOptimizer(학습 방식 설정)
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
    train_error_list = []
    test_error_list = []
    train_label_list = []
    test_label_list = []
    train_step = 0
    test_step = 0
        
    for images, labels in traindataset:    # training epoch 내에서 1 batch size가 1 cycle
        train(model, images, labels)
        loss = loss_fn(model, images, labels)
        mae_train, train_error_list, train_label_list = evaluate(model, images, labels, train_error_list, train_label_list)    #학습 중간에 loss, acc 계산
        sum_loss += loss
        sum_train_mae += mae_train
        train_step += 1
        
    avg_loss = sum_loss / train_step
    avg_train_mae = sum_train_mae / train_step    #평균 계산
    
    losslist.append(avg_loss)
    train_mae_list.append(avg_train_mae)
    
    for images, labels in testdataset:      # 1 epoch가 끝나면 test dataset을 넣고 모델 검증 
        mae_test, test_error_list, test_label_list = evaluate(model, images, labels, test_error_list, test_label_list)        
        sum_test_mae += mae_test
        test_step += 1
            
    avg_test_mae = sum_test_mae / test_step

    test_mae_list.append(avg_test_mae)
        
    epochresult = f"Epoch: {epoch + 1} loss = {avg_loss:.8f} train MAE = {avg_train_mae:.4f}km test MAE = {avg_test_mae:.4f}km"
    print(epochresult)    #epoch 끝날 때마다 training, test 정확도 출력
    f.write(epochresult + nextline)

# Error Histogram for each epoch
    xmax = 30
    xmin = -30
    xnumber = xmax-xmin+1 
    plotnum = epoch+1

    train_error_array = np.array(train_error_list)
    test_error_array = np.array(test_error_list)

    train_outrange = np.count_nonzero(train_error_array < xmin) + np.count_nonzero(train_error_array > xmax)
    test_outrange = np.count_nonzero(test_error_array < xmin) + np.count_nonzero(test_error_array > xmax)

    train_np_mean = np.mean(train_error_list)
    train_np_median = np.median(train_error_list)
    train_np_std = np.std(train_error_list)

    test_np_mean = np.mean(test_error_list)
    test_np_median = np.median(test_error_list)
    test_np_std = np.std(test_error_list)    

    plt.subplot(2, 1, 1)               
    plt.hist(train_error_list, bins=xnumber, range=(xmin,xmax))
    plt.title(f"{modelname}_Epoch {plotnum}")
    plt.ylabel('Number of dataset')
    plt.xlabel('Train Error(km)')
    plt.xlim(xmin, xmax)
    ymin, ymax = plt.ylim()
    plt.text(5, round(ymax/2), f"Mean : {train_np_mean:.3f} \nMedian : {train_np_median} \nStd : {train_np_std:.4f}\nOut of Range : {train_outrange} of {len(train_error_list)}")

    plt.subplot(2, 1, 2)               
    plt.hist(test_error_list, bins=xnumber, range=(xmin,xmax))
    plt.ylabel('Number of dataset')
    plt.xlabel('Test Error(km)')
    plt.xlim(xmin, xmax)
    ymin, ymax = plt.ylim()
    plt.text(5, round(ymax/2), f"Mean : {test_np_mean:.3f} \nMedian : {test_np_median} \nStd : {test_np_std:.4f}\nOut of Range : {test_outrange} of {len(test_error_list)}")

    plt.tight_layout()
    plt.show()
    histname="histogram_epoch"+str(plotnum)
    histplotdir = os.path.join(checkpointdir, histname)    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq/modelplot.png
    plt.savefig(histplotdir,facecolor='#eeeeee')
    plt.clf()

# Error vs Label plot for each epoch
    train_label_array = np.array(train_label_list)
    test_label_array = np.array(test_label_list)

    plt.subplot(2, 1, 1)               
    plt.scatter(train_error_array, train_label_array,s=4)
    plt.title(f"{modelname}_Epoch {plotnum}")
    plt.ylabel('Train Label(km)')
    plt.xlabel('Train Error(km)')
    plt.ylim(0, 100)

    plt.subplot(2, 1, 2)               
    plt.scatter(test_error_array, test_label_array,s=4)
    plt.ylabel('Test Label(km)')
    plt.xlabel('Test Error(km)')
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.show()
    plotname="Label_Error_plot_epoch"+str(plotnum)
    plotdir = os.path.join(checkpointdir, plotname)
    plt.savefig(plotdir,facecolor='#eeeeee')
    plt.clf()

# Absolute Error vs Label plot for each epoch
    train_Abserror_array = np.abs(train_error_array)
    test_Abserror_array = np.abs(test_error_array)

    plt.subplot(2, 1, 1)               
    plt.scatter(train_Abserror_array, train_label_array,s=4)
    plt.title(f"{modelname}_Epoch {plotnum}")
    plt.ylabel('Train Label(km)')
    plt.xlabel('Train Error(km)')
    plt.ylim(0, 100)

    plt.subplot(2, 1, 2)               
    plt.scatter(test_Abserror_array, test_label_array,s=4)
    plt.ylabel('Test Label(km)')
    plt.xlabel('Test Error(km)')
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.show()
    plotname="Label_AbsError_plot_epoch"+str(plotnum)
    plotdir = os.path.join(checkpointdir, plotname)
    plt.savefig(plotdir,facecolor='#eeeeee')
    plt.clf()

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

#개선점(2/23) train acc에 비해 test acc가 너무 낮음 ->이거는 계속 찾아보기(Batch Normalization? Dropout을 layer 마다?)
#validation set고려(train 0.8:test 0.1:validation 0.1?)
#error vs label 분포도 한번 그려보기
#K-fold Cross Evaluation은 후순위로 고려, 일단은 모델 구조부터 생각해서 MAE 향상

#https://www.iris.edu/hq/
#https://service.iris.edu/irisws/rotation/docs/1/help/ ->backazimuth 관련 내용


#알아본 거 메모
#2/23 6000개를 어디서 부터 자르는가 봤더니 그냥 reshape 함수가 사용 되었음 => 무지성으로 시작에서부터 6000번째 값까지 자른 거일수도????


#해야 할 거
#위에 모델을 class화(중요)>>>model.fit, hist 사용하기 위해
#MEMO
""" model.fit(x_train, y_train, epochs = epoch)
#model.fit():model에 대해 학습 시작(datasets, labels 등 지정)
#Epoch 1/3
#32/32 [==============================] - 1s 2ms/step - loss: 1.7292 - mae: 1.1988
#Epoch 2/3
#32/32 [==============================] - 0s 1ms/step - loss: 0.7596 - mae: 0.7407
#Epoch 3/3
#32/32 [==============================] - 0s 1ms/step - loss: 0.3739 - mae: 0.4898
#<tensorflow.python.keras.callbacks.History at 0x7eff1411a6a0> #이런 식으로 표출 됨(history)
model.predict(x_test.reshape(x_test.shape[0]))
#model.predict(array):입력 array에 대해 모델이 순방향으로 연산되어 나온 결과 출력
#입력 array가 (n,1) (데이터가 행 단위로 떨어져 있음)라면 predict에는 reshape로 (n,)로 넣어줘야 됨 """





