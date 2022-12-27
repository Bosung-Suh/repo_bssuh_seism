# -*- coding: utf-8 -*-

#CNN의 전체 과정
""" 
대략적인 과정
1. set hyper parameters(learning rate, training epochs, batch size, etc)
2. make data pipelining(use tf.data)    #dataset을 로드하고 설정한 batch size 만큼 데이터를 가져와서 network에 전달
3. build neural network model(tf.keras.sequential API or Functional API)
4. define loss function    #Cross Entropy or Cosine Distance
5. calculate gradient(use tf.GradientTape)    #weight에 대한 gradient 계산
6. select optimizer(Adam optimizer)    #계산한 gradient로부터 weight를 업데이트
7. define metric for model performance(accuracy)    #학습모델의 성능을 판단하는 지표 설정
8. (optional) make checkpoing for saving    #학습 결과를 임시로 저장(선택)
9. train and validate neural network model
10. save model at every epoch    #model이 바뀔 때마다 model 저장
11. model plot    #model 관련 (Epoch vs. MAE&loss, number of dataset vs. error, label vs error(abserror)) plot
12. prediction 결과 csv 파일 저장
"""
#서버 접속 상태로 top 명령어로 CPU 모니터링!!
# 0.import libraries
from __future__ import absolute_import, division, print_function
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

import datetime
import glob
#import multiprocessing
import math
import os

# import fnmatch
# import parmap
import pickle
import random
import shutil
import time
import numpy as np
import scipy.stats as stats

import tqdm  # 연속적인 작업의 진행률을 시각적으로 표현    for i in tqdm.tqdm(list):같은 형식
from obspy import Stream, Trace, UTCDateTime, read
from sklearn.model_selection import KFold, train_test_split

from tensorflow.keras.utils import plot_model, to_categorical

print(f"Tensorflow {tf.__version__}")  #Tensorflow 2.4.1(20220706)

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
training_epochs = 4
batch_size = 10

tf.random.set_seed(957)

# 1.create checkpoint directory
currentdir = os.getcwd()
checkpt_dir_name = 'checkpoints'
modelname = 'dist_cnn_CD_Regression_20221226_modelsave'
modeltopdir = r'/home/bssuh/repo_bssuh_seism/program/model'
resultmodeldir = os.path.join(modeltopdir, modelname)    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq
checkpointdir = os.path.join(resultmodeldir, checkpt_dir_name)    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq/checkpoints

#1 model folder reset
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

#1 create model folder, checkpoint folder
os.makedirs(resultmodeldir, exist_ok=True)
os.makedirs(checkpointdir, exist_ok=True) 

# GPU setting(watch -n 0.1 nvidia-smi 로 GPU monitoring!!)
    #GPU로 돌릴 때는 VSCODE 터미널로 돌리지 말고 WSL 따로 열어서 돌릴 것!!!!!!!!!!
# SSH 연결 끊겨도 돌아가게 할려면 아래 명령어로 실행 
    # nohup python3 ___.py > /home/bssuh/repo_bssuh_seism/program/nohup_log/___.txt 2>&1 &   
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

# 2.data normalization(특성들의 스케일, 범위가 다른 경우)
# 분석하는 특성이 3성분 데이터만 있으므로 정규화 X
""" def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
labels=to_categorical(labels,101) """

""" label1 = labels[1]
label2 = labels[2] """
""" plt.plot(label1,'ro-',label2,'bs-') #list1,2를 y값으로
plt.axis([0,100,0,1]) #x축의 최소, 최대, y축의 최소, 최대
plt.xlabel('index')
plt.ylabel('probabiliry')
plt.legend(['label1','label2'])

plt.show()
plt.savefig(r'/home/bssuh/repo_bssuh_seism/program/Test/plot/one_hot.png',facecolor='#eeeeee') """

# 2. split train, test dataset
traindata, restdata, trainlabel, restlabel = train_test_split(npys, labels, test_size=0.2, shuffle=True, random_state=42)
valdata, testdata, vallabel, testlabel = train_test_split(restdata, restlabel, test_size=0.5, shuffle=True, random_state=42)
# train 81150(0.8), validation data 10144(0.1), test 10144(0.1)

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
    model.add(keras.layers.Dense(1))    #layer를 순서대로 하나씩 추가
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])    #regression이므로 손실함수는 mse, metric은 mse, mae
    
    return model

# 3.print and save model summary
model = create_model()
model.summary()    #model에 대한 정보 표출
plot_model(model, to_file=os.path.join(resultmodeldir, "modelshape.png"), show_shapes=True)    #model 구조도
modelinfodir = os.path.join(resultmodeldir, "modelinfo.txt")    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_seq/modelinfo.txt
with open(modelinfodir, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + nextline))
print("model info save COMPLETE!")    
f = open(modelinfodir, 'a')
f.write("model info save COMPLETE!\n")
f.close()

#regression에서 패키지 내의 mae, mse를 사용하므로 커스텀 함수 불필요
""" # 4.loss function               #결과 확인해보고 나중에 최대 우도 추정법(MLE;Maximum Likelihood Estimation)으로 바꾸는 것 고려
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
    batchlabellist.extend(batchlabel)    #기존 batchlabellist에 이번 batch에서 사용한 label 추가
    error = tf.subtract(labelmax,logitmax)     #정답(label)과 계산값(logit) 사이 거리의 절댓값
    error = tf.cast(error,dtype=tf.int32).numpy()
    errorlist.extend(error)    #기존 error 리스트에 이번 batch에서 계산된 error들 추가
    errordistance = tf.abs(error)
    mean_abs_error = tf.reduce_mean(errordistance)
    return mean_abs_error, errorlist, batchlabellist """

#sample prediction vs label plot을 위해 랜덤 추출
def randomselect(data, label, selectnum, type):
    randomindex = random.sample(range(0,(len(data)-1)),selectnum)
    print(f"Random Selected {type} Index : {randomindex}")
    d={}
    selnum = 0
    for i in randomindex:
        selnum += 1
        choicedata = data[i]
        choicelabel = label[i]
        d["choice{0}".format(selnum)] = (choicedata, choicelabel, i)
    return d, randomindex

#CustomCallback 작성
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model, trainchoicedict, valchoicedict):
        self.model = model
        self.trainchoicedict = trainchoicedict
        self.valchoicedict = valchoicedict
    
    def on_train_begin(self, logs={}):
        self.sample_traindict={}
        for i in self.trainchoicedict.keys():
            choicedata, choicelabel, choiceindex = self.trainchoicedict.get(i)
            choicedata = np.expand_dims(choicedata, axis = 0)
            samplekey = (choicedata, choicelabel, choiceindex)
            self.sample_traindict["{0}".format(samplekey)] = []
            
        self.sample_valdict={}         
        for i in self.valchoicedict.keys():
            choicedata, choicelabel, choiceindex = self.valchoicedict.get(i)
            choicedata = np.expand_dims(choicedata, axis = 0)
            samplekey = (choicedata, choicelabel, choiceindex)
            self.sample_valdict["{0}".format(samplekey)] = []        
        
    def on_epoch_end(self, epoch, logs={}):
        pass
        """ for i in self.sample_traindict.keys():
            sampledata = i[0]
            sampledata = np.expand_dims(sampledata, axis = 0)
            samplelabel = i[1]
            sampleindex = i[2]
            print(sampledata)
            print(samplelabel)
            print(sampleindex)
            predictionlist = self.sample_traindict.get(i)
            print(predictionlist)
            print(len(sampledata))
            sample_predict = model.predict(sampledata)
            predictionlist.append(sample_predict)
            self.sample_traindict[(sampledata, samplelabel, sampleindex)] = predictionlist """ 

"""         for i in self.sample_valdict.keys():
            sampledata = i[0]
            sampledata = np.expand_dims(sampledata, axis = 0)
            samplelabel = i[1]
            sampleindex = i[2]
            predictionlist = self.sample_valdict.get(i)
            sample_predict = model.predict(sampledata)
            predictionlist.append(sample_predict)
            self.sample_valdict[(sampledata, samplelabel, sampleindex)] = predictionlist
        return self.sample_traindict, self.sample_valdict """

""" #랜덤 추출한 데이터로 epoch 별로 plot
def pred_label_plot(choicedict,model,modelname,epoch,Train_or_Test, checkpointdir):
    #prediction
    d={}
    for i in choicedict.keys():
        choicetuple = choicedict.get(i)
        choicedata = choicetuple[0]
        choicedata = np.expand_dims(choicedata, axis = 0)
        choicelabel = choicetuple[1]
        choiceindex = choicetuple[2]
        choicepredict = model.predict(choicedata)
        choicepredictlist = np.squeeze(choicepredict, axis=0)
        d["{0}".format(i)] = (choicepredictlist, choicelabel, choiceindex)

    #plot(predictplot)
    xlist=list(range(0,len(choicepredictlist)))
    selectnum = len(d)

    n = 0
    for i in d:
        n += 1
        predicttuple = d.get(i)
        predictlist = predicttuple[0]
        predictlabel = predicttuple[1]
        choiceindex = predicttuple[2]

        plt.subplot(2,2,n)
        plt.plot(xlist, predictlist, color='blue')
        plt.title(choiceindex)
        plt.ylabel("calculation")
        plt.xticks(range(0,len(xlist),10))
        ymin, ymax = plt.ylim()
        plt.vlines(predictlabel, ymin, ymax, color='red', linestyle='--')
    plt.suptitle(f"{modelname}_Epoch {epoch+1}\nPrediction Result for randomly selected {selectnum} {Train_or_Test}data",y=1.05)

    plt.tight_layout()
    plt.show()
    filename = f"RandomCalculation_{Train_or_Test}_epoch{str(epoch+1)}.png"
    modelplotdir = os.path.join(checkpointdir, filename)
    plt.savefig(modelplotdir,bbox_inches='tight',facecolor='#eeeeee')
    plt.clf()    

# 8.create checkpoint for save
checkpoint = tf.train.Checkpoint(cnn=model) """

""" # 9.training 함수 정의
@tf.function
def train(model, images, labels):
    grads = grad(model, images, labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) """

# 9.model 학습(modelling time measure 포함)
f = open(modelinfodir, 'a')
f.write(f"Total data number is {len(labellist)}, Data processing took {processtime}.\n")
f.write(f"{len(traindataset)} for training, {len(valdataset)} for validation, {len(testdataset)} for test.\n")
f.write(f"learning rate = {learning_rate} epoch = {training_epochs}\n")
print('Learning started. It takes sometime.')
f.write('Learning started. It takes sometime.\n')
f.close()
modelstart = time.time()
#losslist = []
#train_mae_list = []
#test_mae_list = []

randomselectnum = 4
f = open(modelinfodir, 'a')
train_random_dict, trainrandomindex = randomselect(traindata,trainlabel,randomselectnum, "Train")
f.write(f"{randomselectnum} randomly selected traindata index : {trainrandomindex}\n")
val_random_dict, valrandomindex = randomselect(valdata,vallabel,randomselectnum, "Validation")
f.write(f"{randomselectnum} randomly selected valdata index : {valrandomindex}\n")
test_random_dict, testrandomindex = randomselect(testdata, testlabel, randomselectnum, "Test")
f.write(f"{randomselectnum} randomly selected testdata index : {testrandomindex}\n")
f.close()

history = model.fit(x=traindataset, 
                    y=None,    #dataset이 (data, label) 형태이니 불필요
                    batch_size=None,    #dataset 만들 때 이미 batch size로 잘라주었으니 여기서는 None
                    epochs=training_epochs, 
                    verbose=2,    #epoch 진행상황을 출력(1:progress bar, 2: 시간, loss, metric 등 숫자만 1줄로 출력)
                    callbacks=[CustomCallback(model, train_random_dict, val_random_dict)], 
                    validation_data=valdataset, 
                    initial_epoch=0,    #학습이 시작되는 epoch
                    validation_freq=1,    #validation for every(1) epochs
)


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()





""" for epoch in range(training_epochs):    # 1 training epoch가 1 cycle
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

    pred_label_plot(train_random_dict,model,modelname,epoch,"Train", checkpointdir)
    pred_label_plot(test_random_dict,model,modelname,epoch,"Test", checkpointdir)

    epochresult = f"Epoch: {epoch + 1} loss = {avg_loss:.8f} train MAE = {avg_train_mae:.4f}km test MAE = {avg_test_mae:.4f}km"
    print(epochresult)    #epoch 끝날 때마다 training, test 정확도 출력
    f = open(modelinfodir, 'a')
    f.write(epochresult + nextline)
    f.close()

# Error Histogram for each epoch(hist)
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

# Error vs Label plot for each epoch(plot1)
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
    plot1name="Label_Error_plot_epoch"+str(plotnum)
    plot1dir = os.path.join(checkpointdir, plot1name)
    plt.savefig(plot1dir,facecolor='#eeeeee')
    plt.clf()

# Absolute Error vs Label plot for each epoch(plot2)
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
    plot2name="Label_AbsError_plot_epoch"+str(plotnum)
    plot2dir = os.path.join(checkpointdir, plot2name)
    plt.savefig(plot2dir,facecolor='#eeeeee')
    plt.clf()

# Logit vs Label plot for each epoch, (plot3)
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
    plot1name="Label_Error_plot_epoch"+str(plotnum)
    plot1dir = os.path.join(checkpointdir, plot1name)
    plt.savefig(plot1dir,facecolor='#eeeeee')
    plt.clf()

    checkpoint.save(file_prefix=os.path.join(checkpointdir, "checkpoint"))    #epoch 끝날 때마다 checkpoint save
    modelfilename = "Model_epoch"+str(plotnum)+".h5"
    modelsavedir = os.path.join(checkpointdir,modelfilename)
    model.save(modelsavedir)    #epoch 끝날 때마다 model 자체를 .h5 형식으로 저장해 나중에 불러옴
modelend = time.time()
modelsec = modelend - modelstart
modeltime = str(datetime.timedelta(seconds=modelsec)).split(".")[0]

# model loss, accuracy plot(modelplot)
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

f = open(modelinfodir, 'a')
f.write(f"Learning Finished! It took {modeltime}.")
f.close() """
modelend = time.time()
modelsec = modelend - modelstart
modeltime = str(datetime.timedelta(seconds=modelsec)).split(".")[0]
print(f"Learning Finished! Learning time: {modeltime}")

#개선점 train acc에 비해 test acc가 너무 낮음(데이터 갯수는 충분한데)
#일단 label을 one hot encoding->normal distribution으로 바꿔서 fix된 정답이 아니라 probability로 제공
#그래도 안되면 layer 깊이 줄이고
#K-fold Cross Evaluation은 후순위로 고려, 일단은 모델 구조부터 생각해서 test accuracy 향상
