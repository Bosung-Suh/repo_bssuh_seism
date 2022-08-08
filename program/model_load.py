
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
#import fnmatch
import parmap
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
import csv

#data processing time measure start
processstart = time.time()
# core number count(activate import multiprocessing)
#cores = multiprocessing.cpu_count()    #40

#0 special character definition
bs='\\'
slash='/'
dot='.'
nextline = '\n'
iteration = 0

# Recreate the exact same model, including its weights and the optimizer
modelname = "dist_Unet_CD_OH_20220707_lr0_0001_modelsave"
modelepoch = 3
modelepochname = "Model_epoch"+str(modelepoch)+".h5"

#0 directory of data
data_dir=r'/home/bssuh/jwhan/npy_data_test'

# 1.checkpoint directory
modeltopdir = r'/home/bssuh/repo_bssuh_seism/program/model'
resultmodeldir = os.path.join(modeltopdir, modelname)
checkpointdir = os.path.join(resultmodeldir, "checkpoints")
modelfiledir = os.path.join(checkpointdir, modelepochname)    #/home/bssuh/repo_bssuh_seism/program/model/dist_cnn_CD_OH_20220316_modelsave/checkpoints/Model_epoch1.h5
modelcalcdir = os.path.join(resultmodeldir, "calculation")
os.makedirs(modelcalcdir, exist_ok=True)

stationinfocsv = r'/home/bssuh/jwhan/krs2109.csv'

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

#특정 event의 npy만 추출
oneevent = "2016256_113254"    
#경주지진 5.8= 2016256_113254
#포항지진 5.4= 2017319_052931

npylist = sorted(glob.glob(data_dir+slash+oneevent+'.*.npy'))    #/home/bssuh/jwhan/npy_data_test/2016216_235813.KS.YOCB.HH_42.18_.npy

csvname = f"prediction_{oneevent}.csv"
with open(os.path.join(modelcalcdir,csvname), 'w',newline='') as f:   
# field names  
    fields = ['Event', 'Network', 'Station', 'Channel', 'Logit', 'Label']  
    write = csv.writer(f) 
    write.writerow(fields)

#model load
#custom loss, metric은 불러와서 지정해야 함
def loss_fn(model, images, labels):    #labels=정답
    logits = model(images, training=True)    #training True로 설정하면 model의 dropout layer에서 dropout이 적용
    cos_similarity = tf.keras.losses.cosine_similarity(y_true=labels, y_pred=logits,axis=-1)    #(-1)*cosine similarity[-1,1] 계산(벡터 방향 같으면 -1, 반대면 +1, 직각이면 0)
    cos_distance = tf.add(cos_similarity,1)    #cosine distance[0,2] 계산(벡터 방향 같으면 0, 반대면 +2, 직각이면 +1)
    loss = tf.reduce_mean(cos_distance)     #cosine distance의 평균을 loss로 반환(벡터 방향이 같을수록 loss 값이 작아짐)
    return loss
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

model = keras.models.load_model(modelfiledir, custom_objects={'loss_fn': loss_fn, 'evaluate': evaluate},compile=True)
model.summary()

with open(os.path.join(modelcalcdir,csvname), 'a',newline='') as f:
    for i in tqdm.tqdm(npylist):
        fname = i.split(slash)[-1]    #2016216_235813.KS.YOCB.HH_42.18_.npy
        event = fname.split(".")[0]    #2016216_235813
        network = fname.split(".")[1]    #KS
        station = fname.split(".")[2]    #YOCB
        ch = fname.split(".")[3].split("_")[0]    #HH
        labelfloat = float(fname.split('_')[2])    # 42.18(float)
        labellist = [labelfloat]
        
        label = np.array(labellist)
        npy = load_data(i)

        #dataframe to float32 type
        npy = npy.astype(np.float32)
        label = label.astype(np.float32)

        #prediction for 1 npy
        prediction = model.predict(npy)
        logit = np.argmax(prediction,axis=1)[0]
        
        # data rows of csv file  
        row = [event, network, station, ch, logit, labelfloat]
        # using csv.writer method from CSV package 
        write = csv.writer(f) 
        write.writerow(row)
        
        iteration+=1
print(f"Single event {oneevent} = {iteration} stations")
""" 
#basemap 한반도 지도
plt.figure(figsize=(10,10))

map = Basemap(projection='merc', lat_0=37.35, lon_0=126.58, resolution = 'h',
    urcrnrlat=40, llcrnrlat=32, llcrnrlon=121.5, urcrnrlon=132.5)

map.drawcoastlines()
map.drawcountries()
map.drawmapboundary()

#관측소 csv load
f = open(stationinfocsv, 'r', encoding= 'utf-8')
readcsv = csv.reader(f)
for line in readcsv:
    print(line)
f.close



lon = 126.58
lat = 37.35
x,y = map(lon, lat)
map.plot(x, y, 'bo', markersize=4)

plt.show()
radiusplotfilename = f"Radiusplot_{modelname}_epoch{str(modelepoch)}_{oneevent}.png"
radiusplotdir = os.path.join(modelcalcdir, radiusplotfilename)
plt.savefig(radiusplotdir,bbox_inches='tight',facecolor='#eeeeee')
plt.clf()    

 """