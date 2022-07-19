# -*- coding: utf-8 -*-

import glob
import numpy as np
from obspy import read,UTCDateTime,Trace,Stream
import os
import shutil

#Output 폴더 초기화
resultdir=r'/home/bssuh/DATA/out01/'
files = glob.glob(os.path.join(resultdir,"*"))
shutil.rmtree(r'')

####
print("Output folder RESET")

def folder_reset(dir):
    try:
        if os.listdir(dir):
            for (path, dir, files) in os.walk(os.path.abspath(dir)):
                print(path, files, sep='\t')

        excute()

    except Exception as e:
        return e

def excute():
    print('*'*100)
    print('* 삭제되는 디렉토리와 파일 리스트 입니다. 절대경로로 표기합니다.')
    print('* 경고! rmtree 메소드는 하위 디렉토리 및 모든파일들을 삭제합니다.')
    print('* 휴지통으로 복구할 수 없습니다. 필요시 반드시 백업하시기 바랍니다.')

    deleteKey = input('%s 해당폴더를 삭제하시겠습니까 [Y/N]' %os.path.abspath(dir))

    if deleteKey == 'y':
        try:
            shutil.rmtree(dir)
        except Exception as e:
            print(e)
    else:
        print('중단되었습니다.')
    
#폴더에서 파일경로 리스트 작성:1성분(Z)만 추출
#file_lst_z=glob.glob(r'/home/bssuh/DATA/')
#path=r'C:\Users\RacoonFox\Desktop\Undergraduate\seism\GJ_Data\2016091911335852'
#bs='\\'
#total = len(file_lst_z)
#iteration = 0
#i = r'/home/bssuh/DATA/'     
#리스트의 각 원소에 대해 파일명을 찾고 각각의 변수 추출
#for i in file_lst_z:
#    fname=i.split('\\')[-1]
#    network = fname.split('.')[0]
 #   station = fname.split('.')[1]
  #  event = fname.split('.')[-2]
#    ch = (fname.split('.')[-3])
 #   channel = ch[2]
 #   ch_rest = ch[0:2]
   
    #추출한 변수에 대해 Z 성분과 channel 이외의 변수는 동일한 N,E 성분 파일 경로 선언
   # dir_z=path+bs+network+bs+network+'.'+station+'..'+ch_rest+'Z'+'.'+event+'.mseed'    
  #  dir_n=path+bs+network+bs+network+'.'+station+'..'+ch_rest+'N'+'.'+event+'.mseed'
   # dir_e=path+bs+network+bs+network+'.'+station+'..'+ch_rest+'E'+'.'+event+'.mseed'
        
    #Z,N,E 성분 파일을 읽어서 변수 지정
  #  st_z=read(dir_z, format = "MSEED")
 #   st_n=read(dir_n, format = "MSEED")
   # st_e=read(dir_e, format = "MSEED")
    
    #sampling rate 100.0으로 resample
#    st_z.resample(100.0)
 #   st_n.resample(100.0)
  #  st_e.resample(100.0)
    
    #파일명으로 시작시간을 잡고 +60초까지 범위로 잘라내기
#    start_t = UTCDateTime(event)
 #   st_z.trim(starttime = start_t, endtime = start_t + 60)
  #  st_n.trim(starttime = start_t, endtime = start_t + 60)
   # st_e.trim(starttime = start_t, endtime = start_t + 60)

    #3성분의 stream을 하나의 stream으로 가공
 #   st = Stream()
#    st = st_z + st_n + st_e
    
    #3*(크기 다름) 크기의 빈 array 생성 후 각각의 잘린 파일을 하나의 array 안에 Z>N>E 순서로 삽입
 #   nparray = np.empty((0,st_z[0].count()))
#    nparray = np.append(nparray,st_z,axis=0)
 #   nparray = np.append(nparray,st_n,axis=0)
   # nparray = np.append(nparray,st_e,axis=0)
  #  nparray = nparray[:, 0:6000]

    #st와 array를 각각 mseed와 npy 확장자로 각각 저장
  #  resultdir_mseed = resultdir+bs+network+'.'+station+'..'+ch_rest+'ZNE'+'.'+event+'.mseed'
   # resultdir_npy = resultdir+bs+network+'.'+station+'..'+ch_rest+'ZNE'+'.'+event+'.npy'

 #   st.write(resultdir_mseed,format='MSEED')
  #  np.save(resultdir_npy,nparray)

    #진행 상황 확인을 위해 카운트
 #   iteration = iteration+1
  #  print(f'{iteration} of {total} complete')
#print('Done')

         