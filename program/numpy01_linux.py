# -*- coding: utf-8 -*-

import glob
import numpy as np
from obspy import read,UTCDateTime,Trace,Stream
import os
import shutil
import pathlib

#0 Output folder path
resultdir=r'/home/bssuh/OUTPUT/out01/'

#0 special character definition
bs='\\'
slash='/'
dot='.'
nextline = '\n'
errornum = 0
normalnum = 0
iteration = 0
nofilenum = 0
#0 Error Test Directory
errortestdir=r'/DATA/bssuh/repo_bssuh_seism/program/Test/error.txt'
open(errortestdir,'w').close()
error = open(errortestdir,'a')    #error.write("RESULT"+'\n')

#0 Output folder reset
if os.path.exists(resultdir):
    dirlist = os.listdir(resultdir)
    print('*'*50)
    print('* List of directories and files to be DELETED! listed in abspath.')
    print('* WARNING! rmtree method deletes everything in the folder!')
    print('* UNABLE TO RECOVER after delete. Backup before DELETE!\n')
    print(resultdir,dirlist,sep='\n')
    deleteKey = input('%s Are you sure to format the output folder? [Y/N]' %os.path.abspath(resultdir))
    
    if deleteKey == 'y':
        try:
            shutil.rmtree(resultdir)
            os.mkdir(resultdir)
            print("Output folder RESET")
        except Exception as e:
            print(e)
    else:
        print('STOP')
        exit(0)
else:
     print('No Directory')
     print('Create result folder')
     os.mkdir(resultdir)
     print(resultdir)

#0 for total data    
DATAtopdir=r'/home/bssuh/DATA/*'
yrlist = list(filter(os.path.isdir,glob.glob(DATAtopdir)))

#0 for single year
for yr in yrlist:
    year=yr.split(slash)[-1]
    print(f'{year} in process')
    yrdir=yr    # evdir=r'/home/bssuh/DATA/2019'
    yrdir_in=yrdir+'/*'
    catalog='eventlist_'+year+'.txt'
    catapath=os.path.join(yr,catalog)
    with open(catapath) as file:

#2 for single event
        for line in file:
            evkst=line.rstrip().split(' ')[0]  #directory name
            evkst=evkst.split('\t')[0]  # only 2016 eventlist partitioned by tab(NOT SPACE!!!)
            evutc=UTCDateTime(evkst) - 9*60*60
            evdir=yrdir+slash+evkst    # evdir=r'/home/bssuh/DATA/2019/201912300144'

#3 for single event make file list     
            filelst_hgz=glob.glob(evdir+slash+'*.K?.HGZ')
            filelst_hhz=glob.glob(evdir+slash+'*.K?.HHZ')
            filelst_elz=glob.glob(evdir+slash+'*.K?.ELZ')
            filelst_z=filelst_hgz+filelst_hhz+filelst_elz

#3 for file list
            for f in filelst_z:
                iteration+=1
                path=f    #path=r'/DATA/bssuh/DATA/2019/201912300144/CHOA.KS.HGZ'
                fname = path.split(slash)[-1]
                network = fname.split('.')[1]
                station = fname.split('.')[0]
                ch = fname.split('.')[-1]
                channel = ch[2]
                ch_rest = ch[0:2]

#4 for single file make path
                dir_z=path[0:-1]+'Z'
                dir_n=path[0:-1]+'N'
                dir_e=path[0:-1]+'E'    

#4 read ZNE
                try:
                    st_z=read(dir_z, format = "MSEED")
                    st_n=read(dir_n, format = "MSEED")
                    st_e=read(dir_e, format = "MSEED")
                except:
                    errornum+=1
                    nofilenum+=1
                    if os.path.exists(dir_z):
                        z="Z TRUE"
                    else:
                        z="Z FALSE"
                    
                    if os.path.exists(dir_n):
                        n="N TRUE"
                    else:
                        n="N FALSE"
                    
                    if os.path.exists(dir_e):
                        e="E TRUE"
                    else:
                        e="E FALSE"
                    
                    errorline = evkst+'\t'+fname+'\n'+z+'\t'+n+'\t'+e+'\n'
                    error.write(errorline+'\n')

#4 trim
                start_t = UTCDateTime(evutc)
                st_z.trim(starttime = start_t, endtime = start_t + 60)
                st_n.trim(starttime = start_t, endtime = start_t + 60)
                st_e.trim(starttime = start_t, endtime = start_t + 60)

#4 make stream & 3*6000 numpy
                st = Stream()
                st = st_z + st_n + st_e
                
                try:                
                    resultdir_mseed = resultdir+evkst+"KST"+'.'+network+'.'+station+'.'+ch_rest+'ZNE'+'.mseed'
                    st.write(resultdir_mseed,format='MSEED')
                    
                    column = st_z[0].count()
                    nparray = np.empty((0,column))
                    nparray = np.append(nparray,st_z,axis=0)
                    nparray = np.append(nparray,st_n,axis=0)
                    nparray = np.append(nparray,st_e,axis=0)
                    nparray = nparray[:, 0:6000]
                    
                    resultdir_npy = resultdir+evkst+"KST"+'.'+network+'.'+station+'.'+ch_rest+'ZNE'+'.npy'
                    np.save(resultdir_npy,nparray)
                    normalnum+=1
                
                except:
                    errornum+=1
                    try:
                        zpts=st_z[0].stats.npts
                    except:
                        zpts=0
                    
                    try:
                        npts=st_n[0].stats.npts
                    except:
                        npts=0
                    
                    try:
                        epts=st_e[0].stats.npts
                    except:
                        epts=0
                    
                    errorline = evkst+'\t'+fname+'\n'+str(zpts)+'\t'+str(npts)+'\t'+str(epts)+'\n'
                    error.write(errorline+'\n')
#check progress
    print(f'{year} complete')
totalnum=errornum+normalnum
skipnum=iteration-totalnum
error.write('RESULT'+'\t'+'NO FILE='+str(nofilenum)+'\t'+'ERROR='+str(errornum)+'\t'+'NORMAL='+str(normalnum)+' Out of '+str(totalnum)+'\n'+'Skip='+str(skipnum)+' Out of '+str(iteration))
error.close()
print('Done')