# -*- coding: utf-8 -*-

import glob
import numpy as np
from obspy import read,UTCDateTime,Trace,Stream
import os
import shutil
import pathlib

#0 Output folder path
resultdir=r'/home/bssuh/program/Test/'

#0 special character definition
bs='\\'
slash='/'
dot='.'
nextline = '\n'
#os.mkdir(resultdir)

#0 for total data    
Zdir=r'/home/bssuh/program/Test/453001648..0.2.2022.02.15.03.11.33.000.Z.sac'
Ndir=r'/home/bssuh/program/Test/453001648..0.2.2022.02.15.03.11.33.000.N.sac'
Edir=r'/home/bssuh/program/Test/453001648..0.2.2022.02.15.03.11.33.000.E.sac'

fname=os.path.basename(Zdir)
resultfname = fname[:39]+"ZNE"+fname[40:]
print(resultfname)
st_z=read(Zdir, format = "sac")
print(st_z)
st_n=read(Ndir, format = "sac")
print(st_n)
st_e=read(Edir, format = "sac")
print(st_e)
#4 make stream & 3*6000 numpy
st = Stream()
st = st_z + st_n + st_e
print(st)
resultdir_sac = os.path.join(resultdir,resultfname)
st.write(resultdir_sac,format='sac')
print('Done')
print(read(r'/home/bssuh/program/Test/453001648..0.2.2022.02.15.03.11.33.000.ZNE01.sac',format='sac'))
print(read(r'/home/bssuh/program/Test/453001648..0.2.2022.02.15.03.11.33.000.ZNE02.sac',format='sac'))
print(read(r'/home/bssuh/program/Test/453001648..0.2.2022.02.15.03.11.33.000.ZNE03.sac',format='sac'))