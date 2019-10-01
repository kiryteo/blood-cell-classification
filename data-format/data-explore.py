"""Use this to generate individual
images (single cell) in each directory
containing the corresponding .mat file
should be concatenated later to get the
frame (sequence of cell images - its 
movement)."""


import scipy.io
import sys
import numpy
from scipy import misc
from PIL import Image
import imageio
import pandas as pd
import os
import glob
import shutil

filelist = glob.glob('/media/ashwin/Windows/cnrs/dataNorm/*')

#fname = '/media/ashwin/Windows/cnrs/dataNorm/AE20190307shear10s03_Export.mat'
#mat = scipy.io.loadmat('/media/ashwin/Windows/cnrs/dataNorm/AE20190307shear10s01_Export.mat')

wd = os.getcwd()
flist = glob.glob(wd + '/*')

a = flist[0].split('.')[-1]
b = flist[1].split('.')[-1]

if a=='mat':
    fname = flist[0]
else:
    fname = flist[1]

mat = scipy.io.loadmat(fname)
#print(fname)
v1 = mat['Norm_Tab']

v2 = mat['Labels_Num']

for i in range(v1.shape[0]):
    for j in range(12):
        imageio.imwrite('file{}{}.png'.format(str(i+1)+'-',str(j+1) ), v1[i,j])
        #ls = glob.glob(wd + '/*')
        #lfile = max(ls, key=os.path.getctime)
        #shutil.copy(lfile, dirpath)
        #os.remove(lfile)