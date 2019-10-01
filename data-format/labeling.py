import csv
import glob
import os
import scipy.io
from PIL import Image
import pandas as pd

matlist = glob.glob('/media/ashwin/Windows/blood-cells/files/*')

for eachmat in matlist:
    csvname = eachmat.split('/')
    csvname = '/'+csvname[1]+'/'+csvname[2]+'/'+csvname[3]+'/'+csvname[4]+'/'+csvname[5]+'/'+'0csvs/'+csvname[6][:2] + csvname[6][7:10]+csvname[6][-3:]+'.csv'
    print(csvname)

    mat = scipy.io.loadmat(eachmat)
    v1 = mat['Norm_Tab']
    df = pd.DataFrame()
    
    try:
        df = pd.read_csv(csvname)
    except:
        print("No file")

    labels = []
    for i in range(v1.shape[0]):
        labels.append(mat['Labels_Num'][i][0])


    lb = pd.Series(labels)

    df['Type'] = lb

    df.to_csv(csvname)