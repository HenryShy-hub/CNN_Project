import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import timedelta

from matplotlib import pyplot as plt

data_path = r'D:\大学文档\MAFM\AI in fintech\project 2\Stock_CNN-main'
os.chdir(data_path)
model_result = glob.glob('./*result.csv')
model_NV = glob.glob('./*NV.csv')
result_name = list(map(lambda x: re.findall(r"\\([^\\]*?)_result", x)[0],model_result))
NV_name = list(map(lambda x: re.findall(r"\\([^\\]*?)_NV", x)[0],model_NV))
result_df = pd.concat(list(map(lambda x:pd.read_csv(x,index_col=0),model_result)),axis=1)
NV_df = pd.concat(list(map(lambda x:pd.read_csv(x,index_col=0),model_NV)),axis=1)
NV_df.index = pd.to_datetime(NV_df.index)
result_df.columns,NV_df.columns = result_name,NV_name
NV_df = pd.concat([pd.DataFrame(index=[NV_df.index[0]-timedelta(days=30)],data=[[1 for i in range(len(NV_df.columns))]],columns=NV_df.columns)
                  ,NV_df],axis=0)
NV_df.plot()
plt.title('NV of all models',fontsize=16)
plt.show(block=True)
result_df.to_csv('Model results.csv')