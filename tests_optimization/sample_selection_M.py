# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 09:15:45 2021

@author: sega01
"""
import numpy as np
import pandas as pd
import pickle
import copy
from matplotlib import cm as CM
from matplotlib.lines import Line2D
from elasticsearch import Elasticsearch
import pickle
from cvxopt import solvers, matrix, spdiag, log, exp, div
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pickle
import datetime
import math
source=['Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val', 'SuctSH_Circ1', 'ControlRegCompAC.VarFrequencyHzMSK', 'CondTempCirc1', 'EbmpapstFan_1_Mng.InfoSpeed_EBM_1.CurrSpeed', 'DscgTempCirc1', 'ExtTemp', 'W_OutTempEvap', 'W_OutTempUser', 'W_InTempUser']
objective='EvapTempCirc1'
populations=['population_pool.pkl','population_pool_2.pkl']
with open('error.pkl', 'rb') as handle:
    e = pickle.load(handle)
with open('phi.pkl', 'rb') as handle:
    phi = pickle.load(handle)
with open(populations[1], 'rb') as handle:
    estims = pickle.load(handle)
    
#samples=pd.DataFrame(phi,columns=source)
labs=[]
k=5
for i in range(k):
    labs.append(i)
M=np.zeros((len(source),len(source),k,k))
d={}
for i in range(phi.shape[1]):
    d[source[i]]=pd.cut(np.ravel(phi[:,i]),k,labels=[0,1,2,3,4])
df=pd.DataFrame(d)
permutations=[]
for s in range(len(source)-1):
    for j in range(s+1,len(source)):
        permutations.append((s,j))
        
# we take one sample and fill the corresponding fields in M directly
def fill_M(M,sample,perm,k=5):
    for s,j in perm:
        M[s,j,sample[source[s]],sample[source[j]]] += 1
    return M
# we get back all the locations in a list of samples 
def forge_Sig(sample_set,perm,n=10,k=5):
    sig=[]
    for d in sample_set:
        for s,j in perm:
            sig.append([s,j,d[source[s]],d[source[j]]])
    return sig
# now given the sig, create Ms
def create_m(sig,n=10,k=5):
    M=np.zeros((n,n,k,k))
    for s in sig:
        M[s[0],s[1],s[2],s[3]] += 1
    return M
# we get the general marking of the whole sample set in M
for i in range(df.shape[0]):
    M=fill_M(M,df.iloc[i],permutations,k=k)
print('Elements Filled: {} | Elements Empty: {}'.format(len(M[M>0]),len(M[M==0])))
M=M/np.sum(M)
    
# now we aim to find subset that keeps same proportion 
def splitDataFrameIntoSmaller(df, chunkSize = 10): #10 for default 
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf
subsets=splitDataFrameIntoSmaller(df,chunkSize=50)


