# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:23:33 2020

@author: sega01
"""
import os, requests, uuid, json, pickle
from os import path
import pandas as pd
from fault_detector_class_ES import fault_detector
import math
import matplotlib.pyplot as plt
import numpy as np


def forecast_Brown(X,k,w=4,N=50): 
     # Given a time series X, forecast of k steps ahead wanted and an evaluation window w (this must be even number)
     # Using Brown's double exponential smoothing in two steps:
     # STEP 1: alpha parameter selection --> testing N options
     s_1={}
     s_2={}
     F={}
     E={}
     alpha={}
     a={}
     b={}
     ids=[]
     for j in range(N-1):
         name="alpha_"+str((j+1)/N)
         alpha[name]=(j+1)/N
         ids.append(name)
         s_1j=[X[0]]
         s_2j=[X[0]]
         for i in range(1,len(X)):
             s1t=alpha[name]*X[i]+(1-alpha[name])*s_1j[(i-1)]
             s2t=alpha[name]*s1t+(1-alpha[name])*s_2j[(i-1)]
             s_1j.append(s1t)
             s_2j.append(s2t)
         s_1[name]=s_1j
         s_2[name]=s_2j
     
     for name in ids:
         a[name]=2*s_1[name][len(X)-k]-s_2[name][len(X)-k]
         b[name]=alpha[name]/(1-alpha[name])*(s_1[name][len(X)-k]-s_2[name][len(X)-k])
         ej=0
         F[name]=[]
         for i in range(k):
             l=len(X)-k+i
             next_f=a[name]+(i+1)*b[name]
             F[name].append(next_f)
             ej=ej+(X[l]-next_f)**2
         E[name]=math.sqrt(ej)
     Ew={}

     for i in range(N-1):
         if i<=(w/2):
             s=(w/2)+1+i
             em=0
             for q in range(int(s)):
                 em=em+E[ids[q]]
             Ew[ids[i]]=em/s
         elif (N-i-1)<=(w/2):
             s=(w/2)+1+(N-i-1)
             em=0
             for q in range(int(s)):
                 em=em+E[ids[N-q-2]]
             Ew[ids[i]]=em/s
         else:
             em=0
             for q in range(int(w+1)):
                 em=em+E[ids[i-2+q]]
             Ew[ids[i]]=em/(w+1)
     
     plt.bar(E.keys(), E.values(), color='b')
     plt.show()
     plt.bar(Ew.keys(), Ew.values(), color='g')
     #plt.show()
     
     # STEP 2: Generate a forecas with the best alpha
     min_e=100000
     for i in ids:
         if Ew[i]<min_e:
             min_e=Ew[i]
             selected=i
     
     f=[]
     x_f=[]
     a=2*s_1[selected][len(X)-1]-s_2[selected][len(X)-1]
     b=alpha[selected]/(1-alpha[selected])*(s_1[selected][len(X)-1]-s_2[selected][len(X)-1])
     for i in range(k):
         x_f.append(i+len(X)-1)
         f.append(a+(i)*b)
         
     
     x_original=range(len(X)) 
     x_validation=range((len(X)-k),(len(X)))
     
     plt.figure()
     plt.xlabel('Epoch')
     plt.ylabel('Error')
     plt.title(selected)
     plt.plot(x_f, f,label='Forecast')
     plt.plot(x_original, X,label = 'Model Error')
     plt.plot(x_validation,F[selected],label = 'Validation Forecast')
     plt.legend()
     #plt.show()
     
     return Ew[selected],f,alpha[selected]
         
 
    
#body={'device':69823,'names':["ControlRegCompAC.VarPowerKwMSK",'UnitStatus'],'times':[["2020-01-24T06:00:00.000Z","2020-01-24T16:00:00.000Z"]]}
#r = requests.post('http://db_manager:5001/collect-data',json = body)
#data=r.json()
#db=pd.DataFrame(data)
#filtered = db.loc[db['UnitStatus'] == 9.0]
mean=0.5
std=0.001
s=1000
noise=[]
for i in range(s):
    noise.append(np.random.normal(0.00001*mean*i*i/s,std))

result=forecast_Brown(X=noise,k=500)