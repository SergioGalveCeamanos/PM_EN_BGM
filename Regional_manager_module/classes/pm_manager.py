# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:46:21 2020

@author: sega01

To evaluate the given data and return the apropiate response
https://docs.microsoft.com/es-es/azure/cognitive-services/translator/tutorial-build-flask-app-translation-synthesis
"""
# the root for developed classes must be the same as in the saved files
import os, requests, uuid, json, pickle
from os import path
import pandas as pd

def file_location(device,root_path='/models/model_'):
    filename = root_path+str(device)+'/FM_'+str(device)+'.pkl'
    model_dir =root_path+str(device)+'/'
    return filename, model_dir

def fix_dict(d):
    new_d={}
    for n in d:
        new_d[int(n)]=d[n]
        
    return new_d
        
def get_available_models():
    models=os.listdir('/models')
    #print(models)
    devices={'available_models':[]}
    for m in models:
        if m[0:6]=='model_':
            devices['available_models'].append(m[6:])
    return devices
