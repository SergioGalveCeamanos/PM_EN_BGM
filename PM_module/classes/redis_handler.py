# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:36:28 2020

@author: sega01
"""

from zipfile import ZipFile
import os
from os.path import basename
import base64
import redis
import traceback

dirName='/home/sergio/PM_docker-compose/vol_230920_StudentVM'
zip_name='/home/sergio/PM_docker-compose/sampleDir.zip'
destination='/home/sergio/PM_docker-compose/test_zip'
read_zip='/home/sergio/PM_docker-compose/test_zip/output_file.zip'


def set_redis(machine,sn_device):
    name=machine+':'+sn_device
    return name

def save_model(dirName,zip_name,machine,sn_device,host='localhost', port=6379, version='New'):
    try:
        with ZipFile(zip_name, 'w') as zipObj:
           # Iterate over all the files in directory
           for folderName, subfolders, filenames in os.walk(dirName):
               for filename in filenames:
                   #create complete filepath of file in directory
                   filePath = os.path.join(folderName, filename)
                   # Add file to zip
                   zipObj.write(filePath, basename(filePath))
                                  
        with open(zip_name, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
                
        r = redis.Redis(host=host, port=port, db=0)
        if version=='New':
            keys=redisClient.hkeys(set_redis(machine,sn_device))
            for i in range(0, len(keys)): 
                keys[i] = int(keys[i])
            v=str(max(keys)+1)
        else:
            v=str(version)
        r.hset(set_redis(machine,sn_device),v, encoded_string)
    except:
        traceback.print_exc()
        print(' [!] Error saving model into Redis DB')


def load_model(read_zip,destination,machine,sn_device,host='localhost', port=6379, version='Latest'):
    r = redis.Redis(host=host, port=port, db=0)
    if version=='Latest':
        keys=redisClient.hkeys(set_redis(machine,sn_device))
        for i in range(0, len(keys)): 
            keys[i] = int(keys[i])
        v=str(max(keys))
    else:
        v=str(version)
    test_response=redisClient.hget(set_redis(machine,sn_device), v)
    with open(read_zip, 'wb') as result:
        result.write(base64.b64decode(test_response))
    with ZipFile(read_zip, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
       zipObj.extractall(destination)