# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:49:21 2020

@author: sega01
"""

#from config import Config
#from flask import Flask, render_template, url_for, jsonify, request

# there should be a file containing the table will all the registered tasks and it's status
# table will have the fields: device, time_start, time_stop,date , status


import requests
import datetime 
import time


# this must be triggered every X time, parametrized ... by default it is understood that is from now 
# the length sets how much time to go back from time
def launch_all(ip='http://regional_manager:5000/',length=datetime.timedelta(hours=1),time="now"):
    try:
        # headers=headers
        address=ip+'get-devices'
        
        r = requests.get(address) # 'http://db_manager:5001/collect-data'
        data=r.json()
        devices=data['available_models']
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        raise SystemExit(e)
    
    # date format objective: "2019-12-12T05:45:00.000Z"
    if time=="now":
        now = datetime.datetime.now()#-datetime.timedelta(hours=24)
        end = now.isoformat() # '2020-05-11T09:46:24.878629' ... we must adapt it
    else:
        now = time
        end = time.isoformat()
    start = (now-length).isoformat()
    end=end[:(len(end)-3)]+'Z'
    start=start[:(len(start)-3)]+'Z'
    print(start)
    print(end)
    # device per device ... 2 step requests
    data={}
    print('Available Devices:')
    print(devices)
    for d in devices:
        address=ip+'get-analysis'
        body={"device": int(d),"time_start":start,"time_stop":end}
        r = requests.post(address, json=body, timeout = 3600)
        # It must contain the prediction and also the boundaries
        
    return r,body


# launch service in IP 0.0.0.0

if __name__ == '__main__':
    #app.run(port=5002,threaded=True)
    wait=1800
    while True:
        data=launch_all(length=datetime.timedelta(seconds=wait),time="now")
        #print(data)
        time.sleep(wait)

