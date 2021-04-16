# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:49:21 2020

@author: sega01

https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
"""

from flask import Flask, render_template, url_for, jsonify, request
import os, requests, uuid, json, pickle
from config import Config
from classes.elastic_manager_class import elastic_manager
import traceback

host='52.169.220.43:9200'
machine=00000

# start the object
app = Flask(__name__)
app.config.from_object(Config)
em=elastic_manager(host,machine)
em.connect()

@app.route('/collect-data', methods=['POST'])
def collect_data():
    print("INFO from collect-data: request.")
    #try:
    print("TEST with request.get_json()")
    print(request)
    data = request.get_json()
    
    #except:
        #print("ERROR DB_MANAGER: data could not be retrieved from request received")

    print(data)
    device = data['device']
    names = data['names']
    times = data['times']
    em.agg_seconds=data['aggSeconds']
    em.device=device
    d=em.get_set(names,times)
    return jsonify(d) #jsonify(response)

@app.route('/collect-model-error', methods=['POST'])
def collect_model_error():
    print("INFO from collect-data: request.")
    #try:
    print("TEST with request.get_json()")
    print(request)
    data = request.get_json()
    print(data)
    device = data['device']
    times = data['times']
    version = data['trained_version']
    probs=False
    if 'group_prob' in data:
        if data['group_prob']==1:
            probs=True
    em.device=device
    d=em.get_analytics(times[0],times[1],version,probs)
    return jsonify(d) #jsonify(response)

@app.route('/collect-configuration', methods=['POST'])
def collect_configuration():
    print("INFO from collect-data: request.")
    #try:
    print("TEST with request.get_json()")
    print(request)
    data = request.get_json()
    print(data)
    device = data['device']
    version = data['trained_version']
    times = ["1945-08-06T08:15:00.000Z",data['times']]
    em.device=device
    # we should get the config closest to the given time
    d=em.get_configuration(times,version)
    return jsonify(d) #jsonify(response)

@app.route('/collect-last-priors', methods=['POST'])
def collect_last_priors():
    print("INFO from collect-last-prior: request.")
    #try:
    print(request)
    data = request.get_json()
    print(data)
    device = data['device']
    version = data['trained_version']
    times = ["1945-08-06T08:15:00.000Z",data['times']]
    em.device=device
    # we should get the config closest to the given time
    d=em.get_last_priors(times,data['faults'],version)
    return jsonify(d) #jsonify(response)

@app.route('/upload-analysis', methods=['POST'])
def upload_analysis():
    #print("INFO from collect-data: request.")
    data = request.get_json()
    '''device= data['device']
    t1= data['time_start']
    t2= data['time_stop']
    analysis= data['analysis']'''
    #print('   Received data:')
    #print('    - Analysis:')
    #print(analysis)
    #print('    - Times (1,2): '+str(t1)+', '+str(t2))
    #em.device=device
    #try:
    try:
        response=em.create_new_doc(data)
        worked='True'
    except:
        print('[¡] Error loading result of analysis to DB')  
        traceback.print_exc()
        worked='False'
    #except:
      #print("An exception occurred ...")
      #worked='False'
    return worked

@app.route('/upload-forecast', methods=['POST'])
def upload_forecast():
    data = request.get_json()
    try:
        response=em.load_forecast(data)
        worked='True'
    except:
        print('[¡] Error loading result of forecast to DB')  
        traceback.print_exc()
        worked='False'
    #except:
      #print("An exception occurred ...")
      #worked='False'
    return worked

@app.route('/upload-configuration', methods=['POST'])
def upload_configuration():
    data = request.get_json()
    try:
        response=em.load_configuration(data)
        worked='True'
    except:
        print('[¡] Error loading configuration to DB')  
        traceback.print_exc()
        worked='False'
    #except:
      #print("An exception occurred ...")
      #worked='False'
    return worked

@app.route('/upload-probabilities', methods=['POST'])
def upload_probabilities():
    data = request.get_json()
    try:
        response=em.load_probability(data)
        worked='True'
    except:
        print('[¡] Error loading result of bayesian probability to DB')  
        traceback.print_exc()
        worked='False'
    #except:
      #print("An exception occurred ...")
      #worked='False'
    return worked

@app.route('/upload-report', methods=['POST'])
def upload_report():
    data = request.get_json()
    try:
        response=em.load_report(data)
        worked='True'
    except:
        print('[¡] Error loading report to the DB')  
        traceback.print_exc()
        worked='False'
    #except:
      #print("An exception occurred ...")
      #worked='False'
    return worked

# launch service in IP 127.0.0.1
if __name__ == '__main__':
    app.run(port=5001,threaded=True) #host='127.0.0.1'
    
