# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:49:21 2020

@author: sega01

https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
"""

from classes.pm_manager import get_analysis,set_new_model,get_available_models,get_forecast, get_probability,generate_report
import pandas as pd
import requests
import time
import datetime 
import pickle
import warnings
import traceback
from streamlit import caching
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#from config import Config
#from flask import Flask, render_template, url_for, jsonify, request

# there should be a file containing the table will all the registered tasks and it's status
# table will have the fields: device, time_start, time_stop,date , status

#app = Flask(__name__)
#app.config.from_object(Config)
#NOT CONSIDERING NEW MODEL TRAINING YET


def build_model(data):
    mso_path = data['mso_path']
    host = data['host']
    machine = data['machine']
    matrix = data['matrix']
    sensors_in_tables = data['sensors_in_tables']
    faults = data['faults']
    sensors = data['sensors']
    sensor_eqs = data['sensor_eqs']
    preferent = data['preferent']
    time_bands = data['time_bands']
    aggS= data['aggSeconds']
    samples=data['sample_size']
    filt_val=data['filt_value']
    filt_param=data['filt_parameter']
    filt_delay_cap=data['filt_delay_cap']
    main_ca=data['main_control_actuator_signs']
    max_ca_jump=data['transition_trigger_low_bounds']
    max_ca_jump=data['transition_trigger_low_bounds']
    cont_cond=data['contour_conditions']
    version=data['version']
    spec_list=data['spec_list']
    if 'mso_set' in data:
        mso_set=data['mso_set']
    else:
        mso_set=[]
    print(data)
    response = set_new_model(mso_path,host,machine,matrix,sensors_in_tables,faults,sensors,sensor_eqs,preferent,time_bands,filt_val,filt_param,filt_delay_cap,main_ca,max_ca_jump,cont_cond,version=version,retrain=True,aggSeconds=aggS,sam=samples,mso_set=mso_set)
    
# PRIORITY CRITERIA: Forecast and Probabilities ahead of Analysis (TO BE IMPLEMENTED)
def get_task(file):
    # no indication of who is accesing the file ... to be improved
    try:
        table=pd.read_csv(file,index_col=0)
        todo=table.loc[table['status']=='ToDo']
        if todo.shape[0]==0:
            return 'No task available'
        else:
            task={}
            task['device']=todo['device'].iloc[0]
            task['time_start']=todo['time_start'].iloc[0]
            task['time_stop']=todo['time_stop'].iloc[0]
            task['date']=todo['date'].iloc[0]
            task['status']=todo['status'].iloc[0]
            task['type']=todo['type'].iloc[0]
            task['version']=todo['version'].iloc[0]
            table.loc[(table['device']==task['device'])&(table['date']==task['date']), 'status']= 'Processing'
            table.to_csv(file)
            return task
    except:
        print('Error loading the table ...')
        return 'No task available'

# Funtion to load all the documents --> combined with the bulk function            
def upload_results(documents,task_type):
    http_dic={'analysis':'http://db_manager:5001/upload-analysis','configuration':'http://db_manager:5001/upload-configuration','forecasts':'http://db_manager:5001/upload-forecast','probabilities':'http://db_manager:5001/upload-probabilities','report':'http://db_manager:5001/upload-report'}
    error_dic={'analysis':'[¡] Error uploading model analysis in sample #','configuration':' [!] Error uploading Configuration','forecasts':' [!] Error uploading Forecast','probabilities':' [!] Error uploading Probabilities','report':' [!] Error uploading report'}
    
    try:
        r = requests.post(http_dic[task_type],json = documents)
    except:
        print(error_dic[task_type])
        traceback.print_exc()

def cycle():
    file='/models/tasks.csv'
    n_model='/models/new_model.pkl'
    print('--> Start of cycle')
    r=True
    task=get_task(file)
    print(task)
    options=['Zonotope','KDE_1D']
    extra_names=['','_KDE_1D']
    if task!='No task available':
        try:
            t_a=datetime.datetime.now()
            if task['type']=='analysis':
                print('Task taken to analyze the unit '+str(task['device']))
                for i in range(len(options)):
                    v=task['version']+extra_names[i]
                    to_write, do_prob=get_analysis(task['device'],task['time_start'],task['time_stop'],version=task['version'],option=options[i],extra_name=extra_names[i])
                    print('The analysis is completed for unit '+str(task['device']))
                    
                    if len(to_write)>1:
                        upload_results(to_write,'analysis')
                        # Now get forecasts
                        """new_conf, do_prob_fore, forecast_docs=get_forecast(task['device'],task['time_stop'],version=v)
                        print(new_conf)
                        upload_results(new_conf,'configuration')
                        upload_results(forecast_docs,'forecasts')
                        if do_prob or do_prob_fore:
                            #print(' --> Here you would do the probability thing')
                            #probabilities=get_probability(task['device'],task['time_stop'])
                            #upload_results(probabilities,'probabilities')
                            probabilities,mso_set= get_probability(task['device'],task['time_stop'],start_time=task['time_start'],version=v)
                            upload_results(probabilities,'probabilities')
                            report=generate_report(to_write,probabilities,forecast_docs,mso_set,size_mavg=20,version=v)
                            print(report)
                            upload_results(report,'report')"""
        
                    else:
                        r=False
                        print('[¡] No available data: the time band must have no recorded samples with the actuators working') 
                
            elif task['type']=='build_model':
                n_model='/models/'+str(task['device'])+task['version']+'.pkl'
                filehandler = open(n_model, 'rb') 
                data = pickle.load(filehandler)
                filehandler.close()
                print('Loaded data to build new model from fiel: '+n_model)
                build_model(data)
                caching.clear_cache()
                print('Cleared Cache and Model Trained')
                r=True
            
            elif task['type']=='redo_forecast':
                new_conf,fault, forecast_docs=get_forecast(task['device'],task['time_stop'],version=task['version'])
                upload_results([new_conf],'configuration')
                upload_results(forecast_docs,'forecasts')
                if fault:
                   #probabilities= get_probability(task['device'],task['time_stop'])
                   #upload_results(probabilities,'probabilities')
                   probabilities= get_probability(task['device'],task['time_stop'],start_time=task['time_start'],version=task['version'])
                   upload_results(probabilities,'probabilities')
            elif task['type']=='redo_probabilities':
                #probabilities= get_probability(task['device'],task['time_stop'])
                #upload_results(probabilities,'probabilities')
                probabilities= get_probability(task['device'],task['time_stop'],start_time=task['time_start'],version=task['version'])
                upload_results(probabilities,'probabilities')
            t_b=datetime.datetime.now()
            dif=t_b-t_a
            print(' [I] TOTAL TIME OF TASK |'+task['type']+'| is:  '+str(dif))
        except:
            r=False
            traceback.print_exc()
        if r:
            text='Completed'
        else:
            text='Error uploading data'
        try:
            table=pd.read_csv(file,index_col=0)
            table.loc[(table['device']==task['device'])&(table['date']==task['date']), 'status']= text 
            table.to_csv(file)
        except:
            print('Error updating status ...')
                

# For the test phase the data will be given completly for the sligtly modified version of FM_ES
'''@app.route('/new-model', methods=['POST'])
def new_model():
    data = request.get_json()
    print(data)
    mso_path = data['mso_path']
    host = data['host']
    machine = data['machine']
    matrix = data['matrix']
    sensors_in_tables = data['sensors_in_tables']
    faults = data['faults']
    mso_set = data['mso_set']
    sensors = data['sensors']
    sensor_eqs = data['sensor_eqs']
    preferent = data['preferent']
    time_bands = data['time_bands']
    response = set_new_model(mso_path,host,machine,matrix,sensors_in_tables,faults,mso_set,sensors,sensor_eqs,preferent,time_bands,retrain=True)
    return jsonify(response) #jsonify(response)'''



# launch service in IP 0.0.0.0
if __name__ == '__main__':
    # LOAD MODELS IN DICTIONARY
    #app.run(port=5002,threaded=True)
    while True:
        time.sleep(60)
        cycle()

