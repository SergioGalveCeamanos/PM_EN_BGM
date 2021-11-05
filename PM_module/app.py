# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:49:21 2020

@author: sega01

https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from classes.pm_manager import upload_results,get_analysis,set_new_model,update_model,get_available_models,get_forecast, get_probability,generate_report,load_unit_data,notification_trigger
import pandas as pd
import requests
import time
import datetime 
import pickle
import traceback
from streamlit import caching
import multiprocessing 
#warnings.filterwarnings("ignore", category=DeprecationWarning) 

#from config import Config
#from flask import Flask, render_template, url_for, jsonify, request

# there should be a file containing the table will all the registered tasks and it's status
# table will have the fields: device, time_start, time_stop,date , status

#app = Flask(__name__)
#app.config.from_object(Config)
#NOT CONSIDERING NEW MODEL TRAINING YET


def build_model(data):
    print(' [D] Extracting data from file saved from Regional Manager')
    mso_path = data['mso_path']
    host = data['host']
    machine = data['machine']
    matrix = data['matrix']
    sensors_in_tables = data['sensors_in_tables']
    faults = data['faults']
    sensors = data['sensors']
    sensor_eqs = data['sensor_eqs']
    #preferent = data['preferent']
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
    traductor=data['traductor']
    filter_stab=True
    if data['filter_stab']==0:
        print('   --> OPTION: No Filter Stability')
        filter_stab=False
    #spec_list=data['spec_list']
    if 'mso_set' in data:
        mso_set=data['mso_set']
    else:
        mso_set=[]
    if 'preferent' in data:
        preferent=data['preferent']
    else:
        preferent=[]
    print(data)
    response = set_new_model(mso_path,host,machine,matrix,sensors_in_tables,faults,sensors,sensor_eqs,time_bands,filt_val,filt_param,filt_delay_cap,main_ca,max_ca_jump,cont_cond,preferent=preferent,version=version,retrain=True,aggSeconds=aggS,sam=samples,mso_set=mso_set,filter_stab=filter_stab,traductor=traductor)
    return response
# PRIORITY CRITERIA: Forecast and Probabilities ahead of Analysis (TO BE IMPLEMENTED)
def get_task(file_tasks):
    # no indication of who is accesing the file ... to be improved
    try:
        table=pd.read_csv(file_tasks,index_col=0)
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
            table.to_csv(file_tasks)
            return task
    except:
        print('Error loading the table ...')
        return 'No task available'

def cycle(task):
    print(task)
    file_tasks='/models/tasks.csv'
    n_model='/models/new_model.pkl'
    #print('--> Start of cycle')
    r=True
    options=['Zonotope']
    extra_names=['']
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
                        probabilities,mso_set= get_probability(task['device'],task['time_stop'],start_time=task['time_start'],version=task['version'])
                        upload_results(probabilities,'probabilities')
                        report=generate_report(to_write,probabilities,mso_set,size_mavg=20,version=v)
                        print(report)
                        upload_results(report,'report')
                        
                        notification_rep=notification_trigger(task['device'],task['time_stop'],version=task['version'],option=[],length=24,ma=[5,10,20])
                        if notification_rep!='All clear':
                            upload_results(notification_rep,'notification')
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
            elif task['type']=='notification':
                notification_rep=notification_trigger(task['device'],task['time_stop'],version=task['version'],option=[],length=24,ma=[5,10,20])
                if notification_rep!='All clear':
                    #upload_results(notification_rep,'notification')
                    print(notification_rep)
            elif task['type']=='update_model':
                print('Task taken to update the unit '+str(task['device']))
                v=task['version']
                update_model(task['device'],task['time_start'],task['time_stop'],version=task['version'])
                print('The retrain is completed for unit '+str(task['device']))  
                    
            elif task['type']=='build_model':
                n_model='/models/'+str(task['device'])+task['version']+'.pkl'
                filehandler = open(n_model, 'rb') 
                data = pickle.load(filehandler)
                filehandler.close()
                print('Loaded data to build new model from fiel: '+n_model)
                response=build_model(data)
                """for t in response['times_training']:
                    to_write, do_prob=get_analysis(data['device'],t[0],t[1],version=data['version'],option=options[i],extra_name=extra_names[i])
                    if len(to_write)>1:
                        upload_results(to_write,'analysis')
                        probabilities,mso_set= get_probability(data['device'],t[1],start_time=t[0],version=data['version'])
                        upload_results(probabilities,'probabilities')
                        report=generate_report(to_write,probabilities,mso_set,size_mavg=20,version=data['version'])
                        print(report)
                        upload_results(report,'report')
                
                caching.clear_cache()"""
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
            elif task['type']=='load_data_summary':
                load_unit_data(task['device'],version=task['version'])
            t_b=datetime.datetime.now()
            dif=t_b-t_a
            print(' [I] TOTAL TIME OF TASK |'+task['type']+'| is:  '+str(dif))
        except:
            r=False
            print('   [E] Failed completion of asigned task: ')
            print(task)
            traceback.print_exc()
        if r:
            text='Completed'
        else:
            text='Error uploading data'
        try:
            table=pd.read_csv(file_tasks,index_col=0)
            table.loc[(table['device']==task['device'])&(table['date']==task['date']), 'status']= text 
            table.to_csv(file_tasks)
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
    file_tasks='/models/tasks.csv'
    paralels=1  
    while True:
         room=True
         inem=False
         i=-1
         jobs = []
         print(' [S] Launch of PM processing')
         while room and i<paralels:
             i=i+1
             task=get_task(file_tasks)
             if task!='No task available':
                 p = multiprocessing.Process(target=cycle, args=([task]))
                 p.start()
                 jobs.append(p)
                 inem=True
                 time.sleep(2)
             else:
                 time.sleep(20)
                 room=False
         if inem:
            for proc in jobs:
                proc.join()


