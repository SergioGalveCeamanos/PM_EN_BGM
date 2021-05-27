# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:49:21 2020

@author: sega01

https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
"""

from flask import Flask, render_template, url_for, jsonify, request
from config import Config
from classes.pm_manager import get_available_models
import datetime 
import pandas as pd
import pickle
import traceback

# start the object
app = Flask(__name__)
app.config.from_object(Config)

file='/models/tasks.csv'
n_model='/models/new_model.pkl'
# define some functions callable from http
"""@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Miguel'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template('index.html', title='Home', user=user, posts=posts)"""

@app.route('/get-analysis', methods=['POST'])
def load_task():
    data = request.get_json()
    device = int(data['device'])
    t1 = data['time_start']
    t2 = data['time_stop']
    v = data['version']
    try:
        table=pd.read_csv(file,index_col=0)
        new_row={'device': device,'time_start': t1,'time_stop': t2,'date': str(datetime.datetime.now()),'status': 'ToDo','type':'analysis','version':v}
        print('The new task received is: ')
        print(new_row)
        table=table.append(new_row,ignore_index=True)
        table=table.astype({'device': 'int32'},errors='ignore')
        table.to_csv(file)
        print('The resulting new table is:')
        print(table)
        worked='True'
    except:
        print("Error Loading the task to the CSV -> Analysis not registered")
        worked='False'
        
    return worked #jsonify(response)

@app.route('/re-do-forecast', methods=['POST'])
def redo_forecast():
    data = request.get_json()
    device = int(data['device'])
    t1 = data['time_start']
    t2 = data['time_stop']
    try:
        table=pd.read_csv(file,index_col=0)
        new_row={'device': device,'time_start': t1,'time_stop': t2,'date': str(datetime.datetime.now()),'status': 'ToDo','type':'redo_forecast'}
        print('The new task received is: ')
        print(new_row)
        table=table.append(new_row,ignore_index=True)
        table=table.astype({'device': 'int32'},errors='ignore')
        table.to_csv(file)
        print('The resulting new table is:')
        print(table)
        worked='True'
    except:
        print("Error Loading the task to the CSV -> Redo Forecast not registered")
        worked='False'

    return worked #jsonify(response)

@app.route('/re-do-probabilities', methods=['POST'])
def redo_probabilities():
    data = request.get_json()
    device = int(data['device'])
    t1 = data['time_start']
    t2 = data['time_stop']
    try:
        table=pd.read_csv(file,index_col=0)
        new_row={'device': device,'time_start': t1,'time_stop': t2,'date': str(datetime.datetime.now()),'status': 'ToDo','type':'redo_probabilities'}
        print('The new task received is: ')
        print(new_row)
        table=table.append(new_row,ignore_index=True)
        table=table.astype({'device': 'int32'},errors='ignore')
        table.to_csv(file)
        print('The resulting new table is:')
        print(table)
        worked='True'
    except:
        print("Error Loading the task to the CSV -> Redo Forecast not registered")
        worked='False'

    return worked #jsonify(response)

# To retrieve the available models to loop into
@app.route('/get-devices', methods=['GET'])
def get_devices():
    response = get_available_models()
    return jsonify(response) #jsonify(response)

# For the test phase the data will be given completly for the sligtly modified version of FM_ES
@app.route('/new-model', methods=['POST'])
def new_model():

    data = request.get_json()
    print('Received Data: ')
    print(data)

    #print(data)
    machine = int(data['machine'])
    #response = set_new_model(mso_path,host,machine,matrix,sensors_in_tables,faults,mso_set,sensors,sensor_eqs,preferent,time_bands,retrain=True)
    #return jsonify(response) #jsonify(response)
    #try:
    n_model='/models/'+data['machine']+data['version']+'.pkl'
    
    filehandler = open(n_model, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()
    try:
        table=pd.read_csv(file,index_col=0)
        new_row={'device': machine,'time_start': 'N/A','time_stop': 'N/A','date': str(datetime.datetime.now()),'status': 'ToDo','type':'build_model','version':data['version']}
        print('The new task received is: ')
        print(new_row)
        table=table.append(new_row,ignore_index=True)
        table=table.astype({'device': 'int32'},errors='ignore')
        table.to_csv(file)
        worked='True'
    except:
        print("Error Loading the task to the CSV -> New Training not registered")
        worked='False'
        traceback.print_exc()
    if worked=='False':
        new_row={'device': machine,'time_start': 'N/A','time_stop': 'N/A','date': str(datetime.datetime.now()),'status': 'ToDo','type':'build_model','version':data['version']}
        new={'device': [],'time_start': [],'time_stop': [],'date': [],'status': [],'type':[]}
        db=pd.DataFrame(new)
        db=db.append(new_row,ignore_index=True)
        db.to_csv(file)
        print('New Table Created --> New Model Task Included')
    #except:
        #print("Error Loading the task to the CSV -> New model training not registered")
        #worked='False'
    return worked

@app.route('/load-unit-data', methods=['POST'])
def load_unit_data():
    data = request.get_json()
    data_file='/models/model_data_'
    device=str(data['machine'])
    version= data['version']
    data_file=data_file+device
    do_load=True
    try:
        with open(data_file, 'rb') as handle:
            d = pickle.load(handle)
        if version in d:
            do_load=False
    # Do something with the file
    except:
        print('File not created for model: '+device)
        d = {'message':'Not loaded the requested version ... Added to the tasklist, check again later.'}
    if do_load:
        try:
            table=pd.read_csv(file,index_col=0)
            new_row={'device': int(data['machine']),'time_start': 'N/A','time_stop': 'N/A','date': str(datetime.datetime.now()),'status': 'ToDo','type':'load_data_summary','version':data['version']}
            print('The new task received is: ')
            print(new_row)
            table=table.append(new_row,ignore_index=True)
            table=table.astype({'device': 'int32'},errors='ignore')
            table.to_csv(file)
            worked='True'
        except:
            print("Error Loading the task to the CSV -> New Training not registered")
            worked='False'
    return d
# launch service in IP 0.0.0.0
if __name__ == '__main__':
    app.run(host='0.0.0.0',threaded=True) #debug=True,

