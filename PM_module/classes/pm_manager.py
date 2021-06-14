# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:46:21 2020

@author: sega01

To evaluate the given data and return the apropiate response
https://docs.microsoft.com/es-es/azure/cognitive-services/translator/tutorial-build-flask-app-translation-synthesis
"""
# the root for developed classes must be the same as in the saved files
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import traceback
from streamlit import caching
import os, requests, uuid, json, pickle
from os import path
import numpy as np
import pandas as pd
from .fault_detector_class_ES import fault_detector
from .MSO_selector_GA import find_set_v2
from .test_cross_var_exam import launch_analysis
import datetime 
import traceback
import copy
import multiprocessing

#def get_model(filename,id_device):
    #if id_device in model_dictionary:
        #return model_dictionary[id_device]
        
# Funtions to interact with the fault manager
def load_model(filename,model_dir):
    fault_manager = fault_detector(filename=[],mso_txt=[],host=[],machine=[],matrix=[],sensors=[],faults=[],sensors_lookup=[],sensor_eqs=[],filt_value=9,filt_parameter=[],filt_delay_cap=[],main_ca=[],max_ca_jump=[],cont_cond=[])
    fault_manager.Load(model_dir,filename)
    return fault_manager

def file_location(device,version="",root_path='/models/model_'):
    device=int(device)
    filename = root_path+str(device)+version+'/FM_'+str(device)+'.pkl'
    model_dir = root_path+str(device)+version+'/'
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

def load_unit_data(device,version="",data_file='/models/model_data_'):
    device=str(device)
    file, folder = file_location(device,version)
    fm=load_model(file, folder)
    data_file=data_file+device
    try:
        with open(data_file, 'rb') as handle:
            d = pickle.load(handle)
            d[version]={}
    # Do something with the file
    except:
        print('File not created for model: '+device)
        d = {version:{}}
    d[version]['residuals']=fm.mso_set
    outs={}
    variables={}
    mean={}
    std={}
    for m in fm.mso_set:
        outs[str(m)]=fm.models[m].objective
        variables[str(m)]=fm.models[m].source
        mean[str(m)]=fm.models[m].train_stats.loc[fm.models[m].objective,'mean']
        std[str(m)]=fm.models[m].train_stats.loc[fm.models[m].objective,'std']
    d[version]['outs']=outs
    d[version]['variables']=variables
    d[version]['mean']=mean
    d[version]['std']=std
    print(d[version])
    with open(data_file, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_analysis(device,time_start,time_stop,version="",aggSeconds=1,option=[],extra_name=[]):

    file, folder = file_location(device,version)
    fm=load_model(file, folder)
    names,times_b=fm.get_data_names(option='CarolusRex',times=[[time_start,time_stop]])
    names.append(fm.target_var)
    ######## NOT LIKE THIS --> MUST BE CHANGED #############
    if int(device)==71471 or int(device)==74124:
        aggSeconds=1
    ########################################################
    body={'device':str(device),'names':names,'times':times_b,'aggSeconds':aggSeconds}
    #headers = {'Content-Type': 'application/json'}
    try:
        # headers=headers
        print(' HTTP message Body: ')
        print(body)
        r = requests.post('http://db_manager:5001/collect-data',json = body) # 'http://db_manager:5001/collect-data'
        data=r.json()
    except:  # This is the correct syntax
        print(' [!] Error gathering Telemetry')

    try:
        partial_message=' [E] The filtering process failed after: '
        code_error=0
        activations=[]
        msos=[]
        confidences=[]
        group_probabilities=[]
        residuals={}
        response={}
        db=pd.DataFrame(data)
        #print(db)
        db_sort=copy.deepcopy(db.sort_values('timestamp'))
        partial_message=partial_message+'sorting values, '
        code_error=code_error+1
        # ANOTHER PATCH ---> a Parametric Filter is needed
        if int(device)==71471 or int(device)==74124:
            db_sort.loc[:,'UnitStatus']=db_sort.loc[:,'UnitStatus']*10
            db.loc[:,'UnitStatus']=db.loc[:,'UnitStatus']*10
            partial_message=partial_message+'adapting UnitStatus, '
            code_error=code_error+1
        #db_sort = db_sort.loc[db_sort['UnitStatus'] == 9.0]
        filt_db=fm.filter_samples(db_sort)
        partial_message=partial_message+'filtering samples, '
        code_error=code_error+1
        target=filt_db.pop(fm.target_var)
        if fm.filter_stab:
            filt_db=fm.filter_stability(filt_db,target)
            partial_message=partial_message+'stability filtering '
            code_error=code_error+1
        #print(filt_db)
        do_prob=False
        result=[]
    except:
        print(partial_message)
        traceback.print_exc()
        if code_error==0:
            print(data)
        elif code_error==1:
            print(db_sort)
        elif code_error==2:
            print(db_sort.loc[:,'UnitStatus'])
        elif code_error==3:
            print(db_sort.loc[:,'UnitStatus'])
            print(filt_db)
        elif code_error==4:
            print(filt_db)
    if filt_db.shape[0]>5:

    # CHECK MISSING: is the next horizon among these two windows ... should we reevaluate (rewrite some samples in DB) everytime otherwise?
        times=filt_db['timestamp']
        try:
            t_a=datetime.datetime.now()
            if option!=[]:
                return_dic,forget=fm.evaluate_data(manual_data=filt_db,option=option)
            else:
                return_dic,forget=fm.evaluate_data(manual_data=filt_db)
            t_b=datetime.datetime.now()
            dif=t_b-t_a
            print('  [T] TOTAL evaluation computing time ---> '+str(dif))
            with_probs=False
            if 'group_prob' in return_dic[fm.get_dic_entry(fm.mso_set[0])]:
                with_probs=True
            filter_activations={}
            #print(' [I]  RESPONSE FROM EVALUATE_DATA ')
            #print(return_dic)
            for mso in fm.mso_set:
                entry=fm.get_dic_entry(mso)
                residuals[entry]=return_dic[entry]['error']
                activations.append(return_dic[entry]['phi'])
                filter_activations[entry]=return_dic[entry]['phi']
                confidences.append(return_dic[entry]['alpha'])
                if with_probs:
                    group_probabilities.append(return_dic[entry]['group_prob'])
                #Ew,f,al=fm.forecast_Brown(errors,100)
                msos.append(entry)
                if with_probs:
                    response[entry]={'known':fm.models[mso].known,'error':return_dic[entry]['error'],'high_bound':return_dic[entry]['high'],'low_bound':return_dic[entry]['low'],'activations':return_dic[entry]['phi'],'confidence':return_dic[entry]['alpha'],'group_prob':return_dic[entry]['group_prob']} # ,'residual_forecast':f,'forecast_alpha':al,'forecast_validation_error':Ew
                else:
                    response[entry]={'known':fm.models[mso].known,'error':return_dic[entry]['error'],'high_bound':return_dic[entry]['high'],'low_bound':return_dic[entry]['low'],'activations':return_dic[entry]['phi'],'confidence':return_dic[entry]['alpha']} # ,'residual_forecast':f,'forecast_alpha':al,'forecast_validation_error':Ew
            #prior_evolution=fm.prior_update(activations, confidences)
            
            result=[]
            n=-1
            for t in forget:
                times=times[times!=t]
            transition_filtered=fm.filter_transitions(db.sort_values('timestamp'),filter_activations,times)
            for i in times.index:
                n=n+1
                if times.loc[i] in transition_filtered:
                    if transition_filtered[times.loc[i]]:
                        new={}
                        if extra_name!=[]:
                            new={'timestamp':times.loc[i],'device':str(device),'trained_version':version+extra_name}
                        else:
                            new={'timestamp':times.loc[i],'device':str(device),'trained_version':version}
                        errors=[]
                        low_bounds=[]
                        high_bounds=[]
                        activations=[]
                        confidence=[]
                        group_prob=[]
                        #forecasts=[]
                        #f_error=[]
                        #alphas=[]
                        try:
                            for name in msos:
                                d1=response[name]['error']
                                d2=response[name]['low_bound']
                                d3=response[name]['high_bound']
                                d4=response[name]['activations']
                                d5=response[name]['confidence']
                                errors.append(float(d1[n]))
                                low_bounds.append(float(d2[n]))
                                high_bounds.append(float(d3[n]))
                                activations.append(int(d4[n]))
                                confidence.append(float(d5[n]))
                                if with_probs:
                                    d6=response[name]['group_prob']
                                    group_prob.append(d6[n].astype(float).tolist())
                            new['models_error']=errors
                            new['low_bounds']=low_bounds
                            new['high_bounds']=high_bounds
                            new['activations']=activations 
                            new['confidence']=confidence
                            if with_probs:
                                new['group_prob']=group_prob
                            if sum(activations)>0:
                                do_prob=True
                            result.append(new)
                        except:
                            traceback.print_exc()
                            print('   [¡] Error preparing sample for the get_analysis() return')    
        except:
            traceback.print_exc()
            print('   [¡] Error loading samples for analysis --> Empty sample?')
            result=[]
    return result, do_prob #json.dumps(response)

# get forecast and the evaluation against the boundaries probability distribution 
def load_config(device,current_time,msos,version=""):
    new_conf={}
    new_conf['timestamp']=current_time
    new_conf['device']=str(device)
    new_conf['forecast_window']=[]
    new_conf['sample_size']=[]
    new_conf['alpha']=[]
    new_conf['trained_version']=version
    new_conf['forecast_activation_prob']=[]
    new_conf['forecast_activation_time']=[]
    for i in range(msos):
        new_conf['forecast_window'].append(int(5000))
        new_conf['sample_size'].append(int(15000))
        new_conf['alpha'].append(float(0.5))
        new_conf['forecast_activation_prob'].append(float(0.01))
        new_conf['forecast_activation_time'].append("INITIAL_CONFIG_DOC")
    r = requests.post('http://db_manager:5001/upload-configuration',json = new_conf)
    return new_conf

# get forecast and the evaluation against the boundaries probability distribution 
def get_forecast(device,current_time,agg_sec=5,version=""):
    if device==71471 or device==74124:
        agg_sec=1
    file, folder = file_location(device,version)
    fm=load_model(file, folder)
    # here we just want to hit the closest config to a given date 
    names=['forecast_window','sample_size','alpha']
    body={'device':str(device),'trained_version':version,'names':names,'times':current_time}
    try:
        r = requests.post('http://db_manager:5001/collect-configuration',json = body) # 'http://db_manager:5001/collect-data'
        config=r.json()
    except:  # Extra details:  requests.exceptions.RequestException as e
        traceback.print_exc()
        config=load_config(device,current_time,len(fm.mso_set),version)
        #raise SystemExit(e) 
    if config==[]:
        config=load_config(device,current_time,len(fm.mso_set),version)
    current_dt=datetime.datetime.strptime(current_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    length=datetime.timedelta(seconds=(max(config['sample_size'])))
    start = (current_dt-length).isoformat()
    #if len(start)<20:
        #start=start+'.000Z'
    #else:
    start=start[:(len(start)-3)]+'Z'
    #file, folder = file_location(device)    
    #fm=load_model(file, folder)
    names=['models_error','high_bounds','low_bounds']
    body={'device':str(device),'trained_version':version,'names':names,'times':[start,current_time]}
    try:
        r = requests.post('http://db_manager:5001/collect-model-error',json = body) # 'http://db_manager:5001/collect-data'
        # here data will be a list of dicts, in each one all the fields separated by mso
        all_data=r.json()
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        traceback.print_exc()
        raise SystemExit(e)    

    new_conf={}
    docs=[]
    fault=True
    if len(all_data)>0:
        new_conf['timestamp']=current_time
        new_conf['device']=str(device)
        new_conf['trained_version']=version
        new_conf['forecast_window']=[]
        new_conf['sample_size']=[]
        new_conf['alpha']=[]
        new_conf['forecast_activation_prob']=[]
        new_conf['forecast_activation_time']=[]
        up_cap_fw=17280 # considering a sampling rate of 5s ... this is one day ahead
        low_cap_fw=720  # ... this is one hour ahead
        up_alpha=0.9
        low_alpha=0.1
        change_rate=1.2
        fault_prob_limit=0.05
        template={}
        template['timestamp']=current_time
        template['device']=str(device)
        template['trained_version']=version
        template['mso_name']=[]
        template['mso_error_forecast']=1
        template['aggregationSeconds']=agg_sec
        template['alpha']=[]
        for i in range(len(fm.mso_set)):
            k=config['forecast_window'][i]
            min_k=(2*len(all_data[0][names[0]])/3) # the minimum feasible given the amount of samples
            if k>min_k:
                print(' [?] Not enough datapoints for the given Forecast Window: '+str(k)+' reduced to '+str(min_k))
                k=int(min_k)
            data=all_data[i]
            f_error,forecast,alpha=fm.forecast_Brown(data[names[0]],k)
            time_act, bound=fm.boundary_forecast(forecast,data[names[1]],data[names[2]])
            # New config must be obtained after the forecas
            # the value of alpha might indicate that we need a different focus --> changes in fw and sample size
            # One constrain for the moment will be to set ss=3*fw
            if alpha>up_alpha:
                fw=k/change_rate
                if fw<low_cap_fw:
                    fw=low_cap_fw
            elif alpha<low_alpha:
                fw=k*change_rate
                if fw>up_cap_fw:
                    fw=up_cap_fw
            else: 
                fw=k
            ss=fw*3
            new_conf['forecast_window'].append(int(fw))
            new_conf['sample_size'].append(int(ss))
            new_conf['alpha'].append(float(alpha))
            if bound=='no_activation':
                new_conf['forecast_activation_prob'].append(0)
            else:
                new_conf['forecast_activation_prob'].append(1)
            length=datetime.timedelta(seconds=(time_act*agg_sec))
            new_time = (current_dt+length).isoformat()
            new_time=new_time[:(len(start)-3)]+'Z'
            new_conf['forecast_activation_time'].append(new_time)
            for n in range(len(forecast)):
                length=datetime.timedelta(seconds=((n+1)*agg_sec))
                new_time = (current_dt+length).isoformat()
                new_time = new_time[:(len(start)-3)]+'Z'
                fore_doc=copy.deepcopy(template)
                fore_doc['alpha']=alpha
                fore_doc['timestamp']=new_time
                fore_doc['mso_name']='MSO_'+str(fm.mso_set[i])
                fore_doc['mso_error_forecast']=forecast[n]
                #create a list of files to be uploaded
                docs.append(fore_doc)
                    
        #r = requests.post('http://db_manager:5001/upload-forecast',json = to_write[i])
        
        # evaluate if any case shows a tenency that really goes way above the 
        #for act in new_conf['forecast_activation_prob']:
         #   if act<fault_prob_limit:
          #      fault=True
            
    return new_conf,fault, docs
    #upload forecast results?   
    #upload config --> including evaluation from forecast
    
def get_probability(device,current_time,start_time=[],version=""):
    file, folder = file_location(device,version)
    fm=load_model(file, folder)
    # here we just want to hit the closest config to a given date         
    if start_time==[]:
        names=['forecast_window','sample_size','alpha']
        body={'device':str(device),'trained_version':version,'names':names,'times':current_time}
        try:
            r = requests.post('http://db_manager:5001/collect-configuration',json = body) # 'http://db_manager:5001/collect-data'
            config=r.json()
        except:  # Extra details:  requests.exceptions.RequestException as e
            traceback.print_exc()
            config=load_config(device,current_time,len(fm.mso_set))
            #raise SystemExit(e) 
        if config==[]:
            config=load_config(device,current_time,len(fm.mso_set))
        current_dt=datetime.datetime.strptime(current_time, '%Y-%m-%dT%H:%M:%S.%fZ')
        length=datetime.timedelta(seconds=(max(config['sample_size'])))
        start = (current_dt-length).isoformat()
        start=start[:(len(start)-3)]+'Z'
    else:
        start=start_time
    #file, folder = file_location(device)    
    #fm=load_model(file, folder)
    names=['activations','confidence','timestamp','group_prob']
    body={'device':str(device),'trained_version':version,'names':names,'times':[start,current_time],'group_prob':1}
    try:
        r = requests.post('http://db_manager:5001/collect-model-error',json = body) # 'http://db_manager:5001/collect-data'
        # here data will be a list of dicts, in each one all the fields separated by mso
        all_data=r.json()
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        traceback.print_exc()
        raise SystemExit(e)

    activations=[]
    confidences=[]
    for i in range(len(fm.mso_set)):
        activations.append(all_data[i][names[0]])
        confidences.append(all_data[i][names[1]])
    times=all_data[0][names[2]]
    groups=all_data[0]['group_prob']
    #print(' [I] Example of groups extracted:')
    #print(groups)
    docs=[]
    # one option for the probability calculation, one long interpretation taking all prior probabilities as even   

    if start_time==[]:
        prior_evo=fm.prior_update(activations, confidences,groups, priori=[])
        n=-1
        for i in fm.faults:
            n=n+1
            template={}
            data={}
            template['timestart']=start
            template['timestop']=current_time
            template['device']=str(device)
            template['trained_version']=version
            template['fault']=fm.faults[i]
            template['probability']=prior_evo[n]
            data['doc']=template
            data['index']='prob_'
            data['type']='probability'
            docs.append(data)
    else:
        body={'device':str(device),'trained_version':version,'times':current_time,'faults':fm.faults}
        try:
            r = requests.post('http://db_manager:5001/collect-last-priors',json = body) # 'http://db_manager:5001/collect-data'
            priors=r.json()
        except:  # Extra details:  requests.exceptions.RequestException as e
            traceback.print_exc()
            priors=fm.get_priors_even()
        if sum(priors)<0.95:
            priors=fm.get_priors_even()
        prior_evo=fm.prior_update(activations, confidences,groups, priori=priors)
        for i in range(len(prior_evo[0])-1): # we substract one since prior evo includes as index 0 the prior
            n=-1
            for j in fm.faults:
                n=n+1
                template={}
                template['timestamp']=times[i]
                template['device']=str(device)
                template['trained_version']=version
                template['fault']=fm.faults[j]
                template['probability']=prior_evo[n][i+1]
                docs.append(template)
    return docs,fm.mso_set

# This function will generate the report at the end of the cycle in app.py(if nothing stoped the process)
def generate_report(analysis,probabilities,mso_set,size_mavg=20,version=""):
    # prepare db for each mso
    db_dic={}
    first=True
    for sample in analysis:
        for i in range(len(mso_set)):
            if first:
                d={'timestamp':[sample['timestamp']],'models_error':[sample['models_error'][i]],'low_bounds':[sample['low_bounds'][i]],'high_bounds':[sample['high_bounds'][i]],'activations':[sample['activations'][i]],'confidence':[sample['confidence'][i]]}
                df=pd.DataFrame(d)
                db_dic[mso_set[i]]=df
            else:
                new_row=d={'timestamp':sample['timestamp'],'models_error':sample['models_error'][i],'low_bounds':sample['low_bounds'][i],'high_bounds':sample['high_bounds'][i],'activations':sample['activations'][i],'confidence':sample['confidence'][i]}
                db_dic[mso_set[i]]=db_dic[mso_set[i]].append(new_row,ignore_index=True)
        first=False
    for m in mso_set:
        db_dic[m].sort_values(by=['timestamp'],ascending=True)
    #A. General Health Indicator
    #get total number of activations per mso
    n_samples=db_dic[mso_set[0]].shape[0]
    activations={}
    for i in range(n_samples):
        for mso in mso_set:
            if mso not in activations:
                activations[mso]=0
            if db_dic[mso].iloc[i]['activations']==1:
                activations[mso]=activations[mso]+1
    selection=100
    mso_selected=[]
    for mso in mso_set:
        # we will use x10 instead of % for the better fit with the exponential punctuation
        mso_helth=(n_samples-activations[mso])*10/n_samples
        if mso_helth<selection:
            selection=mso_helth
            mso_selected=mso
    # we use a exponential function to make the health indicator more reactive the more active samples we have, the 0.8 base is set to have a smooth value
    # 1.1202902496016716 makes that when we have 0 activations health goes to 100 
    helth=100*(1-(0.8**selection))*1.1202902496016716
    
    #B. Most Critical Timebands
    # we will find the peaks in the most relevant mso for health --> The decision is made by using the moving average of the confidence of each mso (weighted by his ratio of activations)

    max_disc=0
    interest_time=''
    for i in range(n_samples-2*size_mavg):
        w_avg=0
        j=i+2*size_mavg
        for n in mso_set:
            # 
            w_avg=w_avg+(db_dic[n].iloc[i:j]['confidence'].sum())*activations[n]/n_samples
        # We just select one result ... should we get more? Not really necessary
        if w_avg>max_disc:
            max_disc=w_avg
            t=i+size_mavg
            interest_time=db_dic[n].iloc[t]['timestamp']
    #C. Fault Interpretation
    #load the faults in a Df to filter by faults and sort by date
    df_prob=pd.DataFrame(probabilities)
    df_prob=df_prob.sort_values(by=['timestamp'],ascending=True)
    faults=list(df_prob['fault'].unique())
    times=list(df_prob['timestamp'].unique())
    most_wanted={}
    low_cap=1.5/len(faults) # we set the lower cap a bit higher than the defaut probability to filter out small effects
    for t in times:
        sub=df_prob[df_prob['timestamp'].isin([t])]
        mx=sub.loc[sub['probability'].idxmax()]
        if (mx['fault'] not in most_wanted):
            most_wanted[mx['fault']]={'count':0,'max_probability':0}
        if mx['probability']>low_cap:
            most_wanted[mx['fault']]['count']=most_wanted[mx['fault']]['count']+1      
            if mx['probability']>most_wanted[mx['fault']]['max_probability']:
                most_wanted[mx['fault']]['max_probability']=mx['probability']
    keys=list(most_wanted.keys())
    total_counts=0
    for i in keys:
        if most_wanted[i]['count']==0:
            q=most_wanted.pop(i)
        else:
            total_counts=total_counts+most_wanted[i]['count']
    selection=0
    fault_result={'fault':'No_relevant_faults','perc_among_faults':0,'perc_among_samples':0,'max_probability':0}
    for i in most_wanted:
        points=most_wanted[i]['max_probability']*most_wanted[i]['count']/total_counts
        if points>selection:
            selection=points
            fault_result={'fault':i,'perc_among_faults':most_wanted[i]['count']*100/total_counts,'perc_among_samples':most_wanted[i]['count']*100/len(times),'max_probability':most_wanted[i]['max_probability']}
    
    #D. Urgency Indicator
    # we will skip this for now
    
    # document to be uploaded
    report={'timestamp':times[-1],'health':helth,'critical_time':interest_time,'fault_interpretation':fault_result,'device':str(analysis[0]['device']),'trained_version':version}
    return report

def collect_timeband(time,machine,names,aggSeconds,shared_list):
    body={'device':machine,'names':names,'times':[time],'aggSeconds':aggSeconds}
    r = requests.post('http://db_manager:5001/collect-data',json = body) # 'http://db_manager:5001/collect-data'
    dd=r.json()
    shared_list.append(pd.DataFrame(dd))


# this class will return a subset of size s where the samples are evenly distributed to include maximum diversity 
def homo_sampling(data,cont_cond,s=50000,uncertainty=[]):
    stats=data.describe()
    if uncertainty==[]:
        for i in cont_cond:
            a=(data[i].max()-data[i].min())/(75*stats.loc['std'][i])
            uncertainty.append(a)
    lat_var=[]
    for i in data.columns:
        if i not in cont_cond:
            lat_var.append(i)
    done=False
    ws=copy.deepcopy(data)
    sets=[]
    first=True
    pd_sets=[]
    while not done:
        #ind=list(ws.index)
        it_done=False
        sets=[]
        while not it_done:
            i=np.random.randint(0,high=ws.shape[0])
            #samp=ind.pop(i)
            q=ws.iloc[i]
            filt_bool=np.array([True]*ws.shape[0])
            for cc in range(len(cont_cond)):
                l=q[cont_cond[cc]]-uncertainty[cc]
                h=q[cont_cond[cc]]+uncertainty[cc]
                filt_bool = filt_bool & np.array(ws[cont_cond[cc]]>l) & np.array(ws[cont_cond[cc]]<h)
            # with filt_bool we have extracted 
            filt_ws=ws[filt_bool]
            #ind=list(set(ind) - set(filt_ws.index))
            ws=ws.drop(filt_ws.index)
            sets.append(filt_ws)
            if ws.shape[0]==0:
                it_done=True
                print('[!] Sample set found from ws with size='+str(len(sets)))
        #once the list has been sorted in sets, we check if there are too many or too few sets        
        if len(pd_sets)>s or first:
            # we will do the previous process again using as ws the average of the samples 
            new_ws={'set_behind':[]}
            new_sets=[]
            for i in cont_cond:
                new_ws[i]=[]
            # after new_ws is initialize we fill it with the means of the new sets created
            for i in range(len(sets)):
                new_ws['set_behind'].append(i)
                st=sets[i]
                for i in cont_cond:
                    new_ws[i].append(st[i].mean())
            ws=pd.DataFrame(new_ws)
            print('[!] New working set:')
            print(ws)
            # rearrange the pd_sets so that the new set_behind points out to the combined subsets from previous iteration
            if not first:
                new_sets=[]
                for i in range(len(sets)):
                    temp_set=[]
                    ff=True
                    for j in range(sets[i].shape[0]):
                        to_add=int(sets[i].iloc[j]['set_behind'])
                        if ff:
                            temp_set=pd_sets[to_add]
                            ff=False
                        else:
                            temp_set=temp_set.append(pd_sets[to_add],ignore_index=True) 
                    new_sets.append(temp_set)
            else:
                new_sets=sets
                first=False
            for x in range(len(uncertainty)):
                uncertainty[x]=uncertainty[x]*1.25
            pd_sets=copy.deepcopy(new_sets)
            print('[!] End new set arrangement --> pd_sets size: '+str(len(pd_sets)))
        # Once the number of sets is smaller than the required samples, we stop the agregation and take on sample from each set. To fill in the remaining samples we get new samples from each set according to their size                                
        if len(pd_sets)<s:
            # we will take one sample from each and the remaining to fill up to s will be extracted according to the cumm count of samples among sets
            print('[R] Inside low set cond final part of process!')
            initial=True
            missing=s-len(pd_sets)
            dic_sets={'index':[],'size':[]}
            tot=0
            for i in range(len(pd_sets)):
                dic_sets['index'].append(i)
                tot=pd_sets[i].shape[0]
                dic_sets['size'].append(tot)
            # we arrange it in a DF to sort and obtain the cummulative function
            df_sizes=pd.DataFrame(dic_sets)
            df_sizes=df_sizes.sort_values(by=['size'],ascending=False)
            df_sizes['size']=df_sizes['size'].cumsum(axis = 0)
            hits=np.round(np.linspace(0,df_sizes['size'].max(),num=missing))
            loc=0
            new_samples=[]
            # we go though the evenly spaced numbers and collect from which
            for i in hits:
                not_yet=True
                #print(' [o] New selected size and hit: '+str(df_sizes.iloc[loc]['size'])+' | '+str(i))
                while not_yet and loc<df_sizes.shape[0]:
                    if df_sizes.iloc[loc]['size']>i:
                        new_samples.append(df_sizes.iloc[loc]['index'])
                        not_yet=False
                    else:
                        loc=loc+1
            # now we count how many elements are there from each pd_sets
            start_count=True
            for i in new_samples:
                if start_count:
                    start_count=False
                    count_hits = {i:new_samples.count(i)}
                else:
                    count_hits[i] = new_samples.count(i)
            # just load in a new var one sample from each set plus the ones listed in count_hits
            for i in range(len(pd_sets)):
                if i in count_hits:
                    si=count_hits[i]+1
                    randoms=list(np.random.randint(low=0, high=pd_sets[i].shape[0], size=(si,)))
                    add=pd_sets[i].iloc[randoms]
                else:
                    randoms=list(np.random.randint(low=0, high=pd_sets[i].shape[0], size=(1,)))
                    add=pd_sets[i].iloc[randoms]  
                if initial:
                    final_samp=add
                    initial=False
                else:
                    final_samp=final_samp.append(add,ignore_index=True)
                    
            done=True
            print(' [R] --> Final subset obtained:')
            print(final_samp)
        first=False
    # we return the    
    return final_samp

def set_new_model(mso_path,host,machine,matrix,sensors_in_tables,faults,sensors,sensor_eqs,time_bands,filt_val,filt_param,filt_delay_cap,main_ca,max_ca_jump,cont_cond,retrain=True,aggSeconds=5,sam=100000,version="",preferent=[],mso_set=[],production=False,out_var='W_OutTempUser',target_var='RegSetP',filter_stab=True):
   
    file, folder = file_location(machine,version) 
    do=False
    gen_search=False
    if preferent==[]:
        gen_search=True
    if not os.path.exists(folder):
        print(folder)
        os.mkdir(folder)
        do=True
    else:
        if retrain:
            do=True
    if do:
        print('[I] Started training process for version: '+version)
        sensors_in_tables=fix_dict(sensors_in_tables)
        faults=fix_dict(faults)
        sensors=fix_dict(sensors)
        #print(faults)
        fault_manager = fault_detector(file,mso_path,host,machine,matrix,sensors_in_tables,faults,sensors,sensor_eqs,filt_val,filt_param,filt_delay_cap,main_ca,max_ca_jump,cont_cond,aggS=aggSeconds,filter_stab=filter_stab)
        fault_manager.read_msos()
        fault_manager.MSO_residuals()
        fault_manager.time_bands=time_bands
        names,times_b=fault_manager.get_data_names(option='CarolusRex',times=time_bands)
        names.append(target_var)
        body={'device':machine,'names':names,'times':times_b,'aggSeconds':aggSeconds}
        #headers = {'Content-Type': 'application/json'}
        file_data='/models/file_data.csv'
        file_test='/models/file_test.csv'
        file_kde='/models/file_kde.csv'
        no_files=False
        try:
            fault_manager.training_data=pd.read_csv(file_data,index_col=0)
            fault_manager.test_data=pd.read_csv(file_test,index_col=0)
            fault_manager.kde_data=pd.read_csv(file_kde,index_col=0)
            if abs(sam-fault_manager.training_data.shape[0])>10:
                no_files=True
        except:
            no_files=True
        #print(fault_manager.training_data)
        if no_files:
if True:
            base_time=30 #minutes per request
            if len(times_b)==1:
                start=times_b[0][0]
                end=times_b[0][1]
                new_set=[]
                go_on=True
                from_t=start
                until_t=''
                i=0
                while go_on:
                    i=i+1
                    next_t=datetime.datetime(year=int(start[:4]), month=int(start[5:7]), day=int(start[8:10]), hour=int(start[11:13]),  minute=int(start[14:16]), second=0, microsecond=1000)+datetime.timedelta(minutes=30*i)
                    next_t=next_t.isoformat()
                    until_t=next_t[:(len(next_t)-3)]+'Z'
                    new_set.append([from_t,until_t])
                    from_t=until_t
                    if until_t>=end:
                        go_on=False
                time_set=new_set
            else:
                time_set=times_b

            try:
                #manager = multiprocessing.Manager()
                #shared_list = manager.list()
if True:
                shared_list = []
                #jobs = []
                for time in time_set:
                    #p = multiprocessing.Process(target=collect_timeband, args=(time,machine,names,aggSeconds,shared_list))
                    #p.start()
                    #jobs.append(p)
                #for proc in jobs:
                    #proc.join()
                    body={'device':machine,'names':names,'times':[time],'aggSeconds':aggSeconds}
                    r = requests.post('http://db_manager:5001/collect-data',json = body)
                    dd=r.json()
                    shared_list.append(pd.DataFrame(dd))
                first=True
                for df in shared_list:
                    if first:
                        data=df
                        first=False
                    else:
                        data=data.append(df,ignore_index=True)        
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            print('All Data Collected')
            print(data)
            # Filter Erratic rows --> ONLY for UC8 71471

            if machine=='71471':
                keep_1=data['EvapTempCirc1']<100
                step_one=data[keep_1]
                keep_2=step_one['SubCoolCir1']<100
                filtered=copy.deepcopy(step_one[keep_2])
                filtered.loc[:,'UnitStatus']=filtered.loc[:,'UnitStatus']*10
            elif machine=='74124':
                keep_1=data['EvapTempCirc1']<100
                filtered=copy.deepcopy(data[keep_1])
                filtered.loc[:,'UnitStatus']=filtered.loc[:,'UnitStatus']*10
            #####
            # Limit maximum number of samples to avoid excesive overfitting
            #if filtered.shape[0]>sam:
                #data_sampled=filtered.sample(n=sam)
            #else:
                #data_sampled=filtered
            filt_data=fault_manager.filter_samples(filtered)
            target=filt_data.pop(target_var)
            if filter_stab:
                filt_data=fault_manager.filter_stability(filt_data,target)
            fault_manager.training_data=homo_sampling(filt_data,cont_cond,s=sam)
            common = filt_data.merge(fault_manager.training_data, on=["timestamp"])
            rest=filt_data[~filt_data.timestamp.isin(common.timestamp)]
            fault_manager.test_data=rest.sample(n=int(sam/2))
            fault_manager.kde_data=homo_sampling(filt_data,cont_cond,s=int(sam/2))
            fault_manager.training_data.to_csv(file_data)
            fault_manager.test_data.to_csv(file_test)
            fault_manager.kde_data.to_csv(file_kde)
            print('[I] Saved sampled and filtered data')
            print('[I] Filtered Training Data')
            print(fault_manager.training_data)
            
        #use GA for finding the MSO set
        if gen_search:
            file_pre_ga='/models/pre_GA_search_theta.pkl'
            try:
                filehandler = open(file_pre_ga, 'rb')
                variables = pickle.load(filehandler)
                filehandler.close()
                if variables['version']==version:
                    preprocesed=True
                    theta=variables['theta']
                    preferent=variables['preferent']
                else:
                    preprocesed=False
            except:
                preprocesed=False
            if not preprocesed:
                theta, preferent = launch_analysis(matrix, mso_path, sensors, sensor_eqs, sensors_in_tables, fault_manager.test_data, fault_manager.kde_data, cont_cond, list(sensors_in_tables.values()))
                #from classes.MSO_selector_GA import find_set_v2,ga_search,angle,check_detection,check_isolability,compare_activation,cost_linearizable,cost_variables,dotproduct,length,mutate,sig_cost,single_point_cross
                print('--- [I] Result of new analysis for mso and target selection: ')
                print(theta)
                print(preferent)
                file_pre_ga='/models/pre_GA_search_theta.pkl'
                file = open(file_pre_ga, 'wb')
                pickle.dump({'theta':theta,'preferent':preferent,'version':version}, file)
                file.close()
            if mso_set==[]:
                mso_set=find_set_v2(fault_manager,theta)
        fault_manager.preferent=preferent
        fault_manager.mso_set=mso_set
        fault_manager.fault_signature_matrix_construction()
        t_a=datetime.datetime.now()
        print('[D] Right before training the residuals')
        response_dic=fault_manager.train_residuals(folder,file,cont_cond,predictor='NN',outlayers='No')
        fault_manager.get_weight_sensitivity()
        print('[D] Right after training the residuals')
        print('  [I] Fault MSO Sensitivity:  ')
        print(fault_manager.fault_mso_sensitivity)
        t_b=datetime.datetime.now()
        dif=t_b-t_a
        #fault_manager.Load(folder,file)
        print('  [T] TOTAL residual training computing time ---> '+str(dif))
        #MAKE SURE THE MODEL DOESN'T ALREADY EXIST
        """fi = open(file, 'wb')
        fault_manager.Save(folder)
        pickle.dump(fault_manager, fi)
        fi.close()
        fault_manager.Load(folder)"""
        #After the training of each residual they are stored but not loaded
        #fault_manager=load_model(file, folder)
        t_a=datetime.datetime.now()
        #SM=fault_manager.create_SM(samples=600)
        #print(SM)
        #fault_manager.load_entropy()
        #fault_manager.create_FSSM(SM)
        t_b=datetime.datetime.now()
        dif=t_b-t_a
        #print('  [T] TOTAL sensitivity analysis computing time ---> '+str(dif))
        response={}
        fault_manager.version=version
        fault_manager.Save(folder,file)
        fault_manager.data_creation=datetime.datetime.now()
        response={'filename':file,'training_data shape':fault_manager.training_data.shape,'training result':response_dic} #,'FSSM':fault_manager.FSSM
    else:
        response={'Model already exists':True}
    return response #json.dumps(response)