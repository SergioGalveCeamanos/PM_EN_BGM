# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:10:40 2021

@author: sega01
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
from scipy import linalg
import matplotlib as mpl
from scipy import stats
    

################################################################################
def fix_dict(d):
    new_d={}
    for n in d:
        new_d[int(n)]=d[n]
        
    return new_d
def get_weight_score(weights,alpha=1.0):
    ordered=np.sort(abs(weights))
    tot=sum(abs(weights))
    score=0
    for q in range(len(ordered)-1):
        pos=len(ordered)-1-q
        score=score+((ordered[pos]-ordered[pos-1])/tot)**2
    return alpha*score
        
# Start training all possible combinations
def train_model(train,test,mso_variables,mso_outs):
    new_model={}
    X_train = train[mso_variables]
    y_train = train[mso_outs]
    X_test = test[mso_variables].iloc[:2000]
    y_test = test[mso_outs].iloc[:2000]
    
    lin_reg_mod = ElasticNetCV(l1_ratio=[.1, .2, .4, .5, ],cv=10,n_jobs=4,max_iter=1000)
    
    lin_reg_mod.fit(X_train, y_train)

    pred = lin_reg_mod.predict(X_test)
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
    test_set_r2 = r2_score(y_test, pred)
    
    new_model['model']=lin_reg_mod
    new_model['r2_score']=test_set_r2
    new_model['rmse_score']=test_set_rmse
    new_model['coeff']=lin_reg_mod.coef_
    new_model['offset']=lin_reg_mod.intercept_
    new_model['w_score']=get_weight_score(lin_reg_mod.coef_)
    new_model['total_score']=(1-new_model['r2_score'])+new_model['w_score']
    return new_model

 # 0 = not used, 1 = used, 2 = target
def get_new_ind_row(i,target,source,variables):
    template_index={'index': 0}
    for v in variables:
        template_index[v]=0
    #template_index={'index': 0,'InvInfoCirc1.Info_MotPwr': 0,'EbmpapstFan_1_Mng.ElectrInfo_EBM_1.CurrPower': 0,'Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val': 0,'WaterFlowMeter': 0,'SuctSH_Circ1': 0,'DscgTempCirc1': 0,'SubCoolCir1': 0,'EvapTempCirc1': 0,'CondTempCirc1': 0,'W_OutTempUser': 0,'W_OutTempEvap': 0,'W_InTempUser': 0,'FiltPress': 0,'PumpPress': 0,'ExtTemp': 0}
    template_index['index']=i
    template_index[target]=2
    for s in source:
        template_index[s]=1
    return template_index


def get_biggest(arr,max_amount=0.7): # get the items that account for a 0.7 (max_am) of the total absolute value
    elms =[]   
    sel_we=0
    tot=sum(abs(arr))
    arr_a=abs(arr)
    elms.append( max(range(len(arr_a)), key=arr_a.__getitem__))
    sel_we=sel_we+arr_a[elms[0]]
    arr_b=copy.deepcopy(np.delete(arr_a,elms[0]))
    q=0
    while sel_we/tot<max_amount:
        q=q+1
        new=max(range(len(arr_b)), key=arr_b.__getitem__)
        elms.append(np.where(arr_a == arr_b[new])[0][0])
        sel_we=sel_we+arr_a[elms[q]]
        arr_b=np.delete(arr_b,new)
    return elms

def silhouette_score(estimator, X):
    try:
        clusters = estimator.predict(X)
        score = metrics.silhouette_score(X, clusters, metric='euclidean')
    except:
        score=-1
    return score

############################################################################################
# input examples
#msos_path=r"C:\Users\sega01\Desktop\Temporal Files\LUC - Industrial PhD\Online Service\BNN\msos_list.txt"
#sensors={"1":"wc","2":"wf","3":"v","4":"wp","8":"mw","9":"tr1","10":"tr2","11":"tr3","13":"pr1","14":"pr2","17":"tw1","18":"tw2","19":"tw3","21":"pw1","24":"pw4","25":"tatm"}

#sensor_eqs={"wc":18,"wf":19,"v":20,"mw":21,"tr1":22,"tr2":23,"tr3":24,"pr1":25,"pr2":26,"tw1":27,"tw2":28,"tw3":29,"pw1":30,"pw4":31,"tatm":32},
#sens_codes={"1":"InvInfoCirc1.Info_MotPwr","2":"EbmpapstFan_1_Mng.ElectrInfo_EBM_1.CurrPower","3":"Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val","8":"WaterFlowMeter","9":"SuctSH_Circ1","10":"DscgTempCirc1","11":"SubCoolCir1","13":"EvapTempCirc1","14":"CondTempCirc1","17":"W_OutTempUser","18":"W_OutTempEvap","19":"W_InTempUser","21":"FiltPress","24":"PumpPress","25":"ExtTemp"}
#data_test=pd.read_csv("file_test.csv",index_col=0)
#data_kde=pd.read_csv("file_kde.csv",index_col=0)


# Main trigger
def launch_analysis(str_matrix,msos_path,sensors,sensor_eqs,sens_codes,data_test,data_kde,cont_cond,variables):
    ################################################################################
    # GET LIST OF LINKED VARIABLES TO LAUNCH ITERATIVE TRAINING
    msos=[]
    f = open(msos_path, "r")
    for x in f:
        a=x.split()
        l=[]
        for new in a[2:]:
            l.append(int(new))
        msos.append(l)
    f.close()
    
    var_links={}
    for v in sensors:
        var_links[v]=[]
    n=0
    for mso in msos:
        n=n+1
        known=[] 
        variables=[]
        for eq in mso:
            eq_vars=str_matrix[(eq-1)]
            i=0
            for new in eq_vars:
                i=i+1
                if (i not in variables) and (new==1):
                    variables.append(i)
                    if i in sensors:
                        known.append(i)
        i=-1
        for v in known:
            i=i+1
            rest=copy.deepcopy(known)
            q=rest.pop(i)
            for j in rest:
                if sens_codes[j] not in var_links[v]:
                    var_links[v].append(sens_codes[j])
    

    ################################################################################
    # Prepare data for training ... A cluster only
    data_test=data_test.drop('timestamp',axis=1)
    data_test=data_test.drop('UnitStatus',axis=1)
    data_kde=data_kde.drop('timestamp',axis=1)
    data_kde=data_kde.drop('UnitStatus',axis=1)
    copy_train=copy.deepcopy(data_kde)
    from sklearn.preprocessing import MinMaxScaler
    scalers=[]
    for var in copy_train.columns:
        scaler = MinMaxScaler()
        telem_data=copy_train[var].values.reshape(-1,1)
        scaler.fit(telem_data)
        scalers.append(scaler)
        data_kde[var] = scaler.transform(telem_data)
        data_test[var] = scaler.transform(data_test[var].values.reshape(-1,1))
            
    
    #cont_cond=['ExtTemp', 'InvInfoCirc1.Info_MotPwr', 'W_OutTempUser', 'WaterFlowMeter']
    new_df=data_kde[cont_cond]
    no_outl=new_df[(np.abs(stats.zscore(new_df)) < 3).all(axis=1)]
    grid = GridSearchCV(MiniBatchKMeans(),
                        {'n_clusters': [4],'batch_size':[100,300,500,750,1000],'max_iter':[1200]},n_jobs=3,scoring=silhouette_score) # 20-fold cross-validation
    
    grid.fit(no_outl.values)
    kmeans = MiniBatchKMeans(n_clusters=grid.best_params_['n_clusters'], batch_size=grid.best_params_['batch_size'],max_iter=5000).fit(no_outl.values)
    groups_km=kmeans.predict(no_outl.values)
    print(np.unique(groups_km, return_counts=True))    
    
    regions=np.unique(groups_km)
    clustering=kmeans
    models={}
    pred_weighted={}
    train_groups=clustering.predict(data_kde[cont_cond].values)
    test_groups=clustering.predict(data_test[cont_cond].values)
    #probs_test=clustering.predict_proba(data_test[cont_cond].values)
    train_data_groups={}
    test_data_groups={}
    covs={}
    for t in regions:
        locats=np.where(train_groups == t)[0]
        train_data_groups[t]=data_kde.iloc[locats]
        covs[t]=data_kde.iloc[locats].cov()
        locats=np.where(test_groups == t)[0]
        test_data_groups[t]=data_test.iloc[locats]     
    
    ################################################################################
    # iterate and search    
    regions=[np.argmax((np.unique(groups_km,return_counts=True)[1]))]
    t=regions[0]
    template_index={'index':[0]} # This will be a df for the later model retrieval ... models go in a huge dict and we get indexes from this DF
    for s in sens_codes:
        template_index[sens_codes[s]]=[0] # 0 = not used, 1 = used, 2 = target
    models={}
    index_to_models=pd.DataFrame(template_index)
    i=i-1
    for t in regions:
        for v0 in var_links:
            if len(var_links[v0])>2:
                model=train_model(train_data_groups[t],test_data_groups[t],var_links[v0],sens_codes[v0])
                #load model
                i=i+1
                models[i]=model
                index_to_models=index_to_models.append(get_new_ind_row(i,sens_codes[v0],var_links[v0],variables),ignore_index=True)
                
                main_w_1=get_biggest(model['coeff'])
                for n in main_w_1:
                    source_1=copy.deepcopy(var_links[v0])
                    delet=source_1.pop(n)
                    if len(source_1)>2:
                        model=train_model(train_data_groups[t],test_data_groups[t],source_1,sens_codes[v0])
                        #load model
                        i=i+1
                        models[i]=model
                        index_to_models=index_to_models.append(get_new_ind_row(i,sens_codes[v0],source_1,variables),ignore_index=True)
                        main_w_2=get_biggest(model['coeff'])
                        for r in main_w_2:
                            source_2=copy.deepcopy(source_1)
                            delet=source_2.pop(r)
                            if len(source_2)>2:
                                model=train_model(train_data_groups[t],test_data_groups[t],source_2,sens_codes[v0])
                                #load model
                                i=i+1
                                models[i]=model
                                index_to_models=index_to_models.append(get_new_ind_row(i,sens_codes[v0],source_2,variables),ignore_index=True)
                                main_w_3=get_biggest(model['coeff'])
                                for z in main_w_3:
                                    source_3=copy.deepcopy(source_2)
                                    delet=source_3.pop(z)
                                    if len(source_3)>2:
                                        model=train_model(train_data_groups[t],test_data_groups[t],source_3,sens_codes[v0])
                                        #load model
                                        i=i+1
                                        models[i]=model
                                        index_to_models=index_to_models.append(get_new_ind_row(i,sens_codes[v0],source_3,variables),ignore_index=True)
                                        #main_w_4=get_biggest(model['coeff'])
                                
    theta=np.zeros((len(sens_codes),len(sens_codes)))
    ke=list(sens_codes.keys())
    for i in range(len(sens_codes)):
        for j in range(len(sens_codes)):
            if i!=j:
                search=index_to_models.loc[(index_to_models[sens_codes[ke[j]]] == 1) & (index_to_models[sens_codes[ke[i]]] == 2)]
                if search.shape[0]>0:
                    s=0
                    for q in range(search.shape[0]):
                        s=s+models[search.iloc[q]['index']]['total_score']
                    theta[i][j]=s/search.shape[0]
    
    outs={}
    for i in range(len(sens_codes)):
        search=index_to_models.loc[(index_to_models[sens_codes[ke[i]]] == 2)]
        if search.shape[0]>0:
            s=0
            for q in range(search.shape[0]):
                s=s+models[search.iloc[q]['index']]['total_score']
            outs[sens_codes[ke[i]]]=s/search.shape[0]           
                        
    return theta, outs                     

#msos_path=r"C:\Users\sega01\Desktop\Temporal Files\LUC - Industrial PhD\Online Service\BNN\msos_list.txt"
#sensors=fix_dict({"1":"wc","2":"wf","3":"v","4":"wp","8":"mw","9":"tr1","10":"tr2","11":"tr3","13":"pr1","14":"pr2","17":"tw1","18":"tw2","19":"tw3","21":"pw1","24":"pw4","25":"tatm"})
#sensor_eqs={"wc":18,"wf":19,"v":20,"mw":21,"tr1":22,"tr2":23,"tr3":24,"pr1":25,"pr2":26,"tw1":27,"tw2":28,"tw3":29,"pw1":30,"pw4":31,"tatm":32},
#sens_codes=fix_dict({"1":"InvInfoCirc1.Info_MotPwr","2":"EbmpapstFan_1_Mng.ElectrInfo_EBM_1.CurrPower","3":"Data_EVD_Emb_1.EVD.Variables.EEV_PosPercent.Val","8":"WaterFlowMeter","9":"SuctSH_Circ1","10":"DscgTempCirc1","11":"SubCoolCir1","13":"EvapTempCirc1","14":"CondTempCirc1","17":"W_OutTempUser","18":"W_OutTempEvap","19":"W_InTempUser","21":"FiltPress","24":"PumpPress","25":"ExtTemp"})
#data_test=pd.read_csv("file_test.csv",index_col=0)
#data_kde=pd.read_csv("file_kde.csv",index_col=0)
#th,ou=launch_analysis(str_matrix, msos_path, sensors, sensor_eqs, sens_codes, data_test, data_kde)
#x_axis_labels = ["wc","wf","v","mw","tr1","tr2","tr3","pr1","pr2","tw1","tw2","tw3","pw1","pw4","tatm"] # labels for x-axis
#y_axis_labels = ["wc","wf","v","mw","tr1","tr2","tr3","pr1","pr2","tw1","tw2","tw3","pw1","pw4","tatm"] # labels for y-axis
#import seaborn as sns
#sns.heatmap(th,xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True)