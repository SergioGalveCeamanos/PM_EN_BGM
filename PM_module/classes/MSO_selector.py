# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:15:04 2019

@author: sega01
"""

import copy
from fault_detector_class_ES import fault_detector
import numpy as np
import math
import itertools

mso_path= r"C:\Users\sega01\Desktop\Temporal Files\LUC - Industrial PhD\Online Service\PM_docker-compose\PM_module\models\msos.txt"
machine="manager_UC65_69823.pkl"
matrix= [   [1,0,0,0,0,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # Component Eq
            [1,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
            [0,0,1,0,0,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
            [0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0],
            [1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],# Mass Eq
            [0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0], # Pressure Eq
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1], # Direct relation betweeen variables 
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # Sensor Eq
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]]

host="Not_needed"
sensors_in_tables={1:"ControlRegCompAC__VarPowerKwMSK",2:"EbmpapstFan_1_Mng__CurrPowerkWMSK",3:"Data_EVD_Emb_1__EVD__Variables__EEV_PosPercent__Val",8:"WaterFlowMeter",9:"SuctSH_Circ1",10:"DscgTempCirc1",11:"SubCoolTempCir1",13:"EvapTempCirc1",14:"CondTempCirc1",17:"W_OutTempUser",18:"W_OutTempEvap",19:"W_InTempUser",21:"FiltPress",24:"PumpPress",25:"ExtTemp"}
sensors={1:"wc",2:"wf",3:"v",8:"mw",9:"tr1",10:"tr2",11:"tr3",13:"pr1",14:"pr2",17:"tw1",18:"tw2",19:"tw3",21:"pw1",24:"pw4",25:"tatm"}
faults={1:"fc1",2:"fc2",4:"fc3",5:"fc4",7:"fc5",9:"fl1",10:"fl2",11:"fo1",12:"fo2",13:"fo3",14:"fo4",15:"fo5",21:"fs1",22:"fs2",23:"fs3",24:"fs4",25:"fs5",26:"fs6",27:"fs8",28:"fs9",29:"fs10",30:"fs11",31:"fs12",32:"fs13"}
sensor_eqs={"wc":18,"wf":19,"v":20,"mw":21,"tr1":22,"tr2":23,"tr3":24,"pr1":25,"pr2":26,"tw1":27,"tw2":28,"tw3":29,"pw1":30,"pw4":31,"tatm":32}
mso_set=[46, 1, 53, 13, 3, 59, 0, 6]
filename="fault_detection_manager.pkl"
preferent= ["SuctSH_Circ1","DscgTempCirc1","SubCoolTempCir1","W_OutTempUser","W_OutTempEvap","W_InTempUser"]
filt_value=9.0
filt_param='UnitStatus'
fault_manager = fault_detector(filename,mso_path,host,machine,matrix,sensors_in_tables,faults,mso_set,sensors,sensor_eqs,preferent,filt_value,filt_param)
fault_manager.read_msos()
fault_manager.MSO_residuals()

##################### MARKING #######################################

repeated=0
combos=[]
for mso in fault_manager.models:
    if mso.faults not in combos:
        combos.append(mso.faults)
        
    else:
        repeated=repeated+1
    
fault_activations=[]
fault_lists={}
n=-1
for fault in faults:
    i=0
    n=n+1
    fault_activations.append([])
    fault_lists[fault]=[]
    for mix in combos:
        i=i+1
        if faults[fault] in mix:
            fault_activations[n].append(i)
            fault_lists[fault].append(i)
            
##################### MARKING #######################################
def get_reward(eq_set,fault_manager):
    used_msos=[]
    used_variables=[]
    for mso in eq_set:
        if mso not in used_msos:
            used_msos.append(mso)
            for variable in fault_manager.models[mso].known:
                if variable not in used_variables:
                    used_variables.append(variable)
                
    result=10*(len(used_variables))/len(used_msos)
    return result


# OBSOLETE !!!!!!!!!!!
def compare_activation(f1,f2):
    equal=True
    for mso in f1:
        if mso not in f2:
            equal=False
            
    if equal and (len(f1)!=len(f2)):
        equal=False
            
    return equal

# OBSOLETE !!!!!!!!!!!
#from a set of mso identify the activation of each fault
def mso_set_activation(min_set,fault_manager):
    fault_activations=[]
    n=-1
    for fault in fault_manager.faults:
        i=0
        n=n+1
        fault_activations.append([])
        for m in min_set:
            i=i+1
            if fault_manager.faults[fault] in fault_manager.combos[m]:
                fault_activations[n].append(m) 
                
    return fault_activations

# OBSOLETE !!!!!!!!!!!
#with this function the objective is to get the mso_set proposed and see if all the faults get to be distinguished one from another
def detectable_set(min_set,fault_manager,marks):
    mso_set=mso_set_activation(min_set,fault_manager)
    detectable=True
    n=-1
    #set all marks but the initial marking to -1 ... then change those needed to 0
    for mark in marks:
        n=n+1
        if mark>=0:
            if abs(mark)>0:
                marks[n]=abs(marks[n])
            else:
                marks[n]=-1
    for n in range(len(mso_set)):
        if marks[n]!=[-2]:
            if mso_set[n]==[]:
                detectable=False
            if n<len(mso_set)-1:
                for k in range((n+1),len(mso_set)):
                    if compare_activation(mso_set[n],mso_set[k]):
                        if not (marks[n]==marks[k] and marks[n]<-2):
                            detectable=False
                            marks[n]=0
                            if marks[k]!=[-2]:
                                marks[k]=0
                
    return marks

# OBSOLETE !!!!!!!!!!!
def get_combinations(mso_set,fault_manager):
    combinations=list(itertools.product(mso_set[:]))
    result=copy.deepcopy(combinations)
    for element in combinations:
        i=0
        not_repeated=True
        while not_repeated and i<len(element):
            repeated=element.count(element[i])
            i=i+1
            if repeated>1:
                result.remove(element)
    return result
  
# are all variables included?
def check_variables(mso_set,fault_manager):
    not_all=False
    counts={}
    for i in fault_manager.sensors:
        counts[fault_manager.sensors[i]]=0
        for mso in mso_set:
            if i in fault_manager.models[mso].known:
                counts[fault_manager.sensors[i]]=counts[fault_manager.sensors[i]]+1
        if counts[fault_manager.sensors[i]]==0:
            not_all=True
    return not_all, counts

#evaluate how good a new mso is based on the number of appearances, the reward and the difference in the marking (Y-X) to see how relevant it is
def get_score(Y,appearances,fault_manager,mso_set):
    reward=get_reward(mso_set,fault_manager)
    app= 1/(appearances+10)
    i=-1
    news=np.zeros(len(Y))
    for fault in Y: #enough to check if all equals?++
        i=i+1
        if news[i]==0:
            for check in range(i+1,len(Y)):
                if np.array_equal(fault,Y[check]):
                    news[i]=1
                    news[check]=1
    not_all, counts=check_variables(mso_set,fault_manager)
    
    missing=0
    for c in counts:
        if counts[c]==0:
            missing=missing+1
            
    detectable=3*(len(news)-np.count_nonzero(news))
    score=reward+detectable*app-missing*1
    print(' [*] Score, Reward, Detectable, Missing --> '+str(score)+', '+str(reward)+', '+str(detectable*app)+', '+str(missing*1))
    return score

def is_undetected(Y):
    i=-1
    news=np.zeros(len(Y))
    for fault in Y: #enough to check if all equals?++
        i=i+1
        if news[i]==0:
            for check in range(i+1,len(Y)):
                if np.array_equal(fault,Y[check]):
                    news[i]=1
                    news[check]=1
    
    return (np.count_nonzero(news)!=0)


    
 
def find_set(fault_activations,fault_manager,combos):
    no_isolable=[]
    base_set=[]
    
    
    #get those faults that cant be detectable and isolable so that are not taken into acount in the evaluation of the minimal sets
    i=-1
    for n in range(len(fault_activations)):
        go=True
        for j in no_isolable:
            if n in j:
                go = False
        # the group of no isolable faults are grouped     
        if go:
            new_fa=True
            if fault_activations[n]==[]:
                i=i+1
                no_isolable.append([n])
            else:
                if n<(len(fault_activations)-1):
                    for k in range(n+1,len(fault_activations)):
                        if compare_activation(fault_activations[n],fault_activations[k]):
                            if new_fa:
                                i=i+1
                                new_fa=False
                                no_isolable.append([n,k])
                            if k not in no_isolable[i]:
                                no_isolable[i].append(k)
                
    for no_go_set in no_isolable:
        for no_go in no_go_set:
            for mso in fault_activations[no_go]:
                if mso not in base_set:
                    base_set.append(mso)
        #fault_activations[no_go]=[-1]
        
    
    #with the undetected vector the faults that are left to be detected are highlighted       
    undetected=np.zeros(len(fault_activations)) # array to identify if the 
    team=0
    #to remove the non isolable or detectable faults from the iterative consideration
    fault_size=len(faults)
    excluded_faults=[]
    for no_go_set in no_isolable:
        if len(no_go_set)>1:
            team=team+1
            for no_go in no_go_set:
                fault_size=fault_size-1
                excluded_faults.append(no_go)
                undetected[no_go]=team
        else:
            fault_size=fault_size-1
            excluded_faults.append(no_go_set[0])
            undetected[no_go_set[0]]=-2
    
    #obtain a evaluation of each mso, the fewer appearances the higher the isolability they provide
    #in the same loop the reference marking of faults and msos is obtained without considering the non isolable/detrectable
    R=np.zeros([fault_size,len(fault_manager.models)])
    X=np.zeros([fault_size,len(fault_manager.models)])
    appearances=np.zeros(len(fault_manager.models))
    n=-1
    m=-1
    for t in fault_activations:
        m=m+1
        if m not in excluded_faults:
            n=n+1
            for mso in t:
                R[n][(mso-1)]=1
                appearances[(mso-1)]=appearances[(mso-1)]+1
            
    
    new_marking=np.zeros(fault_size)
    #the mso_set will include each step of the while loop, with a initial base_set for the msos required for the non isolable ones
    mso_set=[] # This is a PROBLEM !! The base set is too large and we dont avoid the msos already included to be considered again
    undetected=True
    #not_all=True
    checks=[]
    while undetected: #or not_all
        best=0
        for n in range(len(fault_manager.models)):
            if (n not in mso_set) and (appearances[n]>0):
                checks.append(n)
                Y=copy.deepcopy(X)
                for m in range(fault_size):
                    if R[m][n]==1:
                        Y[m][n]=1
                test_set=copy.deepcopy(mso_set)
                test_set.append(n)
                if best<get_score(Y,appearances[n],fault_manager,test_set):
                    best=get_score(Y,appearances[n],fault_manager,test_set)
                    next_mso=n  # REMEMBER: The MSO were listed from 1 to N, but for the index in python it was changed to 0 to N-1 in this function
                    next_X=copy.deepcopy(Y)

        X=next_X
        mso_set.append(next_mso)
        undetected=is_undetected(X)
        not_all,counts=check_variables(mso_set,fault_manager)
        # ALMOST WORKING !!!! Good score rule?
        # REMEMBER: The result require to include the MSOs that help to identify the non isolable sets removed before
       
    return mso_set,X, excluded_faults


# Funtions to compute the angle between vectors
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    if v1 != v2:
        aang=math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    else:
        aang=1    
  return aang


# A new approach from scratch
def find_set_v2(fault_manager):
    #prepare the set of msos activated per each fault:
    fault_activations={}
    positions={}
    i=0
    for fault in fault_manager.faults:
        new_f=[]
        for mso in range(len(fault_manager.models)):
            if fault_manager.faults[fault] in fault_manager.models[mso].faults:
                new_f.append(mso)
        fault_activations[fault]=new_f
        positions[i]=fault
        i=i+1
            
    # First create vectors for each MSO, with the non isolable variables grouped and the non-identifiable eliminated
    no_isolable=[]
    no_detectable=[]
    #get those faults that cant be detectable and isolable so that are not taken into acount in the evaluation of the minimal sets
    i=-1
    for n in range(len(fault_activations)):
        go=True
        for j in no_isolable:
            if n in j:
                go = False
        # the group of no isolable faults are grouped     
        if go:
            new_fa=True
            if fault_activations[positions[n]]==[]:
                no_detectable.append(n)
            else:
                if n<(len(fault_activations)-1):
                    for k in range(n+1,len(fault_activations)):
                        if compare_activation(fault_activations[positions[n]],fault_activations[positions[k]]):
                            if new_fa:
                                i=i+1
                                new_fa=False
                                no_isolable.append([n,k])
                            if k not in no_isolable[i]:
                                no_isolable[i].append(k)
    #prepare reference dics that can link the old faults with the new ones while preparing the signature of the msos in a dic format
    dic_new_to_old={}
    dic_old_to_new={}
    pos=0
    for n in fault_activations:
        if n not in no_detectable:
            keep=True
            for g in no_isolable:
                if n in g:
                    keep=False
            if keep:
                dic_new_to_old[pos]=[n]
                dic_old_to_new[n]=pos
                pos=pos+1
    for g in no_isolable:
        dic_new_to_old[pos]=g
        for k in g:
            dic_old_to_new[k]=pos
        pos=pos+1
    
    # Create the list/dic of vectors that define the MSOs
    mso_vec={}
    for mso in range(len(fault_manager.models)):
        l=[]
        for i in range(len(dic_new_to_old)):
            if mso in fault_activations[dic_new_to_old[i][0]]:
                l.append(1)
            else:
                l.append(0)
        mso_vec[mso]=l
        
    theta=[]
    for i in range(len(fault_manager.models)):
        theta.append([])
        for j in range(len(fault_manager.models)):
            theta[i].append(angle(mso_vec[i],mso_vec[j]))
            
    return mso_vec            
    
            
test_v2= find_set_v2(fault_manager)       
                    
            
        
    
"""############# RUN test for marking #############################
fault_manager.fault_signature_matrix()
result,X,excluded_faults=find_set(fault_activations,fault_manager,combos)
mso_isolated=[]
previous=[]
for f in fault_lists:################################## Include Isolable Faults
    if len(fault_lists[f])>0 and fault_lists[f] in previous:
        if fault_lists[f] not in mso_isolated:
            mso_isolated.append(fault_lists[f])
    previous.append(fault_lists[f])
    
not_all,counts=check_variables(result,fault_manager)   
not_used=[]
for n in counts:
    if counts[n]==0:
        for p in sensors_in_tables:
            if sensors_in_tables[p]==n:
                not_used.append(p)
                
                
'''to_add=[]
for candidates in mso_isolated:
    best=-1
    for mso in candidates:
        points=0
        if mso not in result:
            print('MSO: '+str(mso)+' | variables: '+str(fault_manager.models[mso].known))
            for var in fault_manager.models[mso].known:
                
                if var in not_used:
                    points=points+1
            if points>best:
                best=points
                new_mso=mso
                new_vars=fault_manager.models[mso].known
    to_add.append(new_mso)'''
            
            
        
################################################################
# Check that the solution doesn't have extra MSOs
def evaluate_result(result,X):
    evaluation=[]
    for n in result:
        test=np.delete(X,(n),1)
        neccesary=False
        i=0
        for fault in test:
            i=i+1
            for check in range(i,len(test)):
                if  np.array_equal(fault,test[check]):
                    neccesary=True
            if np.count_nonzero(fault)==0:
                neccesary=True
        
        if not neccesary:
            evaluation.append(n)
                    
    return evaluation
                    
evaluation=evaluate_result(result,X)
#lets do a second round of this
new_evaluation=[]
for reduce in evaluation:
    new_x=np.delete(X,(reduce),1)
    new_result=copy.deepcopy(result)
    new_result.remove(reduce)
    #fix indices
    for n in range(len(new_result)):
        if new_result[n]>reduce:
            new_result[n]=new_result[n]-1
    
    # remember that the new 
    new_evaluation.append([[reduce],[evaluate_result(new_result,new_x)]])"""

    
        
                

            