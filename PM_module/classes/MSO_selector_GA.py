# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:15:04 2019

@author: sega01
"""

#import copy
from classes.fault_detector_class_ES import fault_detector
import numpy as np
import pandas as pd
import math
#import itertools
#import multiprocessing
import datetime

# OBSOLETE !!!!!!!!!!!
def compare_activation(f1,f2):
    equal=True
    for mso in f1:
        if mso not in f2:
            equal=False
            
    if equal and (len(f1)!=len(f2)):
        equal=False
            
    return equal

   

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

########## GA functions ##########
def check_isolability(fa):
    fa_keys=list(fa.keys())
    i=-1
    count=0
    while i<(len(fa)-2):
        i=i+1
        j=i
        not_yet=True
        while not_yet and j<(len(fa)-1):
            j=j+1
            if fa[fa_keys[i]]==fa[fa_keys[j]]:
                not_yet=False
                count=count+1
      
    return float(count/len(fa))
# The funtion tries to evaluate efficiently if one signature is able to detect all the faults or not. As parameters the signature to evaluate, the dict of fault activations and the index of detectable faults to take into account
# IT DOESN'T CHECK ISOLABILITY --> 
def check_detection(s,fa,detectable_faults,show=False):
    search=np.where(s==1)[0]
    i=-1
    not_yet=True
    new_fa={}
    # obtain if it is detectable and 
    while i<(len(detectable_faults)-1) and not_yet:
        not_in=True
        i=i+1
        j=-1
        while j<(len(search)-1):
            j=j+1
            if search[j] in fa[detectable_faults[i]]:
                if detectable_faults[i] not in new_fa:
                    new_fa[detectable_faults[i]]=[search[j]]
                else:
                    new_fa[detectable_faults[i]].append(search[j])
                not_in=False
        not_yet=not(not_in)
    
    # if the loop didnt finishis because one fault wasnt detected
    if i<(len(detectable_faults)-1):
        penalty=1
    else:
        penalty=check_isolability(new_fa)
    if show:
        print('   [D] The penalty is: '+str(penalty))
    return penalty

# a penalty to force the selecction of the smallest msos
def cost_variables(sig,variables):
    search=np.where(sig==1)[0]
    ev=0
    for s in search:
        if s==search[0]:
            ev=np.count_nonzero(variables[s])
        else:
            ev=ev+np.count_nonzero(variables[s])
    return ev/len(search)
# cost function calculation. In this version the parameter is the Conf. Matrix, the fault activation (dic[n_faults)]) and the signature to evaluate. The constrain is added as a penalty
def sig_cost(cm,fa,sig,detectable_faults,variables,show=False):
    sig_t=np.transpose(sig)
    right_dot=np.dot(cm,sig_t)
    C=np.dot(sig,right_dot)           
    penalty=check_detection(sig,fa,detectable_faults,show=show)           
    P=penalty*100000
    V=cost_variables(sig,variables)*50
    total_cost=C+P+V
    if show:
        print('   [D] Total costs: '+str(total_cost)+' | '+'Penalty: '+str(penalty)+' | '+'P: '+str(P)+'V: '+str(V))
    return total_cost

# From two parents perform the single point crossover. The inputs are two df with one parent each.
def single_point_cross(p1,p2):
    sp1=p1['signature']
    sp2=p2['signature']
    crispr=np.random.randint(0,(len(sp1)-1))
    ch1=np.concatenate((sp1[0:crispr],sp2[crispr:len(sp1)]))
    ch2=np.concatenate((sp2[0:crispr],sp1[crispr:len(sp1)]))
    return ch1, ch2
    
# Set the mutation with the given rate mu (all rand num smaller than mu are mutated). For each element in the signature the mutation is applied:
def mutate(sig,mu):
    dices=np.random.random(len(sig))
    mu_list=np.where(dices<mu)[0]
    a=np.zeros(len(sig))
    a[mu_list]=1
    result=abs(sig-a)
    return np.array(result,dtype=int)

# LIBRARY ISSUE!! --> single iteration work to paralelize 
"""def ga_iteration(parents,dic,indx,cm,nP,mu):
    print('Inside process: ')
    print(indx)
    for j in range(math.floor(len(indx)/2)):
        index_1=indx[j]
        index_2=indx[math.floor(len(indx)/2)+j]
        #select parents
        handmaids=np.random.permutation(nP)[0:2]
        #crossbreed - currently only SPC but it could be an option among different methods
        ch1_sig,ch2_sig=single_point_cross(parents.iloc[handmaids[0]],parents.iloc[handmaids[1]])
        # mutate with mu rate
        ch1_mut=mutate(ch1_sig,mu)
        ch2_mut=mutate(ch2_sig,mu)
        # evaluate children and load the table of children
        ch1_cost=sig_cost(cm,fault_activations,ch1_mut,detectable_faults)
        ch2_cost=sig_cost(cm,fault_activations,ch2_mut,detectable_faults)
        dic[index_1]={'cost':ch1_cost,'signature':ch1_sig}
        dic[index_2]={'cost':ch2_cost,'signature':ch2_sig}"""
        
        
# Main function to launch the GA optimization process. As parameters the confussion matrix is requested and the search parameters
def ga_search(cm,fault_activations,detectable_faults,variables,nP=1000,nI=500,pC=1,mu_o=0.05):
    #seeds=
    mu=mu_o
    mso_size=len(cm)
    cm=np.array(cm)
    parents=pd.DataFrame({'cost':[],'signature':[]})
    #initialize parents
    for i in range(nP):
        sig_P=np.random.randint(2, size=mso_size)
        dices=np.random.random(len(sig_P))
        mu_list=np.where(dices<3*mu)[0]
        for m in mu_list:
            if sig_P[m]==1:
                sig_P[m]=0
        cost_P=sig_cost(cm,fault_activations,sig_P,detectable_faults,variables)
        parents=parents.append({'cost':cost_P,'signature':sig_P},ignore_index=True)
        
    parents=parents.sort_values('cost')
    # TO CALL THIS COST is: best_ever['cost']
    best_ever=parents.iloc[0]
    #start the "culture"
    nC=nP*pC
    stop_1=True
    stop_2=True
    stop_3=True
    for i in range(nI):
        """crossover splited in multiple processes
        manager = multiprocessing.Manager()
        children_dic=manager.dict()
        proc=int(multiprocessing.cpu_count())
        dic_indx=np.arange(nC)
        chunks=np.array_split(dic_indx,proc)
        jobs=[]
        for c in chunks:
            p = multiprocessing.Process(target=ga_iteration, args=(parents,children_dic,c,cm,nP,mu))
            p.start()
            jobs.append(p)
        for q in jobs:
            q.join()
        children=pd.DataFrame(children_dic)
        children=children.transpose"""
        #t_a=datetime.datetime.now()
        children=pd.DataFrame({'cost':[],'signature':[]})
        for j in range(math.floor(nC/2)):
            #select parents
            handmaids=np.random.permutation(nP)[0:2]
            #crossbreed - currently only SPC but it could be an option among different methods
            ch1_sig,ch2_sig=single_point_cross(parents.iloc[handmaids[0]],parents.iloc[handmaids[1]])
            # mutate with mu rate
            ch1_mut=mutate(ch1_sig,mu)
            ch2_mut=mutate(ch2_sig,mu)
            # evaluate children and load the table of children
            ch1_cost=sig_cost(cm,fault_activations,ch1_mut,detectable_faults,variables)
            ch2_cost=sig_cost(cm,fault_activations,ch2_mut,detectable_faults,variables)
            children=children.append({'cost':ch1_cost,'signature':ch1_mut},ignore_index=True)
            children=children.append({'cost':ch2_cost,'signature':ch2_mut},ignore_index=True)
        # mix with parents in new population, sort, check best and do all over again
        i_mix=parents.append(children,ignore_index=True)
        i_mix=i_mix.sort_values('cost')
        if i_mix.iloc[0]['cost']==best_ever['cost']:
            mu=mu*0.1
            if mu>0.15:
                mu=0.15
        else:
            mu=mu_o
        best_ever=i_mix.iloc[0]
        # set parents as the best among the new mix
        parents=i_mix.iloc[0:nP]
        if (i%100)==0:
            print('Completed Iteration #'+str(i))
            print(list(np.where(best_ever['signature']==1)[0]))
            sig_cost(cm,fault_activations,best_ever['signature'],detectable_faults,variables,show=True)
        #sig_cost(cm,fault_activations,children.iloc[3]['signature'],detectable_faults,show=True)
        #t_b=datetime.datetime.now()
        #dif=t_b-t_a
        #☺print('   [I] total time of iteration is:  '+str(dif))
    return best_ever
        
        
###################################################################################################################################
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
    detectable_faults=list(fault_activations.keys())
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
                detectable_faults.remove(positions[n])
            else:
                if n<(len(fault_activations)-1):
                    for k in range(n+1,len(fault_activations)):
                        if compare_activation(fault_activations[positions[n]],fault_activations[positions[k]]):
                            if new_fa:
                                i=i+1
                                new_fa=False
                                no_isolable.append([n,k])
                                detectable_faults.remove(positions[k])
                            if k not in no_isolable[i]:
                                no_isolable[i].append(k)
                                detectable_faults.remove(positions[k])
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

    variables=np.zeros([len(fault_manager.models),len(fault_manager.sensors)])
    v_pos={}
    i=-1
    for val in fault_manager.sensors:
        i=i+1
        v_pos[val]=i
        
    for mso in range(len(fault_manager.models)):
        for item in fault_manager.models[mso].known:   
            variables[mso,v_pos[item]]=1
                       
    best_ever = ga_search(theta,fault_activations,detectable_faults,variables)
    return list(np.where(best_ever['signature']==1)[0])#theta,fault_activations,detectable_faults,theta,best_ever      
    

                

            