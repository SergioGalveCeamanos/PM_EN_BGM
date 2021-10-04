# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:56:09 2021

@author: sega01
"""
import numpy as np
import pandas as pd
import copy

def get_joint_tables(var_names,telemetry,bins,activations,confidences,labels,mso_set):
    joint_activ={}
    for var in var_names:
        to_pd={var:pd.cut(telemetry[var].values[:,0],bins[var],labels=labels)}
        for i in range(len(mso_set)):
            name='MSO_'+str(i)
            to_pd[name]=activations[i]
        joint_activ[var]=pd.DataFrame(to_pd)
      
    # get the conditional probabilities for activations only (each var and each MSO)
    N_appear=joint_activ[var_names[0]][var_names[0]].shape[0]
    joint_results_activ={}
    for var in var_names:
        joint_results_activ[var]={}
        N_appear=joint_activ[var].shape[0]
        for i in range(len(mso_set)):
            name='MSO_'+str(i)
            joint_results_activ[var][name]={}
            subset=joint_activ[var].loc[joint_activ[var][name]==1]
            joint_results_activ[var][name]['Total_MSO_activ']=subset.shape[0]
            a = subset[var].unique()
            for j in a:
                if j==0:
                    interval='-Inf to '+str(np.round(bins[var][j+1],decimals=2))
                elif j==len(labels):
                    interval=str(np.round(bins[var][j],decimals=2))+' to +Inf'
                else:
                    interval=str(np.round(bins[var][j],decimals=2))+' to '+str(np.round(bins[var][j+1],decimals=2))
                joint_results_activ[var][name][interval]={}
                joint_results_activ[var][name][interval]['Legend_index']=j
                joint_results_activ[var][name][interval]['Activations_%']=subset.loc[joint_activ[var][var]==j].shape[0]*100/joint_results_activ[var][name]['Total_MSO_activ']
                joint_results_activ[var][name][interval]['P_joint']=subset.loc[joint_activ[var][var]==j].shape[0]/N_appear
                joint_results_activ[var][name][interval]['P_var']=joint_activ[var].loc[joint_activ[var][var]==j].shape[0]/N_appear
                joint_results_activ[var][name][interval]['P_cond']=joint_results_activ[var][name][interval]['P_joint']/joint_results_activ[var][name][interval]['P_var']
    
    # get the conditional probabilities for confidences only (each var and each MSO)  
    #create the joint tables
    joint={}
    for var in var_names:
        to_pd={var:pd.cut(telemetry[var].values[:,0],bins[var],labels=labels)}
        for i in range(len(mso_set)):
            name='MSO_'+str(i)
            to_pd[name]=confidences[i]
        joint[var]=pd.DataFrame(to_pd)
        
    # get the conditional probabilities for activations only (each var and each MSO)
    N_appear=joint[var_names[0]][var_names[0]].shape[0]
    joint_results={}
    for var in var_names:
        joint_results[var]={}
        N_appear=joint[var].shape[0]
        for i in range(len(mso_set)):
            name='MSO_'+str(i)
            joint_results[var][name]={}
            a = joint[var][var].unique()
            for j in a:
                if j==0:
                    interval='-Inf to '+str(np.round(bins[var][j+1],decimals=2))
                elif j==len(labels):
                    interval=str(np.round(bins[var][j],decimals=2))+' to +Inf'
                else:
                    interval=str(np.round(bins[var][j],decimals=2))+' to '+str(np.round(bins[var][j+1],decimals=2))
                joint_results[var][name][interval]={}
                joint_results[var][name][interval]['Legend_index']=j
                
                subset=joint[var].loc[joint[var][var]==j]
                joint_results[var][name][interval]['C_mean']=subset[name].describe()['mean']
                joint_results[var][name][interval]['P_var']=joint[var].loc[joint[var][var]==j].shape[0]/N_appear
                joint_results[var][name][interval]['C_std']=subset[name].describe()['std']
                
    return joint_results_activ,joint_results
    
def get_cond_activ_mtrs(joint_results,mso_set,labels,var_names,bins):
    mtr_condactiv={}
    mtr_perc_activ={}
    for n in range(len(mso_set)):       
        name='MSO_'+str(n)
        matr=np.zeros([len(var_names),len(labels)])
        i=-1
        for var in var_names:
            i=i+1
            for j in labels:
                if j==0:
                    interval='-Inf to '+str(np.round(bins[var][j+1],decimals=2))
                elif j==len(labels):
                    interval=str(np.round(bins[var][j],decimals=2))+' to +Inf'
                else:
                    interval=str(np.round(bins[var][j],decimals=2))+' to '+str(np.round(bins[var][j+1],decimals=2))
                if interval in joint_results[var][name]:
                    matr[i,j]=np.round(joint_results[var][name][interval]['P_cond']*100,decimals=3)
                    mtr_perc_activ[name][i,j]=joint_results[var][name][interval]['Activations_%']
                else:
                    matr[i,j]=0
                    mtr_perc_activ[name][i,j]=0
        mtr_condactiv[name]=matr
    return mtr_condactiv,mtr_perc_activ

def get_mean_std_mtrs(joint_results,mso_set,labels,var_names,bins):
    mtr_mean_set={}
    mtr_std_set={}
    for n in range(len(mso_set)):       
        name='MSO_'+str(n)
        matr_mean=np.zeros([len(var_names),len(labels)])
        matr_std=np.zeros([len(var_names),len(labels)])
        i=-1
        for var in var_names:
            i=i+1
            for j in labels:
                if j==0:
                    interval='-Inf to '+str(np.round(bins[var][j+1],decimals=2))
                elif j==len(labels):
                    interval=str(np.round(bins[var][j],decimals=2))+' to +Inf'
                else:
                    interval=str(np.round(bins[var][j],decimals=2))+' to '+str(np.round(bins[var][j+1],decimals=2))
                if interval in joint_results[var][name]:
                    matr_mean[i,j]=np.round(joint_results[var][name][interval]['C_mean'],decimals=3)
                    matr_std[i,j]=np.round(joint_results[var][name][interval]['C_std'],decimals=3)
                else:
                    matr_mean[i,j]=0
                    matr_std[i,j]=0
        mtr_mean_set[name]=matr_mean
        mtr_std_set[name]=matr_std
    
    return mtr_mean_set,mtr_std_set

#get the best value for a moving average of the size of 20% of the cells >0 in the prob matr
def moving_average_variables(matr,matr_prbs,var_names,wind_ratio=0.20):
    key_intervals={}
    best_array=[]
    for i in range(matr.shape[0]):
        best=-1
        pos=0
        wind=int(np.where(matr_prbs[i]>0)[0].shape[0]*wind_ratio)
        if wind%2==0:
            wind=wind+1
        half=int((wind-1)/2)
        if half>=1:
            for j in range(half,matr.shape[1]-half):
                mean=sum(matr[i,j-half:j+half+1])/wind
                if mean>best:
                    best=mean
                    pos=j
            best_array.append(best)
            key_intervals[var_names[i]]={'score':best,'window':wind,'position':pos}
        else:
            key_intervals[var_names[i]]={'score':'too narrow to evaluate'}
    return key_intervals,best_array

# generate analysis corrected by baseline, intermediate corrected matrices
def corrected_matrices(fm,matr_prbs,mtr_condactiv_fault,mtr_mean_fault,mtr_std_fault):
    # R weights to consider how abnormal is the appearance because if it was very common in training the baseline should be considered more
    R=np.clip(np.nan_to_num(0.5+np.sqrt(fm.matr_prbs/matr_prbs)),0.5,1.5)
    # get the corrected matrices by substracting the baseline values weighted by R
    corrected_cond={}
    corrected_mean={}
    corrected_std={}
    for n in range(len(fm.mso_set)):
        name='MSO_'+str(n)
        corrected_cond[name]=np.round(np.clip(mtr_condactiv_fault[name]-R*fm.mtr_condactiv_fault[name],0.0,None),decimals=3)
        corrected_mean[name]=np.round(np.clip(mtr_mean_fault[name]-R*fm.mtr_mean_current[name],0.0,None),decimals=3)
        # only for the std ... the substraction was not as useful
        corrected_std[name]=np.round(np.nan_to_num(np.clip(mtr_std_fault[name]/(R+fm.mtr_std_current[name]),0.0,None)),decimals=3)
    # get the last adjustment to eliminate those high means/conditionals that were only artifacts due to being barely represented
    ratio_cond={}
    ratio_mean={}
    ratio_std={}
    for n in range(len(fm.mso_set)):
        name='MSO_'+str(n)
        ratio_cond[name]=np.round(np.nan_to_num(corrected_cond[name]*(matr_prbs**0.75)),decimals=4)
        ratio_mean[name]=np.round(np.nan_to_num(corrected_mean[name]*(matr_prbs**0.75)),decimals=4)
        ratio_std[name]=np.round(np.nan_to_num(corrected_std[name]*(matr_prbs**0.75)),decimals=4)
    return R,ratio_cond,ratio_mean,ratio_std,corrected_cond,corrected_mean,corrected_std

# get best elements per var, per MSO and overall 
def final_selection(mso_set,var_names,matr_prbs,ratio_cond,ratio_mean,mtr_perc_activ,corrected_std,corrected_cond,corrected_mean,bins,activations):
    # 3) find the points with best aggregation in each variable
    results_selection_activ={}
    results_selection_mean={}
    best_arrays_activ={}
    best_arrays_mean={}
    template_print=[]
    name='MSO_'+str(0)
    for i in range(ratio_cond[name].shape[0]):
        template_print.append([])
        for j in range(ratio_cond[name].shape[1]):
            template_print[i].append(' ')
    template_print=np.array(template_print)
    
    for n in range(len(mso_set)):
        name='MSO_'+str(n)
        template_activ=copy.deepcopy(template_print)
        template_mean=copy.deepcopy(template_print)
        results_selection_activ[name],best_arrays_activ[name]=moving_average_variables(ratio_cond[name],matr_prbs,var_names,wind_ratio=0.20)
        results_selection_mean[name],best_arrays_mean[name]=moving_average_variables(ratio_mean[name],matr_prbs,var_names,wind_ratio=0.20)
        i=-1
        for var in var_names:
            i=i+1
            if results_selection_activ[name][var]['score']!='too narrow to evaluate':
                half=int((results_selection_activ[name][var]['window']-1)/2)
                template_activ[i,results_selection_activ[name][var]['position']-half:results_selection_activ[name][var]['position']+half+1]='*'
                template_mean[i,results_selection_mean[name][var]['position']-half:results_selection_mean[name][var]['position']+half+1]='*'

        
    # 4) Select / order -- Take into account Activ and Mean conf + STD of Mean
    # we get all the necesary metrics for the final ranking ... with several categories and then we add them up (like a talent contest)
    sum_up_data={}
    for n in range(len(mso_set)):
        name='MSO_'+str(n)
        sum_up_data[name]={}
        sum_up_data[name]['activ_acc']=[]
        sum_up_data[name]['prob_acc']=[]
        sum_up_data[name]['ratio_score']=[]
        sum_up_data[name]['interval']=[]
        sum_up_data[name]['mean_pos_dist_vs_wind']=[]
        sum_up_data[name]['conf_std_values_vs_avg']=[]
        sum_up_data[name]['cond_prob_mean']=[]
        sum_up_data[name]['conf_mean']=[]
        i=-1
        for var in var_names:
            i=i+1
            var=var_names[i]
            sum_up_data[name][var]={}
            if results_selection_activ[name][var]['score']!='too narrow to evaluate':
                half=int((results_selection_activ[name][var]['window']-1)/2)
                sum_up_data[name]['activ_acc'].append(sum(mtr_perc_activ[name][i,results_selection_activ[name][var]['position']-half:results_selection_activ[name][var]['position']+half+1]))
                sum_up_data[name]['prob_acc'].append(sum(matr_prbs[i,results_selection_activ[name][var]['position']-half:results_selection_activ[name][var]['position']+half+1]))
                sum_up_data[name]['ratio_score'].append(sum_up_data[name]['activ_acc'][i]/sum_up_data[name]['prob_acc'][i])
                sum_up_data[name]['interval'].append([bins[var][results_selection_activ[name][var]['position']-half],bins[var][results_selection_activ[name][var]['position']+half+1]])
                sum_up_data[name]['mean_pos_dist_vs_wind'].append(abs(results_selection_activ[name][var]['position']-results_selection_mean[name][var]['position'])/results_selection_activ[name][var]['window'])
                sum_up_data[name]['conf_std_values_vs_avg'].append(np.mean(corrected_std[name][i,results_selection_activ[name][var]['position']-half:results_selection_activ[name][var]['position']+half+1])/np.mean(corrected_std[name][i,:]))
                #print(sum_up_data[name]['conf_std_values_vs_avg'])
                sum_up_data[name]['cond_prob_mean'].append(np.mean(corrected_cond[name][i,results_selection_activ[name][var]['position']-half:results_selection_activ[name][var]['position']+half+1]))
                sum_up_data[name]['conf_mean'].append(np.mean(corrected_mean[name][i,results_selection_activ[name][var]['position']-half:results_selection_activ[name][var]['position']+half+1]))
            else:
                sum_up_data[name]['activ_acc'].append(-1)
                sum_up_data[name]['prob_acc'].append(-1)
                sum_up_data[name]['ratio_score'].append(-1)
                sum_up_data[name]['interval'].append(-1)
                sum_up_data[name]['mean_pos_dist_vs_wind'].append(-1)
                sum_up_data[name]['conf_std_values_vs_avg'].append(-1)
                sum_up_data[name]['cond_prob_mean'].append(-1)
                sum_up_data[name]['conf_mean'].append(-1)
    
    # To merge, each of the major 3 indicators is taken and ranked. 
    #The average position is weighted by the possition distances between cond_prob and mean_conf combined with the rank of std_conf
    
    import scipy.stats as ss
    result_scores={}
    selected_key_vars={}
    for n in range(len(mso_set)):
        name='MSO_'+str(n)
        selected_key_vars[name]=[]
        ratio_score=ss.rankdata(sum_up_data[name]['ratio_score'],method='dense')
        cond_prob_mean=ss.rankdata(sum_up_data[name]['cond_prob_mean'],method='dense')
        conf_mean=ss.rankdata(sum_up_data[name]['conf_mean'],method='dense')
        conf_std_values_vs_avg=np.array(sum_up_data[name]['conf_std_values_vs_avg'])
        mean_pos_dist=np.array(sum_up_data[name]['mean_pos_dist_vs_wind'])
        print(name)
        print('   Total Count Results: '+str((ratio_score+cond_prob_mean+conf_mean)))
        print('   Boost Multiplier Dist: '+str((1/(1+np.abs(mean_pos_dist)))))
        print('   Boost Multiplier STD: '+str(np.sqrt(np.abs(conf_std_values_vs_avg))))
        result_scores[name]=(ratio_score+cond_prob_mean+conf_mean)*(1/(1+np.abs(mean_pos_dist)))*np.sqrt(np.abs(conf_std_values_vs_avg)) 
        print('   Final Scores: '+str(result_scores[name]))
        #Only pay attention to the vars that represent 60(?)% of the total points
        not_assembled=True
        acc=0
        result_order=np.argsort(result_scores[name])
        i=0
        to_print=''
        while not_assembled:
            i=i+1
            acc=acc+result_scores[name][result_order[-i]]/sum(result_scores[name])
            selected_key_vars[name].append(result_order[-i])
            to_print=to_print+var_names[result_order[-i]]+', '
            if acc>=0.6:
                not_assembled=False
                print('   The Selected variables are: '+to_print)
        print('------------------------------------------------')      
        
    # now across all MSOs, what is the final result --> the MSOs with more total activations more weight right?
    weight=[]
    for n in range(len(mso_set)):
        name='MSO_'+str(n)
        weight.append(sum(activations[n]))
        
    weight=(np.array(weight)/sum(weight))**0.5
    weight=weight/sum(weight)
    
    
    for n in range(len(mso_set)):
        name='MSO_'+str(n)
        if n==0:
            combined_result=result_scores[name]*weight[n]
        else:
            combined_result=result_scores[name]*weight[n]+combined_result
    
    not_assembled=True   
    acc=0     
    result_order=np.argsort(combined_result)
    vars_final=[]
    i=0
    to_print=''
    while not_assembled:
        i=i+1
        acc=acc+combined_result[result_order[-i]]/sum(combined_result)
        vars_final.append(result_order[-i])
        to_print=to_print+var_names[result_order[-i]]+', '
        if acc>=0.6:
            not_assembled=False
            print('   The Selected variables are: '+to_print)
        
    return sum_up_data, result_scores, combined_result
    