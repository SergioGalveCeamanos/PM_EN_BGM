# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:56:09 2021

@author: sega01
"""
import numpy as np
import pandas as pd
import copy
from matplotlib import cm as CM
import matplotlib.pyplot as plt
import os
from .pdf_printing import PDF
import datetime
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.colors import LogNorm
from socket import gethostname
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import json

def get_joint_tables(var_names,telemetry,bins,activations,confidences,labels,mso_set):
    joint_activ={}
    for var in var_names:
        to_pd={var:pd.cut(telemetry[var].values,bins[var],labels=labels)}
        for i in range(len(mso_set)):
            name='MSO_'+str(i)
            to_pd[name]=activations[i]
            print(str(len(to_pd[name]))+str(to_pd[var]))
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
        to_pd={var:pd.cut(telemetry[var].values[:],bins[var],labels=labels)}
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
                
    return joint_results_activ,joint_results,N_appear,joint
    
def get_cond_activ_mtrs(joint_results,mso_set,labels,var_names,bins):
    mtr_condactiv={}
    mtr_perc_activ={}
    for n in range(len(mso_set)):       
        name='MSO_'+str(n)
        matr=np.zeros([len(var_names),len(labels)])
        mtr_perc_activ[name]=np.zeros([len(var_names),len(labels)])
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
    R=np.clip(np.nan_to_num(0.5+np.sqrt(fm.matr_prbs/(matr_prbs+0.00001))),0.5,1.5)
    # get the corrected matrices by substracting the baseline values weighted by R

    corrected_cond={}
    corrected_mean={}
    corrected_std={}
    for n in range(len(fm.mso_set)):
        name='MSO_'+str(n)
        corrected_cond[name]=np.round(np.clip(mtr_condactiv_fault[name]-R*fm.mtr_condactiv_fault[name],0.0,None),decimals=3)
        corrected_mean[name]=np.round(np.clip(mtr_mean_fault[name]-R*fm.mtr_mean_current[name],0.0,None),decimals=3)
        corrected_std[name]=np.round(np.nan_to_num(np.clip(mtr_std_fault[name]/(R+fm.mtr_std_current[name]),0.0,None)),decimals=3)# only for the std ... the substraction was not as useful
    # get the last adjustment to eliminate those high means/conditionals that were only artifacts due to being barely represented
if True:    
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
if True:
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
if True:
    template_activ_set={}
    template_mean_set={}
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
        template_activ_set[name]=template_activ
        template_mean_set[name]=template_mean
        
    # 4) Select / order -- Take into account Activ and Mean conf + STD of Mean
    # we get all the necesary metrics for the final ranking ... with several categories and then we add them up (like a talent contest)
if True:    
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
                sum_up_data[name]['cond_prob_mean'].append(np.mean(corrected_cond[name][i,results_selection_activ[name][var]['position']-half:results_selection_activ[name][var]['position']+half+1]))
                sum_up_data[name]['conf_mean'].append(np.mean(corrected_mean[name][i,results_selection_activ[name][var]['position']-half:results_selection_activ[name][var]['position']+half+1]))
            else:
                sum_up_data[name]['activ_acc'].append(0)
                sum_up_data[name]['prob_acc'].append(0)
                sum_up_data[name]['ratio_score'].append(0)
                sum_up_data[name]['interval'].append(0)
                sum_up_data[name]['mean_pos_dist_vs_wind'].append(0)
                sum_up_data[name]['conf_std_values_vs_avg'].append(0)
                sum_up_data[name]['cond_prob_mean'].append(0)
                sum_up_data[name]['conf_mean'].append(0)
    
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
        
    return sum_up_data, result_scores, combined_result, template_activ_set, template_mean_set

def send_email_pdf_figs(path_to_pdf, subject, message, destination, password):
    ## credits: http://linuxcursor.com/python-programming/06-how-to-send-pdf-ppt-attachment-with-html-body-in-python-script

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('sega01luc@gmail.com',password)
    msg = MIMEMultipart()

    message = f'{message}\nSend from Hostname: {gethostname()}'
    msg['Subject'] = subject
    msg['From'] = 'sega01luc@gmail.com'
    msg['To'] = destination
    # Insert the text to the msg going by e-mail
    msg.attach(MIMEText(message, "plain"))
    # Attach the pdf to the msg going by e-mail
    with open(path_to_pdf, "rb") as f:
        #attach = email.mime.application.MIMEApplication(f.read(),_subtype="pdf")
        attach = MIMEApplication(f.read(),_subtype="pdf")
    attach.add_header('Content-Disposition','attachment',filename=str(path_to_pdf))
    msg.attach(attach)
    # send msg
    server.send_message(msg)
        
#                            device,version,to_plot,ma,heard_faults,list(,matr_prbs,fm.matr_prbs,    y_label_hm,mtr_condactiv_fault,mtr_mean_fault,template_activ_set, template_mean_set, sum_up_data, combined_resul
def launch_report_generation(device,version,health,ma,heard_faults,faults,mtr_prbs,train_prbs,labels,y_labe,mtr_condactiv,mtr_mean_fault,template_activ_set, template_mean_set, sum_up_data, combined_result):
    root_folder='/models/output_document/'
    if not os.path.exists(root_folder):
        print(root_folder)
        os.mkdir(root_folder)
    
    document=PDF(tit='Health Report - SN: '+str(device)+' - v: '+version+' - '+datetime.datetime.now().ctime())

    fe = fm.FontEntry(fname='BrandonGrotesqueOffice-Light.ttf',name='Brandom')
    fm.fontManager.ttflist.insert(0, fe) # or append is fine
    #mpl.rcParams['font.family'] = fe.name # = 'your custom ttf font name'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = fe.name
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    
    # page 0 - Summary page: 3 faults, 3 vars, timebands relevant 
    
    
    # page 1 - Health Evolution   
    name_health=root_folder+'health_evolution.png'
    fig = plt.figure(figsize=(15.0, 20.0))
    colors=['r','orange','y','g','purple','b','grey']
    upper=[]
    lower=[]
    middle=[]
    x=[]
    for i in range(len(health[ma[0]])):
        x.append(i)
        sam=[]
        for m in ma:
            sam.append(health[m][i])
        upper.append(max(sam))
        lower.append(min(sam))
        middle.append(np.mean(sam))
    check_df=[]
    t=-1
    plt.plot(x,middle,c='royalblue',linewidth=2, alpha=0.7, label='Average health evolution')
    plt.fill_between(x,lower, upper, alpha=0.4,color='palegreen')
    plt.ylabel('Health % Indicator')
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    #plt.legend()
    plt.title('Last 24h Health Evolution')
    plt.savefig(name_health, dpi=300, bbox_inches='tight')
    body_health='The Monitoring system detected a significant health decrement and this notification report is created in response. This page presents the health indicator evolution the last 24h, smoothed from 30min analysis windows.'
    document.print_page([name_health],body_health,'TCU Health Warning')
    
    # page 2 - Feasible Fault Analysis 
    # https://scentellegher.github.io/visualization/2018/10/10/beautiful-bar-plots-matplotlib.html
    name_faults=root_folder+'feasible_faults.png'
    active_perc=[]
    for i in range(heard_faults.shape[0]):
        active_perc.append(np.round(sum(heard_faults[i])*100/len(heard_faults[i]),3))
    percentages = pd.Series(active_perc, index=faults)
    df = pd.DataFrame({'percentage' : percentages})
    df = df.sort_values(by='percentage')
    my_range=list(range(1,len(df.index)+1))
    fig, ax = plt.subplots(figsize=(10,7))
    plt.hlines(y=my_range, xmin=0, xmax=df['percentage'], color='#007ACC', alpha=0.2, linewidth=5)
    plt.plot(df['percentage'], my_range, "o", markersize=5, color='#007ACC', alpha=0.6)
    ax.set_xlabel('Percentage', fontsize=15, fontweight='black', color = '#333F4B')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.yticks(my_range, df.index)
    fig.text(-0.23, 0.96, 'Fault Type', fontsize=15, fontweight='black', color = '#333F4B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds((1, len(my_range)))
    ax.set_xlim(1)
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))
    plt.savefig(name_faults, dpi=300, bbox_inches='tight')
    body_faults='During the analyzed period the observed activations point out which faults could explain the issue. Usually more than one fault can be responsible but when agreggated the faults that are the better explanation are sorted in the following plot. Which percentage of the observed time where this faults a valid explanation is not reason enough to isolate one, but points out to the best candidates.'
    document.print_page([name_faults],body_faults,'Analysis of Best Fault Candidates')
    
    # page 3 - probability distribution
    name_probs = [root_folder+'new_probs.png',root_folder+'train_probs.png']
    fig = plt.figure(figsize=(20.0, 15.0))
    sns.heatmap(mtr_prbs,cmap='Spectral',annot=False,xticklabels=labels,yticklabels=y_labe,norm=LogNorm(),square=True, linewidth=0.1, linecolor='silver',cbar_kws = dict(use_gridspec=False,location="bottom"))
    plt.title('Probability distribution of the different variables in last 24h (%)')
    plt.savefig(name_probs[0], dpi=300, bbox_inches='tight')
    fig = plt.figure(figsize=(20.0, 15.0))
    sns.heatmap(train_prbs,cmap='Spectral',annot=False,xticklabels=labels,yticklabels=y_labe,norm=LogNorm(),square=True, linewidth=0.1, linecolor='silver',cbar_kws = dict(use_gridspec=False,location="bottom"))
    plt.title('Probability distribution of the different variables in training data (%)')
    plt.savefig(name_probs[1], dpi=300, bbox_inches='tight')
    body_probs='It is important to take into account in which condition was the machine working. The following plots show the probability of each variable taking the discretized values shown, comparing them with the distribution among the training data set.'
    document.print_page(name_probs,body_probs,'Probability Distribution Comparison')
    
    # page 4 - Preliminary Analysis of Model Results 1
    name_condactiv=[root_folder+'cond_prob_activ.png']
    names_msos=list(mtr_condactiv.keys())
    q=int(len(names_msos)/2)
    if len(names_msos)%2==1:
        q=q+1
    fig = plt.figure(figsize=(15.0, 15.0))
    i=0
    for name in names_msos:
        i=i+1
        ax1 = fig.add_subplot(q,2,i)
        sns.heatmap(mtr_condactiv[name],cmap='Spectral',annot=template_activ_set[name], fmt ='',xticklabels=labels,norm=LogNorm(),yticklabels=y_labe, linewidth=0.1,linecolor='silver')
        ax1.title.set_text(name)         
    fig.suptitle("Conditional Activations per Model")
    plt.savefig(name_condactiv, dpi=300, bbox_inches='tight')
    body_cond='One of the interpretations to the model results is to construct the conditional probability of having activation of a model when a variable takes a specific value. This information help us to understand which variables are behind the fault, being responsible for relations not accounted by the models. In each matrix the most relevant parts have been pointed out.'
    document.print_page(name_condactiv,body_cond,'Conditional Probability of Activations')
    
    # page 5 - Preliminary Analysis of Model Results 2
    name_mean=[root_folder+'mean_conf.png']
    names_msos=list(mtr_condactiv.keys())
    q=int(len(names_msos)/2)
    if len(names_msos)%2==1:
        q=q+1
    fig = plt.figure(figsize=(15.0, 15.0))
    i=0
    for name in names_msos:
        i=i+1
        ax1 = fig.add_subplot(q,2,i)
        sns.heatmap(mtr_mean_fault[name],cmap='Spectral',annot=template_mean_set[name], fmt ='',xticklabels=labels,norm=LogNorm(),yticklabels=y_labe, linewidth=0.1,linecolor='silver')
        ax1.title.set_text(name)         
    fig.suptitle("Mean Confidence per Model")
    plt.savefig(name_mean, dpi=300, bbox_inches='tight')
    body_mean='Complementary to the information obtained with conditional probabilities of activations, the mean value of each model confidence conditioned by each variable is useful information. The previous analysis might not be as sensitive as the following plots, where we can see more subtle correlations between the fault effect and the resulting estimations.'
    document.print_page(name_mean,body_mean,'Mean Confidence of Models')
    
    # page 6 - Sum up A : score by var and by MSO for 
    # https://scentellegher.github.io/visualization/2018/10/10/beautiful-bar-plots-matplotlib.html
    name_faults=root_folder+'feasible_faults.png'
    fig = plt.figure(figsize=(15.0, 10.0))
    ax_cond = fig.add_subplot(3,1,1)
    ax_mean = fig.add_subplot(3,1,2)
    ax_ratio = fig.add_subplot(3,1,3)
    colours=iter(CM.Spectral(np.linspace(0,1,(len(names_msos)))))
    my_range=[]
    for i in range(len(y_labe)):
        my_range.append(i)
    for name in names_msos:
        cond=[]
        mean=[]
        ratio=[]
        c=next(colours)
        i=-1
        for var in y_labe:
            i=i+1
            cond.append(sum_up_data[name]['cond_prob_mean'][i])
            mean.append(sum_up_data[name]['conf_mean'][i])
            ratio.append(sum_up_data[name]['ratio_score'][i])
        cond=np.nan_to_num(cond)
        mean=np.nan_to_num(mean)
        ratio=np.nan_to_num(ratio)
        ax_cond.plot(my_range, cond/max(cond), "o", markersize=9, color=c, alpha=0.7,label=name)
        ax_mean.plot(my_range, mean/max(mean), "o", markersize=9, color=c, alpha=0.7,label=name)
        ax_cond.plot(my_range, cond/max(cond), "-", linewidth=2, color=c, alpha=0.8)
        ax_mean.plot(my_range, mean/max(mean), "-", linewidth=2, color=c, alpha=0.8)
        ax_ratio.plot(my_range, ratio/max(ratio), "o", markersize=9, color=c, alpha=0.7,label=name)
        ax_ratio.plot(my_range, ratio/max(ratio), "-", linewidth=2, color=c, alpha=0.8)
    #ax_cond.hlines(y=my_range, xmin=0, color='#007ACC', alpha=0.2, linewidth=5)
    #ax_mean.hlines(y=my_range, xmin=0, color='#007ACC', alpha=0.2, linewidth=5)
    #ax_cond.set_xlabel('Variables', fontsize=15, fontweight='black', color = '#333F4B')
    #ax_mean.set_xlabel('Variables', fontsize=15, fontweight='black', color = '#333F4B')
    ax_cond.title.set_text('Score A: Corrected Conditional Probability.')
    ax_mean.title.set_text('Score B: Corrected Mean Confidence')
    ax_ratio.title.set_text('Score C: Ratio activations / probability')
    ax_cond.set_ylabel('Score A')
    ax_mean.set_ylabel('Score B')
    ax_ratio.set_ylabel('Score C')
    ax_mean.tick_params(axis='both', which='major', labelsize=12)
    ax_cond.tick_params(axis='both', which='major', labelsize=12)
    ax_ratio.tick_params(axis='both', which='major', labelsize=12)
    
    fig.text(-0.23, 0.96, 'Scores', fontsize=15, fontweight='black', color = '#333F4B')
    ax_cond.spines['top'].set_visible(False)
    ax_cond.spines['right'].set_visible(False)
    #ax_cond.spines['left'].set_bounds((1, len(my_range)))
    ax_cond.set_ylim(-0.05,1.05)
    ax_cond.spines['left'].set_position(('outward', 8))
    ax_cond.spines['bottom'].set_position(('outward', 5))
    ax_mean.spines['top'].set_visible(False)
    ax_mean.spines['right'].set_visible(False)
    #ax_mean.spines['left'].set_bounds((1, len(my_range)))
    ax_mean.set_ylim(-0.05,1.05)
    ax_mean.spines['left'].set_position(('outward', 8))
    ax_mean.spines['bottom'].set_position(('outward', 5))
    
    ax_ratio.spines['top'].set_visible(False)
    ax_ratio.spines['right'].set_visible(False)
    #ax_cond.spines['left'].set_bounds((1, len(my_range)))
    ax_ratio.set_ylim(-0.05,1.05)
    ax_ratio.spines['left'].set_position(('outward', 8))
    ax_ratio.spines['bottom'].set_position(('outward', 5))
    
    ax_mean.legend()
    ax_cond.legend()
    ax_ratio.legend()
    ax_mean.set_xticks(my_range) 
    ax_mean.set_xticklabels(y_labe)
    ax_cond.set_xticks(my_range) 
    ax_cond.set_xticklabels(y_labe)
    ax_ratio.set_xticks(my_range) 
    ax_ratio.set_xticklabels(y_labe)
    plt.tight_layout(h_pad=1.0)
    plt.savefig(name_faults, dpi=300, bbox_inches='tight')
    body_faults='The analysis performed digests all the model results and the statistical characteristics of the gathered telemetry. As a result a set of scores is created to evaluate the effect of the hinted fault on each variable and find the best ones. The following plots show the main scores used:'
    document.print_page([name_faults],body_faults,'Analysis of Best Fault Candidates')
    
    # Page 7 - resulting selection
    name_faults=root_folder+'final_scores.png'
    percentages = pd.Series(combined_result, index=y_labe)
    df = pd.DataFrame({'Final Score' : percentages})
    df = df.sort_values(by='Final Score')
    my_range=list(range(1,len(df.index)+1))
    fig, ax = plt.subplots(figsize=(10,7))
    plt.hlines(y=my_range, xmin=0, xmax=df['Final Score'], color='crimson', alpha=0.2, linewidth=5)
    plt.plot(df['Final Score'], my_range, "o", markersize=5, color='crimson', alpha=0.6)
    ax.set_xlabel('Final Score', fontsize=15, fontweight='black', color = '#333F4B')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.yticks(my_range, df.index)
    fig.text(-0.23, 0.96, 'Variables', fontsize=15, fontweight='black', color = '#333F4B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds((1, len(my_range)))
    ax.set_xlim(1)
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))
    plt.savefig(name_faults, dpi=300, bbox_inches='tight')
    body_faults='The models trained that have generated an interpretation for each fault is combined with the statistical characteristics of the TCU to identify the fault that triggered this report. The fault ranked in the previous page can be reasoned with the final ranking of each variable presented here.' 
    document.print_page([name_faults],body_faults,'Analysis of Best Fault Candidates')
    
    ################### SAVE / SEND ######################
    tit='HR- SN:'+str(device)+'- v: '+version+'- '+datetime.datetime.now().isoformat()
    report_file=root_folder+tit+'.pdf'
    document.output(report_file, 'F')
    
    send_email_pdf_figs(report_file, subject='Notification - TCU Health Warning', message='Test service to send reports', destination='sergio.galve@lauda-ultracool.com', password='Lauda2020')
    print(' --------- MAIL SENT WITH COMPLETE REPORT -----------')    