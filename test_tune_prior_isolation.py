# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 08:52:47 2021

@author: sega01
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm as CM
from matplotlib.lines import Line2D

from elasticsearch import Elasticsearch

def get_index_analytics(date,ty):
    index=ty+date[5:7]+date[0:4]
    return index 
def get_analytics(client,time_start, time_stop, device,version, names_analysis):
         ty='pm_bgm_data_' #'pm_bgm_data_'
         ind=get_index_analytics(time_start,ty) 
         response = client.search(
            index=ind,
            body={
                  "query": {
                    "bool": {
                      # Also: filter, must_not, should
                      "must": [ 
                        {
                          "match": {
                            "device": device
                          }
                        },
                        {
                          "match": {
                            "trained_version": version
                          }
                        },
                        {
                        "range": {
                        # Timestap format= "2019-12-30T09:25:20.000Z"
                        "timestamp": { 
                        "gt": time_start, # Date Format in Fault Manager: '2019-05-02 08:00:10'
                        "lt": time_stop 
                                     }
                                 }
                        }
                      ],
                      "must_not": [],
                      "should": []
                    }
                  },
                  "from": 0,
                  "size": 100,
                  "sort": [{ "timestamp" : {"order" : "asc"}}],
                  "aggs": {}
                },
            scroll='5m'
         )
         # DATA ARRANGE: each mso will have a dictionary with as many temporal series as in self.names_analysis --> all msos in the list data
         data=[]
         first=True
         for hit in response['hits']['hits']:
             #print(hit)
             if first:
                 first=False
                 n_msos=len(hit['_source'][names_analysis[0]])
                 for i in range(n_msos):
                     new_mso={}
                     for name in names_analysis:
                         field=hit['_source'][name]
                         if name=='timestamp':
                             new_mso[name]=[field]
                         else:
                             new_mso[name]=[field[i]]
                     data.append(new_mso)
             else: 
                 for i in range(n_msos):
                     for name in names_analysis:
                         field=hit['_source'][name]
                         if name=='timestamp':
                             data[i][name].append(field)
                         else:
                             data[i][name].append(field[i])

            
         sc_id=response['_scroll_id']
         more=True
         while more:
             sc=client.scroll(scroll_id=sc_id,scroll='2m') # ,scroll='1m'
             #sc_id=response['_scroll_id']
             if len(sc['hits']['hits'])==0: #|| total>20
                 more=False
             else:
                 for hit in sc['hits']['hits']:
                     if len(hit['_source'][names_analysis[0]])==n_msos:
                         for i in range(n_msos):
                             for name in names_analysis:
                                 field=hit['_source'][name]
                                 if name=='timestamp':
                                     data[i][name].append(field)
                                 else:
                                     data[i][name].append(field[i])
                     else:
                         print('  [!] WARNING: The gathered analysis data might come from different models, two sizes of MSO_SET: '+str(len(hit['_source'][names_analysis[0]]))+', '+str(n_msos)+'  | timestamp: '+hit['_source']['timestamp'])
                    
         return data
    
def fix_dict(d):
    new_d={}
    for n in d:
        new_d[int(n)]=d[n]   
    return new_d

device=71471
fault_mso_sensitivity={3: {'fs8': [0.47533388213769734, 0.48901432318795374, 0.49042977279299216, 0.4451960872456241, 0.4980184250560261], 'fs9': [0.4859063404137496, 0.484880852427425, 0.3871138606729532, 0.37334128414097034, 0.4288906527018617], 'fs10': [0.004999281376448872, 0.013714633008722853, 0.1181512839310764, 0.07678772408130648, 0.07230476828559457], 'fs1': [0.033760496072104254, 0.012390191375898323, 0.004305082602978141, 0.10467490453209916, 0.0007861539565175036]}, 15: {'fc4': [0.27823568774248725, 0.2795205684775241, 0.2537322469683078, 0.2854776217771798, 0.2711994165510718], 'fl1': [0.010017699125448535, 0.01129500605400957, 0.025345600711439896, 0.009387720822071961, 0.01672623757151993], 'fo2': [0.1962713120090863, 0.1915519080045958, 0.17741415912711667, 0.20558416019145498, 0.21541309395318003], 'fo3': [0.007896837200559684, 0.02224788974521167, 0.021775388211704872, 0.0024072872394135195, 0.0], 'fs2': [0.081964375733401, 0.08796866047292828, 0.07631808784119114, 0.07989346158572476, 0.0557863225978918], 'fs4': [0.0024890065186376673, 0.00025540222435028824, 0.017365103636375338, 0.0010343869184549486, 0.0012150295955886028], 'fs5': [0.1962713120090863, 0.1915519080045958, 0.17741415912711667, 0.20558416019145498, 0.21541309395318003], 'fs6': [0.007896837200559684, 0.02224788974521167, 0.021775388211704872, 0.0024072872394135195, 0.0], 'fc3': [0.008881330602809958, 0.0019410977646631751, 0.028497765556896374, 0.0013198692042716053, 0.017941267167108532], 'fs9': [0.09221668457352852, 0.08025883679824433, 0.11596369244253056, 0.06461934280967366, 0.19218121002328847], 'fc5': [0.0, 0.0, 0.0, 0.0, 0.0], 'fs10': [0.0942464734768533, 0.10555989260720576, 0.04582727389274091, 0.1309301955898508, 0.014124328587170712], 'fs11': [0.011806221903770854, 0.0028004700507297284, 0.019285567136437528, 0.005677253215517768, 0.0], 'fo5': [0.011806221903770854, 0.0028004700507297284, 0.019285567136437528, 0.005677253215517768, 0.0]}, 17: {'fc5': [0.056170938996273734, 0.01612149967879056, 0.0740936877636926, 0.04116666618877308, 0.05049185651264265], 'fs3': [0.06210802615699436, 0.05140201468165699, 0.04736446182654624, 0.06628206048447831, 0.04243737306336403], 'fc2': [0.36432790192207753, 0.3970174200974027, 0.5272359209167918, 0.4524615939625664, 0.44531832950990585], 'fl1': [0.05517653521014789, 0.030399296834218582, 0.01754621133443108, 0.032455789867516585, 0.06604249530574131], 'fo3': [0.05370821137556533, 0.09824408299610521, 0.11622411678960033, 0.10041791960095456, 0.11501610665012979], 'fs4': [0.007999546333328608, 0.006709334550150932, 0.01761033140066319, 0.008227584853607639, 0.0023969248202629417], 'fs6': [0.05370821137556533, 0.09824408299610521, 0.11622411678960033, 0.10041791960095456, 0.11501610665012979], 'fs8': [0.1459517144997313, 0.13968375996479404, 0.009607465414981906, 0.07859017612195, 0.05664115570688659], 'fs10': [0.1446779751340423, 0.14605700852198514, 0.0, 0.07881362313042596, 0.056147795268294394], 'fs12': [0.056170938996273734, 0.01612149967879056, 0.0740936877636926, 0.04116666618877308, 0.05049185651264265]}, 234: {'fc1': [0.15075329383113448, 0.1603467806452157, 0.1662829414588326, 0.17875695655839727, 0.20985819836647923], 'fc2': [0.31472421271504664, 0.3603228860603308, 0.4960625122304999, 0.4058575356871101, 0.41309733878375415], 'fc4': [0.012595578750335805, 0.006211890149714036, 0.00802347451442648, 0.009042935881190714, 0.022118353402915878], 'fo2': [0.012595578750335805, 0.006211890149714036, 0.00802347451442648, 0.009042935881190714, 0.022118353402915878], 'fc3': [0.03187890901127658, 0.011713476143143923, 0.017040735198164726, 0.015835707534098967, 0.014543500863422437], 'fc5': [0.042044439907757275, 0.015457238297600956, 0.059378955453283574, 0.034865584990491284, 0.039506091183359376], 'fs8': [0.10948888963225714, 0.12103117248681668, 0.0024852745200854972, 0.06339181612972566, 0.014850373583891976], 'fs10': [0.1012424546516878, 0.1311871709815035, 0.0, 0.053748278254807415, 0.0], 'fl1': [0.06202856881353827, 0.02686167453219724, 0.007894478772405577, 0.030767386961914507, 0.0588421134837604], 'fs3': [0.048098735365501855, 0.04789139312423365, 0.04632313892468226, 0.05431333281428197, 0.02547641167964896], 'fs4': [0.006874185325487942, 0.005325121664948505, 0.017040735198164726, 0.00943547526918656, 8.75760916546056e-05], 'fs5': [0.012595578750335805, 0.006211890149714036, 0.00802347451442648, 0.009042935881190714, 0.022118353402915878], 'fs6': [0.05303513458754719, 0.0857701773172662, 0.1040418492473183, 0.09103353316592253, 0.11787724457192181], 'fs12': [0.042044439907757275, 0.015457238297600956, 0.059378955453283574, 0.034865584990491284, 0.039506091183359376]}, 362: {'fc1': [0.17294505560636003, 0.14442889704731895, 0.16384826989296025, 0.14908358607165187, 0.14185296684761559], 'fc2': [0.36261072538394196, 0.2901712808367025, 0.37199171760756705, 0.311269804127937, 0.564684666098444], 'fc4': [0.012732971527497985, 0.002017232137440146, 0.0054423259339877956, 0.00772294587332633, 0.0055345764840929726], 'fo2': [0.0, 0.0, 0.0, 0.0, 0.0], 'fc3': [0.0015541107015372363, 0.0061921569562612905, 0.0019155931072151744, 0.013652776920550965, 0.020099439732547398], 'fs9': [0.08561759886911155, 0.15319902612415975, 0.10755971291130055, 0.13469297139556485, 0.012906168239579303], 'fs10': [0.07874845162722881, 0.15808648036901024, 0.02560988352629789, 0.017619662227501706, 0.03661569173977217], 'fl1': [0.045543596733664246, 0.02282996429871183, 0.042052998612478076, 0.05842350296387336, 0.020099439732547398], 'fo3': [0.11129191997642536, 0.09528387252552699, 0.07564684325338601, 0.05248031407397265, 0.056354084277785786], 'fs2': [0.012732971527497985, 0.002017232137440146, 0.0054423259339877956, 0.00772294587332633, 0.0055345764840929726], 'fs3': [0.004930678070309687, 0.03048998504190129, 0.04262169520032354, 0.04410960008103051, 0.07996430608573682], 'fs6': [0.11129191997642536, 0.09528387252552699, 0.07564684325338601, 0.05248031407397265, 0.056354084277785786], 'fs8': [0.0, 0.0, 0.08222179076710992, 0.15074157631729193, 0.0]}, 370: {'fc1': [0.10674624483053323, 0.08295490140821268, 0.057225993445734334, 0.15478275902369276, 0.09695993923996803], 'fc2': [0.35131315535240404, 0.3321123178895527, 0.41600622687169714, 0.30481403005371, 0.38631916808242245], 'fc4': [0.0628472665987237, 0.0524443586371764, 0.019235662802877977, 0.0894358106314101, 0.06101202019469033], 'fo2': [0.05075391594406489, 0.0295336188874794, 0.0023386988648921815, 0.0655406601927475, 0.0527701492606412], 'fc3': [0.10279334693827569, 0.03400713608534147, 0.016592970421044205, 0.008070267570129293, 0.021448667824356966], 'fs9': [0.08370019064221715, 0.1324747525309406, 0.20576134219332687, 0.10018859567221877, 0.10451683968252838], 'fs10': [0.012132250845398403, 0.1410837812787269, 0.20588862185311566, 0.054474748189326405, 0.15753434197206936], 'fl1': [0.11121587422423843, 0.05248078150577116, 0.04527610984559142, 0.06534694839228265, 0.03848739047175714], 'fo3': [0.0, 0.0, 0.0, 0.0, 0.0], 'fs1': [0.020174037079574138, 0.07842709578901604, 0.003131519680533304, 0.05984010207294296, 0.0010302666789982454], 'fs2': [0.01209335065465881, 0.02291073974969699, 0.016896963937985794, 0.023895150438662603, 0.00824187093404914], 'fs4': [0.03547645094584678, 0.012036897350606597, 0.009307191218309148, 0.008070267570129293, 0.018909196397877512], 'fs5': [0.05075391594406489, 0.0295336188874794, 0.0023386988648921815, 0.0655406601927475, 0.0527701492606412]}}
v="_test_XIII_NewSelec_150421"
d=["2020-11-10T16:30:00.000Z","2020-11-10T20:00:00.000Z"]
names_analysis=['models_error', 'low_bounds', 'high_bounds', 'activations', 'confidence','group_prob','timestamp']
host='52.169.220.43:9200'
client=Elasticsearch(hosts=[host])
data_issue=get_analytics(client,d[0],d[1],device,v,names_analysis)

mso_set=[3, 15, 17, 234, 362, 370]
faults=fix_dict({"1":"fc1","2":"fc2","3":"fc3","4":"fc4","6":"fc5","8":"fl1","10":"fo2","11":"fo3","12":"fo4","13":"fo5","19":"fs1","20":"fs2","21":"fs3","22":"fs4","23":"fs5","24":"fs6","25":"fs8","26":"fs9","27":"fs10","28":"fs11","29":"fs12"})
fault_signature_matrix=[[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1], [0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]]
priori=np.ones(len(faults))/len(faults)

if True:
    activations=[]
    confidences=[]
    error=[]
    high=[]
    low=[]
    for i in range(len(mso_set)):
        activations.append(data_issue[i]['activations'])
        confidences.append(data_issue[i]['confidence'])
        error.append(data_issue[i]['models_error'])
        high.append(data_issue[i]['high_bounds'])
        low.append(data_issue[i]['low_bounds'])
    times=data_issue[0]['timestamp']
    groups=data_issue[0]['group_prob']

sams=len(activations[0])
if False:   
    activations=[]
    start_1=int(sams/3)
    start_2=int(2*sams/3)
    for i in range(len(mso_set)):
        if i==5:
            victor_1=np.zeros(start_1)
            victor_2=np.random.randint(4, size=sams-start_1)
            victor_2=np.where(victor_2>1, 1, victor_2)
            activations.append(np.concatenate((victor_1,victor_2)))
        elif i==6:
            victor_1=np.zeros(start_2)
            victor_2=np.random.randint(2, size=sams-start_2)
            activations.append(np.concatenate((victor_1,victor_2)))
        else:
            activations.append(np.zeros(sams))
    
k_factor=0.95

up_lim_prior=0.33
option='GG'
ma=20
record=[]
priori=[]
for j in range(len(faults)):
    priori.append(1/len(faults))
    
    
# Key Strokes
sound=np.zeros((len(activations),len(activations[0])))
key_strokes=-1*np.ones(len(activations))
alpha=0.005
betha=alpha/18
gamma_y=0.5
for i in range(1,len(activations[0])):
    for j in range(len(activations)):
        if activations[j][i]==1:
            key_strokes[j]=i
            sound[j][i]=1
        else:
            #Double Decay 
            if 1-sound[j][i-1]<=gamma_y:
                sound[j][i]=sound[j][i-1]-alpha
            else:
                sound[j][i]=sound[j][i-1]-betha
                if sound[j][i]<0:
                    sound[j][i]=0
if True:
         if priori==[]:
             priori=priori
         prior_evolution=[]
         for j in range(len(faults)):
             prior_evolution.append([priori[j]])
         for i in range(ma):
             for j in range(len(faults)):
                 prior_evolution[j].append(priori[j])
         test_weights=[]  
         f_keys=list(faults.keys())  
         keystrokes=np.zeros(len(mso_set))
         
         for i in range(ma,len(activations[0])):
             saves={}
             ma_prior=[]
             for j in range(len(faults)):
                 new_ma=sum(prior_evolution[j][i-ma:i])/ma
                 if new_ma>up_lim_prior:
                     ma_prior.append(up_lim_prior)
                 else: 
                     ma_prior.append(new_ma)
             for j in range(len(faults)):
                 ma_prior[j]=ma_prior[j]/sum(ma_prior)
             saves['prior_ma']=ma_prior
             gr=groups[i].index(max(groups[i]))
             p_phi=[]
             fault=False
             activ_sample=[]
             for l in range(len(mso_set)):
                 activ_sample.append(activations[l][i])
                 if sound[l][i]>0:
                     fault=True
             #print(' Activations: '+str(activ))
             #if an anomally is detected ...  
             if fault:
                 #first you get the probability of this marking for each fault
                 fault_phi={}
                 for j in range(len(faults)):
                     p_phi.append(0)
                     zvf=1
                     tot=0
                     contr=[]
                     for l in range(len(mso_set)):
                         contr.append(0)
                         if ((fault_signature_matrix[j][l]==0) and (sound[l][i]>gamma_y)):
                             zvf=0
                         if fault_signature_matrix[j][l]==1:
                             tot=tot+1
                             contr[l]=confidences[l][i]*abs(fault_mso_sensitivity[mso_set[l]][faults[f_keys[j]]][gr]/2)
                             #print(fault_mso_sensitivity[mso_set[l]][faults[f_keys[j]]][gr])
                             p_phi[j]=p_phi[j]+confidences[l][i]*abs(fault_mso_sensitivity[mso_set[l]][faults[f_keys[j]]][gr]/2)
                     
                     fault_phi[faults[f_keys[j]]]=contr
                     if sum(fault_signature_matrix[j])>0:
                         if zvf==1:
                             p_phi[j]=p_phi[j]/tot 
                         elif tot>0:
                             p_phi[j]=p_phi[j]*0.5/tot 
                         #print([j,tot,p_phi[j]])
                 #then you compute the posterior probabilities (being the new prior probabilities)
                 saves['mso_contrib']=fault_phi
                 record.append(saves)
                 try:
                     base=0
                     for j in range(len(faults)):
                         # we test to work without bayesian convergence
                         base=base+ma_prior[j]*p_phi[j]
                     to_weight=[]
                     for j in range(len(faults)):
                         s=(p_phi[j]*ma_prior[j])/base
                         to_weight.append(s)
                     # We pass the information to the function to evaluate the probabilities using the FSSM (and maybe FSOM)
                     to_activate=[]
                     for l in range(len(mso_set)):
                         to_activate.append(activations[l][i])
                     test_weights.append(to_weight)
                     if option=='SensitivityWeight':
                         print('Miracle')
                     else:
                         s=[]
                         base=0
                         for j in range(len(faults)):
                             s.append(((1-k_factor)*to_weight[j]+k_factor*ma_prior[j])/2)#prior_evolution[j][i]
                             base=base+s[j]
                         for j in range(len(faults)):     
                             prior_evolution[j].append(s[j]/base)
                 except:
                     #traceback.print_exc()
                     print('  [!] Error preparing sample for Prior Evolution')
                     fault=False
             # in case no anomally is detected get the prior probabilities close to the original
             if fault==False:
                 base=0
                 s=[]
                 for j in range(len(faults)):
                     s.append((k_factor*prior_evolution[j][i]+(1-k_factor)*prior_evolution[j][0])/2)
                     base=base+s[j]  
                 for j in range(len(faults)):     
                     prior_evolution[j].append(s[j]/base)

for j in range(len(faults)):
    prior_evolution[j]=prior_evolution[j][ma:]
if True:
        s=-1
        end=np.floor(len(faults)/3)
        color=iter(CM.rainbow(np.linspace(0,1,(int(end)-int(s)))))
        fig, ax = plt.subplots()
        i=-1
        custom_lines=[]
        names=[]
        for fault in faults:
            i=i+1
            if i>s and i<=end:
                c=next(color)
                custom_lines.append(Line2D([0], [0], color=c, lw=4))
                names.append(faults[fault])
                ax.plot(prior_evolution[i],c=c,linewidth=1.9,alpha=0.6)
        plt.xlabel('Samples')
        plt.ylabel('Fault probabilities')
        plt.title("Fault Probability Evolution")
        plt.legend(custom_lines,names)
        plt.show() 
        
        s=np.floor(len(faults)/3)
        end=np.floor(len(faults)*2/3)
        color=iter(CM.rainbow(np.linspace(0,1,(int(end)-int(s)))))
        fig, ax = plt.subplots()
        i=-1
        custom_lines=[]
        names=[]
        for fault in faults:
            i=i+1
            
            if i>s and i<=end:
                c=next(color)
                custom_lines.append(Line2D([0], [0], color=c, lw=4))
                names.append(faults[fault])
                ax.plot(prior_evolution[i],c=c,linewidth=1.9,alpha=0.6)
        plt.xlabel('Samples')
        plt.ylabel('Fault probabilities')
        plt.title("Fault Probability Evolution")
        plt.legend(custom_lines,names)
        plt.show() 
        
        s=np.floor(len(faults)*2/3)
        end=len(faults)
        color=iter(CM.rainbow(np.linspace(0,1,(int(end)-int(s)))))
        fig, ax = plt.subplots()
        i=-1
        custom_lines=[]
        names=[]
        for fault in faults:
            i=i+1
            if i>s and i<=end:
                c=next(color)
                custom_lines.append(Line2D([0], [0], color=c, lw=4))
                names.append(faults[fault])
                ax.plot(prior_evolution[i],c=c,linewidth=1.9,alpha=0.6)
        plt.xlabel('Samples')
        plt.ylabel('Fault probabilities')
        plt.title("Fault Probability Evolution")
        plt.legend(custom_lines,names)
        plt.show() 
        
        # only activations
        color=CM.rainbow(np.linspace(0,1,(len(mso_set))))
        fig, ax = plt.subplots()
        custom_lines=[]
        names=[]
        for mso in range(len(mso_set)):
            c=color[mso]
            #custom_lines.append(Line2D([0], [0], color=c, lw=4))
            names.append('MSO #'+str(mso_set[mso]))
            ax.plot(activations[mso],color=c,linewidth=2.5,alpha=0.8,label='MSO #'+str(mso_set[mso]))
        plt.xlabel('Samples')
        plt.ylabel('Activation Pressure')
        plt.title("MSO Activations")
        plt.legend()#custom_lines,names
        plt.show() 
        
        #only sound from activation strokes
        color=CM.rainbow(np.linspace(0,1,(len(mso_set))))
        fig, ax = plt.subplots()
        custom_lines=[]
        names=[]
        for mso in range(len(mso_set)):
            c=color[mso]
            #custom_lines.append(Line2D([0], [0], color=c, lw=4))
            names.append('MSO #'+str(mso_set[mso]))
            ax.plot(sound[mso],color=c,linewidth=2.5,alpha=0.8,label='MSO #'+str(mso_set[mso]))
        plt.xlabel('Samples')
        plt.ylabel('Activation Pressure')
        plt.title("MSO Activations")
        plt.legend()#custom_lines,names
        plt.show() 
        
        # mix of activ, 
        color=['grey','orange','r','r','b']
        custom_lines=[]
        names=[]
        for mso in range(len(mso_set)):
            fig = plt.figure(figsize=(15.0, 15.0))
            #custom_lines.append(Line2D([0], [0], color=c, lw=4))
            names.append('MSO #'+str(mso_set[mso]))
            ax1 = fig.add_subplot(2,1,1)
            ax1.plot(np.array(activations[mso])/10,color=color[0],linewidth=2.5,alpha=0.8,label='Activations')
            ax1.plot(confidences[mso],color=color[1],linewidth=2.5,alpha=0.8,label='Confidences')
            ax2 = fig.add_subplot(2,1,2)
            ax2.plot(high[mso],color=color[2],linewidth=2.5,alpha=0.8,label='High Bounds')
            ax2.plot(low[mso],color=color[3],linewidth=2.5,alpha=0.8,label='Low Bounds')
            ax2.plot(error[mso],color=color[4],linewidth=2.5,alpha=0.8,label='Error')
            plt.xlabel('Samples')
            #plt.ylabel('Activation Pressure')
            fig.suptitle('MSO #'+str(mso_set[mso]))
            plt.legend()#custom_lines,names
            plt.show() 
            
            

        