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
fault_mso_sensitivity={3: {'fs8': [0.49764028931080434, 0.48102854823222463, 0.4923052308237017, 0.45495006160326507, 0.4905338155578082], 'fs9': [0.41649340258661705, 0.4872421092350673, 0.47850878797647284, 0.3863071129380218, 0.3870651645300091], 'fs10': [0.08503468189906167, 2.9866026035109796e-05, 0.015799542811097324, 0.07197723783141544, 0.11809161537547316], 'fs1': [0.0008316262035168874, 0.03169947650667297, 0.013386438388728175, 0.08676558762729782, 0.00430940453670969]}, 15: {'fc4': [0.04691602815497721, 0.10694653170419087, 0.017358015475310698, 0.13317457899838414, 0.020418681664293708], 'fl1': [0.2221137023737432, 0.2534238915129473, 0.139313950719868, 0.11333887260731516, 0.22893877227407938], 'fo2': [0.04691602815497721, 0.09449837122239427, 0.004978158115511481, 0.0935428936534569, 0.0], 'fo3': [0.13652619130940366, 0.11978479574838614, 0.16795778151528937, 0.1754309571023818, 0.3315053689715648], 'fs2': [0.0, 0.012448160481796602, 0.01237985735979922, 0.03963168534492722, 0.020418681664293708], 'fs4': [0.01898564654341459, 0.0018816006343891614, 0.004644275718262972, 0.014904325202284418, 0.005536166765065367], 'fs5': [0.04691602815497721, 0.09449837122239427, 0.004978158115511481, 0.0935428936534569, 0.0], 'fs6': [0.13652619130940366, 0.11978479574838614, 0.16795778151528937, 0.1754309571023818, 0.3315053689715648], 'fc3': [0.12003657094603228, 0.12865156746158402, 0.017848031724287887, 0.02607420876424084, 0.04029908063045007], 'fs9': [0.08625785839469707, 0.04471442985354127, 0.20850047801594443, 0.01399444666884953, 0.0], 'fc5': [0.0, 0.0, 0.0, 0.0, 0.0], 'fs10': [0.09210305723375047, 0.01486112419884126, 0.20647112941021592, 0.09184464701144751, 0.021377879058688237], 'fs11': [0.023351348712311662, 0.004253180105574413, 0.023806191157354627, 0.014544766945436912, 0.0], 'fo5': [0.023351348712311662, 0.004253180105574413, 0.023806191157354627, 0.014544766945436912, 0.0]}, 17: {'fc5': [0.05408256972786417, 0.001766922642320281, 0.0002077837213560962, 0.04783009019780472, 0.05760444714203598], 'fs3': [0.05509649747971496, 0.020848183730095723, 0.023926089871292, 0.0597773105960531, 0.03836667551679725], 'fc2': [0.37605863668174927, 0.3406984638173229, 0.366004496828645, 0.4652704381191464, 0.5168751939655734], 'fl1': [0.0681256418836158, 0.06715042944730798, 0.06078543942210212, 0.02897601979421057, 0.03546480327048411], 'fo3': [0.06471522155027808, 0.1481826444832354, 0.12010486254856602, 0.10516307584745387, 0.1372189112617503], 'fs4': [0.007798481210043837, 0.0015517338069664193, 0.004778506611607546, 0.007546443195101713, 0.019088047025760207], 'fs6': [0.06471522155027808, 0.1481826444832354, 0.12010486254856602, 0.10516307584745387, 0.1372189112617503], 'fs8': [0.12842192632594668, 0.13166893234349192, 0.14370489734331432, 0.06465185164267631, 0.0], 'fs10': [0.12690323386264488, 0.13818312260370366, 0.16017527738319468, 0.0677916045622948, 0.0005585634138124936], 'fs12': [0.05408256972786417, 0.001766922642320281, 0.0002077837213560962, 0.04783009019780472, 0.05760444714203598]}, 234: {'fc1': [0.20925636996511218, 0.15047048309369052, 0.1758520516618093, 0.16422467773271188, 0.15943588719526802], 'fc2': [0.40527295358779813, 0.3111289300791982, 0.40841910940739207, 0.4923233178320143, 0.35663510577566254], 'fc4': [0.022239272685165255, 0.01329234388331517, 0.009282078071696502, 0.008739756872806918, 0.0058280641014017325], 'fo2': [0.022239272685165255, 0.01329234388331517, 0.009282078071696502, 0.008739756872806918, 0.0058280641014017325], 'fc3': [0.020140212797197094, 0.032559868771647404, 0.015369889073572474, 0.018540652546967632, 0.012442289863842507], 'fc5': [0.038152761813665205, 0.041502099276891184, 0.03591595401908943, 0.061182614057513685, 0.014524437487620731], 'fs8': [0.015149811889922435, 0.11077257040545606, 0.06446656850915654, 0.002301279747985309, 0.12431058803156202], 'fs10': [0.0, 0.10244890946425711, 0.05427437643111605, 0.0, 0.13459294889650958], 'fl1': [0.06772250497089197, 0.061652318963135874, 0.024705688791583566, 0.006794644197619165, 0.026438216474735394], 'fs3': [0.02464761082429604, 0.047718516524148484, 0.05463663954413007, 0.045845985454823567, 0.047241054750349315], 'fs4': [0.0007868802144537615, 0.007424329748769012, 0.010340697585557659, 0.018540652546967632, 0.005812796246789865], 'fs5': [0.022239272685165255, 0.01329234388331517, 0.009282078071696502, 0.008739756872806918, 0.0058280641014017325], 'fs6': [0.11400031406750226, 0.05294284274596937, 0.092256836742414, 0.10284429120746226, 0.08655804548583423], 'fs12': [0.038152761813665205, 0.041502099276891184, 0.03591595401908943, 0.061182614057513685, 0.014524437487620731]}, 362: {'fc1': [0.17613230568149724, 0.14856668140764454, 0.15829478818746578, 0.12136142369459665, 0.14393973333927665], 'fc2': [0.35717770024607987, 0.3060725465822585, 0.36752104780508443, 0.5821455114765325, 0.2893876252910322], 'fc4': [0.012297647455133995, 0.008029583323529165, 0.0059524742122997, 0.006809834391623532, 0.001981530984263734], 'fo2': [0.0, 0.0, 0.0, 0.0, 0.0], 'fc3': [0.006420121522666206, 0.013259160947875772, 0.0029593546902954133, 0.015189377569589145, 0.006534642770711531], 'fs9': [0.08001686900292988, 0.136011592773802, 0.11439536172507908, 0.026447920388557707, 0.15368643316577982], 'fs10': [0.07429084171221455, 0.020550705656323248, 0.0289468712369389, 0.06464645485874954, 0.15868776697553844], 'fl1': [0.05536078878028069, 0.057841705518834115, 0.03967621840790254, 0.020674470325951547, 0.022513404440266248], 'fo3': [0.1111120871753148, 0.0511988469595121, 0.07436215884938824, 0.033977980294237285, 0.09530789136340977], 'fs2': [0.012297647455133995, 0.008029583323529165, 0.0059524742122997, 0.006809834391623532, 0.001981530984263734], 'fs3': [0.0037819037934339727, 0.044755706553644914, 0.041263291408170734, 0.07508851625237344, 0.030671549322048405], 'fs6': [0.1111120871753148, 0.0511988469595121, 0.07436215884938824, 0.033977980294237285, 0.09530789136340977], 'fs8': [0.0, 0.15448503999353444, 0.08631380041568724, 0.01287069606192785, 0.0]}, 370: {'fc1': [0.09869694344606782, 0.13200232450184746, 0.09687443918678228, 0.08702398995118085, 0.1462626529534691], 'fc2': [0.36967520485588584, 0.19731340630560704, 0.33628673401975545, 0.41616175241452075, 0.4092978975649167], 'fc4': [0.04964122954541354, 0.054260380284974506, 0.06216920765790171, 0.03204836800333512, 0.051958736031623196], 'fo2': [0.040574013516777765, 0.04954974858810983, 0.048855644710552135, 0.03204836800333512, 0.04124435346826571], 'fc3': [0.10367160858398125, 0.12044479139331664, 0.04770099440323968, 0.02432517026563804, 0.017103822043254297], 'fs9': [0.08514048262796495, 0.08185203671040586, 0.1459758277747573, 0.13477665287032672, 0.04782351298869613], 'fs10': [0.025449379486618913, 0.11002099423852646, 0.07802529231590546, 0.16213435125618525, 0.013009255414567164], 'fl1': [0.11335805345791848, 0.18709391090434188, 0.07643981147641396, 0.06169624608274941, 0.0943039169218459], 'fo3': [0.0, 0.0, 0.0, 0.0, 0.0], 'fs1': [0.02478258590724076, 0.0021092020820482503, 0.03953642634108396, 0.0001321870186590106, 0.10993329453848404], 'fs2': [0.009067216028635772, 0.004710631696864686, 0.013313562947349578, 0.0, 0.010714382563357501], 'fs4': [0.03936926902671704, 0.011092824705847703, 0.005966414455706287, 0.017604546130734363, 0.017103822043254297], 'fs5': [0.040574013516777765, 0.04954974858810983, 0.048855644710552135, 0.03204836800333512, 0.04124435346826571]}}
v="_test_XIV_ZonTrain_210421"
d=["2020-11-06T18:00:00.000Z","2020-11-06T23:59:00.000Z"]
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
            if mso_set[mso]==234:
                ax.plot(sound[mso],color=color[mso+1],linewidth=2.5,alpha=0.8,label='Damped MSO #'+str(mso_set[mso]))
                ax.plot(activations[mso],color=c,linewidth=2.5,alpha=0.8,label='Raw Activations MSO #'+str(mso_set[mso]))
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
            
fig = plt.figure(figsize=(15.0, 20.0))
color=CM.rainbow(np.linspace(0,1,(len(faults))))
fault_names=list(faults.values())
re=3
for mso in range(len(mso_set)):
    c=color[mso]
    ax1 = fig.add_subplot(6,1,mso+1)
    bars=[]
    for f in fault_names:
        if f in fault_mso_sensitivity[mso_set[mso]]:
            bars.append(fault_mso_sensitivity[mso_set[mso]][f][re])
        else:
            bars.append(0)
    ax1.bar(fault_names,bars)
    ax1.title.set_text('MSO #'+str(mso_set[mso]))
fig.suptitle("MSO sensitivities")
plt.show()
        