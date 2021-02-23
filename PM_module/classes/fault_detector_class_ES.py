# -*- coding: utf-8 -*-
"""
Load the MSOs and create the predictors
"""
"https://www.tensorflow.org/tutorials/keras/basic_regression"



import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm as CM
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import ElasticNet
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA

import copy

from sklearn.model_selection import GridSearchCV
from scipy import linalg
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
#from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import datetime

import multiprocessing 
import logging
import warnings
import traceback
warnings.filterwarnings("error")


#################################################################################################################
#class to create objects that contain the PDF of several variables passed
class PDF_multivariable:
    def __init__(self):
        self.P=[]
        self.stats=[]
        self.bins=[]
        self.replace=[]
        self.max_edge=2
        
    def norm(self,x,train_stats):
         y={}
         for name in x.columns:
             y[name]=[]
             for i in x.index:
                 a=float((x.loc[i,name]-train_stats.loc[name,'mean'])/train_stats.loc[name,'std'])
                 y[name].append(a)  #.apply(lambda num: (num-train_stats.loc[name,'mean'])/train_stats.loc[name,'std'])
         return pd.DataFrame(y)

    def init_discretize(self, bins, data):
        # SOURCE OF ERROR: If norm data exceeds the upper and lower limits it will provide a crash
        # do the discretization and find the edges of the bins that will be used --> over normed data
         edges=[-1000000, -self.max_edge]
         replace={0:(-self.max_edge-2*self.max_edge/(bins-2))}
         for i in range(1,bins-1):
             edges.append(edges[i]+2*self.max_edge/(bins-2))
             replace[i]=(edges[i]+edges[i+1])/2
         replace[bins-1]=(self.max_edge+2*self.max_edge/(bins-2))
         edges.append(1000000)
         lab=pd.DataFrame()
         self.edge_list=edges
         for name in self.names:
             lab[name]=pd.cut(data[name].values, bins=edges, labels=False)
         #disc=lab.replace(replace)
         return lab,replace
     
    def discretize(self,data):
        # do the discretization using the already loaded values --> over normed data
        lab=pd.DataFrame()
        for name in self.names:
             lab[name]=pd.cut(data[name].values, bins=self.edge_list, labels=False)
         #disc=lab.replace(replace)
        return lab
        
    def create_P(self,data):
        # step 1: Normalize
        self.names=data.columns
        n=len(self.names)
        data=data.astype(float)
        train_stats = data.describe()
        self.stats = train_stats.transpose()
        normed_data = self.norm(data,self.stats)
        
        # step 2: Discretize
        self.bins=int(np.floor(10*5**3/n**3)) # how do we revisit this ...
        disc_data, self.replace=self.init_discretize(self.bins,normed_data)
        
        # step 3: build P of dimension n
        s=[]
        for i in range(n):
            s.append(self.bins)
        self.P=np.zeros(s)
        
        # step 4: count occurrences
        for row in disc_data.iterrows():
            ind=()
            for name in self.names:
                ind= ind+((row[1][name]),)
            self.P[ind]=self.P[ind]+1
        
        # step 5: create probabilities dividing by total number of samples
        self.P=self.P/disc_data.shape[0]
        self.original=self.P
        # step 6: apply smoothing kernel --> respecting that some values are non lineal and should not be smoothed
        #make difference between before and after and admit only the cells where diff is not bigger than a threshold tau
        tau=1/(0.5*self.bins**n)
        sm1=gaussian_filter(self.P, sigma=1)
        sm2=gaussian_filter(sm1, sigma=3)
        self.smooth=sm2
        diff=abs(self.P-sm2)
        hard=(diff>tau)*self.P
        soft=(diff<=tau)*sm2
        self.mask=[hard,soft]
        mix=hard+soft
        mix1=(mix==0)*sm2
        mix2=(mix!=0)*mix
        self.P=(mix1+mix2)/(mix1+mix2).sum()
        # step Alt: obtain summatories in each direcction for conditional probabilities --> high dimmensions is very expensive
        if n>1:
            sums=[]
            for i in range(n):
                sums.append(np.sum(self.P,axis=i))
            self.sum_cond=sums
        
    def gen_probabilities(self,data):
        # expecting a dataframe to iterate through
        prob=[]
        data=data.astype(float)
        normed_data = self.norm(data,self.stats)
        disc_data = self.discretize(normed_data)
        for row in disc_data.iterrows():
            ind=()
            for name in self.names:
                ind= ind+((row[1][name]),)
            try:
                prob.append(self.P[ind])
            except:
                print('  [!] Index not valid in Probabilities for Forecast: '+str(ind))
        return np.array(prob)
    
    def gen_conditionals(self,cond_var,data):
        # obtain the conditional probabilities for "name" given certain data --> variable xi conditional to all others
        # only for dim 2
        prob=[]
        data=data.astype(float)
        normed_data = self.norm(data,self.stats)
        disc_data = self.discretize(normed_data)
        for row in disc_data.iterrows():
            ind=()
            cond=()
            j=-1
            for name in self.names:
                j=j+1
                ind= ind+((row[1][name]),)
                if name!=cond_var:
                    # does this hold for higher dimensions, the order the variables are picked???
                    cond= cond+((row[1][name]),)
                else:
                    name_id=j
            tot=self.sum_cond[name_id]
            try:
                next_prob=self.P[ind]/tot[cond]
                prob.append(next_prob)
            except:
                print('Probability append failed for:')
                print('    Components P[ind] and tot[cond]: '+str(self.P[ind])+' ,'+str(tot[cond]))
                print('    Components [ind] and [cond]: '+str(ind)+' ,'+str(cond))
        return np.array(prob)        
                

#################################################################################################################
# Create one object per each MSO in order to load the models and parameters necesary to perform the preditions 
class residual:
     def __init__(self,mso_index,mso_reduced_index,sensed_vars,variables,faults,equations,sensor_names,cont_cond):
         
         self.mso_index=mso_index
         self.mso_reduced_index=mso_reduced_index
         self.known=sensed_vars
         self.variables=variables
         self.faults=faults
         self.equations=equations
         #predictor
         self.model=[]
         self.shape_nn=0
         self.epochs = 1000
         self.objective=[]
         self.source=[]
         self.train_stats=[]
         #uncertainty
         self.kde=[]
         self.kde_dims=["ControlRegCompAC.VarPowerKwMSK","ExtTemp"] #for Q_app: "WaterFlowMeter","W_InTempUser","W_OutTempUser",
         self.pca=[]
         self.pca_stats=[]
         self.kde_stats=[]
         self.untrained=True
         self.sensor_names=[]
         self.spec_list=[]  # list of all the model details to build it accordingly
         self.cont_cond=cont_cond
         for i in sensor_names:
             self.sensor_names.append(sensor_names[i])
         
     def save(self,folder,indx):
         name=folder+'MSO_Model_'+str(indx)
         name_att=name+'.pkl'
         variables=self.__dict__
         file = open(name_att, 'wb')
         pickle.dump(variables, file)
         file.close()
         print('  [*] Residual '+str(indx)+ ' Model --> SAVED')

     def load(self, folder,index):
         name=folder+'MSO_Model_'+str(index) #str(self.mso_index)
         name_att=name+'.pkl'
         filehandler = open(name_att, 'rb') 
         variables = pickle.load(filehandler)
         filehandler.close()
         for att in variables:
             setattr(self,att,variables[att])
         print('  [*] Residual '+str(index)+ ' Model --> LOADED')
         
     def norm(self,x,train_stats):
         y={}
         for name in x.columns:
             y[name]=[]
             for i in x.index:
                 a=float((x.loc[i,name]-train_stats.loc[name,'mean'])/train_stats.loc[name,'std'])
                 y[name].append(a)  #.apply(lambda num: (num-train_stats.loc[name,'mean'])/train_stats.loc[name,'std'])
         return pd.DataFrame(y)
     
     def dropna(self,test,train):
         nas=train.isna()
         delete=[]
         for index, row in nas.iterrows():
             isna=False
             for i in row:
                 if i:
                     isna=True
             if isna:
                 delete.append(index)
         test = test.drop(delete) 
         train = train.drop(delete) 
         return test,train
     
     def derivate(self,data):
         values=[]
         first=True
         for i in data:
             if first:
                 first=False
                 past=i
             else:
                 values.append((i-past))
                 past=i
         
         return values
     
     def kde_gathering(self,errors,test_dataset,option):
         
         try:
             test_dataset=test_dataset.drop('timestamp',axis=1)
         except:
             print(' ')  
         test_dataset = test_dataset.astype(float)
         if option!='PCA':
             test_labels = test_dataset.pop(self.objective)
             #Get variables for KDE
             uncertainty_dic={}
             for name in self.kde_dims:
                 if name == self.objective:
                     uncertainty_dic[name] = test_labels.values
                 elif (name not in self.source):
                     uncertainty_dic[name] = test_dataset.pop(name)
                 else:
                     uncertainty_dic[name] = test_dataset.loc[:,name]
             variables=pd.DataFrame(uncertainty_dic)
         
         errors=errors.reset_index()
         errors=errors.drop(['index'],axis=1)
         dic={'Errors':errors.values[:,0]}
         if option=='default':
             if len(self.kde_dims)>1:
                 #NOT IMPLEMENTED
                 plt_names=[self.kde_dims[0], self.kde_dims[1]]
             else:
                 plt_names=[self.kde_dims[0]]
                 #index=uncertainty_dic[self.kde_dims[0]].index
             for dim in self.kde_dims:
                 new=variables.pop(dim)
                 new=new.reset_index()
                 new=new.drop(['index'],axis=1)
                 dic[dim]=new.values[:,0]
            
             #dic['Errors'].reindex(index)
                 
         elif option=='Q_app':
             # NOT Fully IMPLEMENTED
             plt_names=[option, self.kde_dims[3]]
             q=variables[self.kde_dims[0]]*(variables[self.kde_dims[1]]-variables[self.kde_dims[2]])
             dic[option]=q.values[:,0]
             for i in range(3,len(self.kde_dims)):
                 dic[self.kde_dims[i]]=variables.pop(self.kde_dims[i])
                
         elif option=='derivative':
             plt_names=[self.kde_dims[0]]
             dic[self.kde_dims[0]]=self.derivate(variables.pop(self.kde_dims[0]).values)
             index=np.linspace(0,len(dic[self.kde_dims[0]])-1,len(dic[self.kde_dims[0]]))
             dic['Errors']=errors.values[1:,0]
             #dic['Errors'].reindex(index)
             
         elif option=='PCA':
             plt_names=['PCA']
             for name in test_dataset.columns:
                 if (name not in self.source) and (name!=self.objective):
                     test_dataset=test_dataset.drop([name],axis=1)
             #index=np.linspace(0,len(dic[self.kde_dims[0]])-1,len(dic[self.kde_dims[0]]))
             if self.untrained:   
                 self.pca_stats = test_dataset.describe()
                 self.pca_stats = self.pca_stats.transpose()
             x = self.norm(test_dataset,self.pca_stats)
             if self.untrained:
                 self.untrained=False
                 self.pca = PCA(n_components=1)
                 principalComponents = self.pca.fit_transform(x)
             else:
                 principalComponents = self.pca.transform(x)
             principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1'])
             new=principalDf.pop('principal component 1')
             new=new.reset_index()
             new=new.drop(['index'],axis=1)
             dic['PCA']=new.values[:,0]
                     
         #ind=np.linspace(0,len(dic['Errors'])-1,len(dic['Errors']))
         kde_data=pd.DataFrame(data=dic)
         kde_data.astype(float)
         return kde_data,plt_names
         
     
     def multivariate_data(self,dataset, target, start_index, end_index, history_size,target_size, step, single_step=False):
         data = []
         labels = []
         dataset=dataset.reset_index()
         dataset=dataset.drop(['index'],axis=1)
         target=target.reset_index()
         target=target.drop(['index'],axis=1)
         start_index = start_index + history_size
         if end_index is None:
           end_index = len(dataset) - target_size
        
         for i in range(start_index, end_index):
             indices = range(i-history_size, i, step)
             data.append(dataset.loc[indices].values)
        
             if single_step:
                 labels.append(target.loc[i+target_size].values)
             else:
                 labels.append(target.loc[i:i+target_size].values)
        
         return np.array(data), np.array(labels)
         
     def plot_history(self,history):
         hist = pd.DataFrame(history.history)
         hist['epoch'] = history.epoch
         plt.figure()
         plt.xlabel('Epoch')
         plt.ylabel('Mean Abs Error')
         plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')
         plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label='Val Error')
         plt.ylim([0,5])
         plt.legend()
         plt.figure()
         plt.xlabel('Epoch')
         plt.ylabel('Mean Square Error')
         plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
         plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
         plt.ylim([0,20])
         plt.legend()
         #plt.show()
         

###################################### Training ################################################
     # Function to launch the training of the model given a set of data and the variable to predict (aka objective as a string)
     #@ignore_warnings(category=ConvergenceWarning)
     def train(self,data,validation,kde_data,source,objective,cont_cond,predictor='NN',option2='PCA',acceptance=2):  
         self.cont_cond=cont_cond
         text1="The MSO #"+str(self.mso_reduced_index)+" takes the variable "+objective+" as output"
         print(text1)
         text_source=""
         for name in source:
             text_source=text_source+name+", "
         text2="The variables used  are: "+text_source
         print(text2)
         self.objective=objective
         self.source=source
         
         # Filter by mode if UnitStatus available --> Parametrize to configure a filtering variable
         #try:
             #data = data.loc[data['UnitStatus'] == 9.0]
         #except:
             #print('[!] No Filtering Parameter')
         # drop unused variables before starting 

         for name in list(data.columns):
             if (name not in source) and (name!=objective) and (name not in self.kde_dims) and (name not in self.cont_cond):
                 data=data.drop([name],axis=1)
                 validation=validation.drop([name],axis=1)
                 kde_data=kde_data.drop([name],axis=1)
 
         data = data.astype(float)
         validation=validation.astype(float)
         kde_data=kde_data.astype(float)
         original=copy.deepcopy(data)
         # Norm and prepare data
         train_stats = data.describe()
         #train_stats.pop(objective)
         train_stats = train_stats.transpose()
         self.train_stats=train_stats
         train_labels = data[objective]
         test_labels = validation[objective]
         kde_labels = kde_data[objective]
         normed_train_data = self.norm(data,train_stats)
         normed_test_data = self.norm(validation,train_stats)
         normed_kde = self.norm(kde_data,train_stats)
         
         print("The total amount of training samples is "+str(len(data)))
         print("The total amount of test samples is "+str(len(validation)))
         #data normalization to avoid bias of unit systems
         t_a=datetime.datetime.now()      
         #LIMITED GROUPS UNTIL ISSUES SOLVED WITH CONVERGENCE AND LOW WEIGHT GROUPS !!!!
       
         not_converged=True
         tole=0.001
         while not_converged:
             try:
                 grid = GridSearchCV(BayesianGaussianMixture(),
                            {'weight_concentration_prior': np.linspace(0.00001,1000, 30),'tol':[tole],'n_components':[7],'max_iter':[800],'init_params':['random']}) # 20-fold cross-validation ,n_jobs=3
                 print('In MSO '+str(self.mso_index)+' the cont cond ---> ')
                 print(self.cont_cond)
                 print(normed_kde)
                 grid.fit(normed_kde[self.cont_cond].values)
                
                 self.bgm = BayesianGaussianMixture(n_components=grid.best_params_['n_components'], random_state=42,max_iter=5000,weight_concentration_prior=grid.best_params_['weight_concentration_prior'],tol=tole,init_params='random',weight_concentration_prior_type='dirichlet_process').fit(normed_kde[self.cont_cond].values)
                 groups=self.bgm.predict(normed_train_data[self.cont_cond].values)
                 #probs=self.bgm.predict_proba(normed_train_data[self.cont_cond].values)
                 self.regions=np.unique(groups)
                 t_b=datetime.datetime.now()
                 dif=t_b-t_a
                 not_converged=False
             except ConvergenceWarning:
                     print('  - Convergence Warning training BGM in MSO '+str(self.mso_reduced_index)) 
                     print("Tolerance: "+str(tole))
                     tole=tole*5
         print('    [*] MSO'+str(self.mso_index)+' Clustering BGM ---> '+str(dif))
         ###################################################################
         print(self.bgm.weights_)
         test_groups=self.bgm.predict(normed_test_data[self.cont_cond].values)
        
         train_data_groups={}
         test_data_groups={}
         kde_data_groups={}
         for t in self.regions:
             locats=np.where(groups == t)[0]
             train_data_groups[t]=normed_train_data.iloc[locats]
             locats=np.where(test_groups == t)[0]
             test_data_groups[t]=normed_test_data.iloc[locats]

         # Now we prepare a model for each region
         new_model_set={}
         error=[]
         for t in self.regions: 
             new_model={}
             X_train = train_data_groups[t][source]
             y_train = train_data_groups[t][objective]
             X_test = test_data_groups[t][source]
             y_test = test_data_groups[t][objective]
             not_converged=True
             tole=0.0001
             while not_converged:
                 grid = GridSearchCV(ElasticNet(),
                            {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],'alpha':np.linspace(0,1,5)}) # 20-fold cross-validation ,n_jobs=3
                 print('In MSO '+str(self.mso_index)+' the cont cond ---> ')
                 print(self.cont_cond)
                 print(normed_kde)
                 grid.fit(X_train, y_train)
                 lin_reg_mod = ElasticNet(alpha=grid.best_params_['alpha'],l1_ratio=grid.best_params_['l1_ratio'],max_iter=5000,tol=tole)#,n_jobs=3
                 try:
                     lin_reg_mod.fit(X_train, y_train)
                     if t in test_groups:
                         pred = lin_reg_mod.predict(X_test)
                         test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
                         test_set_r2 = r2_score(y_test, pred)
                         if len(error)==0:
                             dif=y_test-pred
                             error=dif.values
                         else:
                             dif=y_test-pred
                             error=np.concatenate((error, dif.values), axis=0)
                     else:
                         pred=[]
                         test_set_rmse = 0
                         test_set_r2 = 0
                     print('  [I]  SPARSE COEFFICIENTS FOR EN')
                     print(lin_reg_mod.sparse_coef_)
                     new_model['model']=lin_reg_mod
                     new_model['cov']=lin_reg_mod.sparse_coef_
                     new_model['r2_score']=test_set_r2
                     new_model['rmse_score']=test_set_rmse
                     new_model['y_test']=y_test
                     new_model['pred']=pred
                     new_model['coeff']=lin_reg_mod.coef_
                     new_model['offset']=lin_reg_mod.intercept_
                     new_model_set[t]=new_model
                     not_converged=False
                 except ConvergenceWarning:
                     print('  - Convergence Warning training EN in MSO '+str(self.mso_reduced_index)+' in region #'+str(t)) 
                     print("Tolerance: "+str(tole))
                     tole=tole*5
         self.model=new_model_set
         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         kde_groups=self.bgm.predict(normed_kde[self.cont_cond].values)
         kde_feed={}
         # we prepare a dic of arrays with the errors pertinent to each model
         for t in self.regions:
             locats=np.where(kde_groups == t)[0]
             data_grouped=normed_kde.iloc[locats]
             kde_tr=data_grouped[source]
             kde_ts=data_grouped[objective]
             kde_fore=self.model[t]['model'].predict(kde_tr)
             kde_feed[t]={'error':kde_ts.values-kde_fore,'data':kde_tr}
         
         #print("Size of samples provided to KDE: "+str(len(uncertainty_vars)))
         print('  ')
         print('  [*] MEAN ERROR OBTAINED:'+str(np.mean(error)))
         print('  [*] STD OF OBJECTIVE OBTAINED:'+str(self.train_stats.loc[objective,'std']))
         print('  [*] STD OF ERROR OBTAINED:'+str(np.std(error)))
         if acceptance*np.std(error)>self.train_stats.loc[objective,'std']:
             not_done=True
             print('  [!] Model Rejected: Objective STD = '+str(self.train_stats.loc[objective,'std']))
         else:
             not_done=False
             print('  [V] Model Accepted')
             kde_data[kde_labels.name]=kde_labels
             self.model_uncertainty(kde_feed,option=option2)
         
         return_d={}
         return_d['error']=error
         return_d['groups']=test_groups # is it parallel to the errors? Sure? ...
         return_d['original']=original
         return_d['train_dataset_index']=data.index
         return_d['not_done']=not_done
         return return_d
         
     # alternatives to Uncertainty:
     #      https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
     # KDE implementations comparison
     #      https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
     def model_uncertainty(self,data,option='default'):
         self.hos={}
         self.lamdas={}
         self.kde_stats={}
         self.kde={}
         # NOW THE KDE IS OVER THE DATA WITHOUT NORMALIZATION --> STATS ARE STILL USEFUL
         for t in self.regions:
             
             ho=self.model[t]['cov'].values
             err=data[t]['error']
             tel=data[t]['data']
             kde_data=pd.DataFrame({'error':err})
             train_stats = kde_data.describe()
             self.kde_stats[t] = train_stats.transpose()
             trial=[]
             for i in range(len(err)):
                 val=tel.iloc[i].values
                 nor=np.linalg.norm(val.dot(ho),ord=1)
                 trial.append(abs(err[i])/nor)
             self.lamdas[t]=max(trial)
             self.hos[t]=ho
             
             # batch-stepped search for best H
             
             
             
             
             # KDE to compare... baseline
             kde_data=pd.DataFrame({'error':err})
             #print(kde_data)
             #is it ok to get the std from such small subsample?
             train_stats = kde_data.describe()
             self.kde_stats[t] = train_stats.transpose()
             #normed_data = self.norm(kde_data,self.kde_stats[t])
             #normed_data=normed_data.dropna()
             #kde_data.plot.kde()
             grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.01, 10, 20)},
                        cv=3) # 20-fold cross-validation
             grid.fit(kde_data.values)
             
             #print(grid.best_params_)
             # construct a kernel density estimate of the distribution
             self.kde[t] = KernelDensity(bandwidth=grid.best_params_['bandwidth'], metric='euclidean',kernel='gaussian', algorithm='ball_tree')
             self.kde[t].fit(kde_data.values)
     # to use it in many places ...
     def get_prediction_input(self,new_data):
         for name in list(new_data.columns):
             if (name not in self.source) and (name!=self.objective) and (name not in self.cont_cond):
                 new_data=new_data.drop([name],axis=1)
         # the prediction is over the normed GOAL
         normed_predict_data = self.norm(new_data,self.train_stats)
         return normed_predict_data
     
     def predict(self,new_data,plot='No'):
         # given a DF we clean it, norm it, classify it and make a prediction to each sample depending on its position                
         normed_predict_data=self.get_prediction_input(new_data)
         measured_value=normed_predict_data[self.objective]
         source_value=normed_predict_data[self.source]
         groups=self.bgm.predict(normed_predict_data[self.cont_cond].values)
         predictions=np.zeros(source_value.shape[0])
         probs=[]
         for i in range(source_value.shape[0]):
             probs.append(0)
         for t in self.regions:
             locats=np.where(groups == t)[0]
             selection=source_value.iloc[locats]
             if selection.shape[0]!=0:
                 predictions[locats]=self.model[t]['model'].predict(selection)
                 select_probs=self.model[t]['model'].predict_proba(selection)
                 print(select_probs)
                 for l in range(len(locats)):
                     probs[locats[l]]=select_probs[l]
         error = measured_value - predictions  
         print('Errors generated')
         print(error)

         return error,groups,probs
     # just the adjustment of x given the gaussian params (usually center on 0), x must be an array of np
     def gaussian(self,x,sig,mu=0):
         return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
     
     # given the set of samples we evaluate how liekly are those errors to happen -- For the zonotope approach
     def get_probs(self,telem,error_set,group):
         lim=self.lamdas[group]*np.linalg.norm(telem.dot(self.hos[group]),ord=1)
         probs=self.gaussian(error_set,lim/2)
         return probs
     
     
     # prepared to compute limits when the uncertainty is given in probabilistic form   
     def sample_score(self,variable_set,telem_set,group_set,m_y,sensor_acc,sample_n,plt_names,low,high,errors,alpha,phi,avoid,option,conf_factor=0.01,samples=[300,50]):                    
         label_num=-1
         for k in range(len(variable_set)):
             sample=variable_set[k]
             t=group_set[k]
             telem=telem_set.iloc[k].values
             label_num=label_num+1
             try:
                 #first get the whole area
                 new_cut=[]
                 new_measure=[]
                 new_max=[]
                 new_cut = np.linspace(m_y[t][0],m_y[t][1],samples[0]).reshape(-1,1)
                 if option=='Zonotope':
                     Z=self.get_probs(telem,new_cut,t)
                 else:
                     Z=np.exp(self.kde[t].score_samples(new_cut))
                 mp=new_cut[np.where(Z == max(Z))[0][0]]
                 max_point=mp
                 #print(max_point)
                 integral=sum(Z)
                 if integral>0:
                     P=Z/integral
                     #second get the sensor accuracy area
                     area=[(sample[0]-sensor_acc*self.kde_stats[t].loc["error","std"]),(sample[0]+sensor_acc*self.kde_stats[t].loc["error","std"])]
                     new_measure= np.linspace(area[0],area[1],samples[1]).reshape(-1,1)
                         
                     #using the point that got bigger probability to get the reference A_max for the confidence computation
                     area_max=[(max_point-sensor_acc*self.kde_stats[t].loc["error","std"]),(max_point+sensor_acc*self.kde_stats[t].loc["error","std"])]
                     new_max= np.linspace(area_max[0],area_max[1],samples[1]).reshape(-1,1)
                     #assume width 1 to simplify the integral of the full fampling and scale for the other
                     width_full=(m_y[t][1]-m_y[t][0])/samples[0]
                     width_measure=(area[1]-area[0])/samples[1]
                     if option=='Zonotope':
                         Z_m=self.get_probs(telem,new_measure,t)
                         Z_max=self.get_probs(telem,new_max,t)
                     else:
                         Z_m=np.exp(self.kde[t].score_samples(new_measure))
                         Z_max=np.exp(self.kde[t].score_samples(new_max))
                         
                     integral_measure=sum(Z_m)*width_measure/width_full
                     A_max=sum(Z_max)*width_measure/width_full
                     acc=0
                     #97% confidence
                     for j in range((len(P)-1)):
                         acc=acc+P[j]
                         if acc<conf_factor and (acc+P[j+1])>conf_factor:
                             nlow=new_cut[j][0]
                             low[str(label_num+sample_n)]=nlow
                         if acc<(1-conf_factor) and (acc+P[j+1])>(1-conf_factor):
                             nhigh=new_cut[j][0]
                             high[str(label_num+sample_n)]=nhigh
                     errors[str(label_num+sample_n)]=sample[0]
                     if ((sample[0]>nlow) and (sample[0]<nhigh)):
                         alpha[str(label_num+sample_n)]=(1-integral_measure/A_max)
                         phi[str(label_num+sample_n)]=0
                     else:
                         alpha[str(label_num+sample_n)]=(1-integral_measure/A_max)
                         phi[str(label_num+sample_n)]=(1)
             except RuntimeWarning:
                 avoid.append(str(label_num+sample_n))
                 print('  - Runtime Warning in Score!')
     
     # this splits the evaluation in sets where option defines the type of boundary to be used
     def score(self,variables,telemetry,groups,dic_fill,plt_names,times,option='default',std=5):        
         m_y={}
         for t in self.regions:
             m_y[t]=[(self.kde_stats[t].loc["error","mean"]-std*self.kde_stats[t].loc["error","std"]),(self.kde_stats[t].loc["error","mean"]+std*self.kde_stats[t].loc["error","std"])]
         manager = multiprocessing.Manager()
         low=manager.dict()
         high=manager.dict()
         errors=manager.dict()
         alpha=manager.dict()
         phi=manager.dict()
         avoid=manager.list()
         sensor_acc=0.10
         sample_n=-1
         jobs = []
         proc=int(multiprocessing.cpu_count())
         sample_start=0
         for i in range(proc):
             sample_n=sample_start
             sample_stop=int((i+1)*(len(variables))/proc)
             p = multiprocessing.Process(target=self.sample_score, args=(variables[sample_start:sample_stop].values,telemetry[sample_start:sample_stop],groups[sample_start:sample_stop],m_y,sensor_acc,sample_n,plt_names,low,high,errors,alpha,phi,avoid,option))
             p.start()
             jobs.append(p)
             sample_start=sample_stop
         for q in jobs:
             q.join()
  
         dic_fill['error']=[]
         dic_fill['high']=[]
         dic_fill['low']=[]
         dic_fill['phi']=[]
         dic_fill['alpha']=[]
         #print('Result of scores:')
         #print(errors)
         for i in range(len(variables)):
             if str(i) not in avoid:
                 do_it=True
                 try:
                     e=errors[str(i)]
                     h=high[str(i)]
                     l=low[str(i)]
                     p=phi[str(i)]
                     a=alpha[str(i)]
                 except:
                     do_it=False
                     print(str(i))
                     avoid.append(str(i))
                 if do_it:
                     dic_fill['error'].append(e)
                     dic_fill['high'].append(h)
                     dic_fill['low'].append(l)
                     dic_fill['phi'].append(p)
                     dic_fill['alpha'].append(a)
         forget=[]
         for i in avoid:
             forget.append(times.iloc[int(i)])
         return forget
     
     #when calculating confidences now 1/10 of the std is used as sensor range of uncertainty
     def evaluate_performance(self,data,option1='Zonotope',option2='PCA'): #option1 indicates which type of boundaries it will have ... default is the zonotope 

         #Clean, arrange and separate the data
         #try:
             #data = data.loc[data['UnitStatus'] == 9.0]
         #except:
             #print(' ')   
         data_2=copy.deepcopy(data)
         t_a=datetime.datetime.now()
         errors,groups,probs_bgm=self.predict(data_2)
         t_b=datetime.datetime.now()
         dif=t_b-t_a
         normed_predict_data=self.get_prediction_input(data_2)
         source_telemetry=normed_predict_data[self.source]
         print('    [*] MSO'+str(self.mso_index)+' prediction computing time ---> '+str(dif))
         # select the values for the scores

         #arrange the values for the kde evaluation
         #kde_data,plt_names=self.kde_gathering(errors,data_2,option2)

         kde_data=pd.DataFrame({'error':errors})
         plt_names=[]
         times=data_2['timestamp']
         #normed_kde=pd.DataFrame({'error':np.zeros(kde_data.shape[0])})
         """for t in self.regions:
             locats=np.where(groups == t)[0]
             selection=kde_data.iloc[locats]
             if selection.shape[0]!=0:
                 #print('[i] Region #'+str(t))
                 #print(self.kde_stats)
                 normed_kde.iloc[locats]=self.norm(selection,self.kde_stats[t])"""
         kde_arr=[]
         i=-1
         # get scores
         t_a=datetime.datetime.now()
         dic_fill={}
         forget=self.score(kde_data,source_telemetry,groups,dic_fill,plt_names,times,option=option1)
         t_b=datetime.datetime.now()
         dif=t_b-t_a
         print('    [*] MSO'+str(self.mso_index)+' scoring computing time ---> '+str(dif))
         dic_fill['group_prob']=probs_bgm
         return dic_fill,forget

     
         
#################################################################################################################


####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####   
####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####  ####       
class fault_detector:
     def __init__(self,filename,mso_txt,host,machine,matrix,sensors,faults,sensors_lookup,sensor_eqs,preferent,filt_value,filt_parameter,filt_delay_cap,main_ca,max_ca_jump,cont_cond,aggS=5):
         self.file_name=filename
         self.version=""
         self.msos_path=mso_txt                 # Path to the txt where the msos are defined indicating the variables numbers
         self.msos=[]

         self.faults=faults                     # dictionary that links those equations that represent a faultwith the fault names
         self.str_matrix=matrix                 # Structural Matrix relating variables (cols) and equations (rows) --> [[eq1],[eq2],[eq3]...]
         self.sensors=sensors                   # Sensors name by order according to the SM, using a dictionary to look for their names in the database
         self.time_bands = [["2019-12-12T05:45:00.000Z","2019-12-12T19:00:00.000Z"]] # timebands  of the selected data to train the samples -- ,["2019-12-10T05:45:00.000Z","2019-12-10T19:00:00.000Z"],["2019-12-09T05:45:00.000Z","2019-12-09T19:00:00.000Z"],["2019-10-29T05:45:00.000Z","2019-10-29T19:00:00.000Z"],["2019-10-28T05:45:00.000Z","2019-10-28T19:00:00.000Z"],["2019-10-21T05:45:00.000Z","2019-10-21T19:00:00.000Z"],["2019-10-10T05:45:00.000Z","2019-10-10T19:00:00.000Z"],["2019-10-08T05:45:00.000Z","2019-10-08T19:00:00.000Z"],["2019-10-02T05:45:00.000Z","2019-10-02T19:00:00.000Z"],["2019-10-03T05:45:00.000Z","2019-10-03T19:00:00.000Z"],["2019-10-01T05:45:00.000Z","2019-10-01T19:00:00.000Z"],["2019-09-30T05:45:00.000Z","2019-09-30T19:00:00.000Z"],["2019-11-08T05:45:00.000Z","2019-11-08T19:00:00.000Z"],["2019-11-04T05:45:00.000Z","2019-11-04T19:00:00.000Z"],["2019-11-05T05:45:00.000Z","2019-11-05T19:00:00.000Z"],["2019-09-18T05:25:20.000Z","2019-09-18T19:25:20.000Z"],["2019-09-23T05:25:20.000Z","2019-09-23T19:25:20.000Z"],["2020-01-09T05:45:00.000Z","2020-01-09T17:00:00.000Z"],["2020-01-10T05:45:00.000Z","2020-01-10T15:00:00.000Z"],["2020-01-13T05:45:00.000Z","2020-01-13T18:00:00.000Z"],["2020-01-14T05:45:00.000Z","2020-01-14T18:00:00.000Z"]
         self.training_data = []                # loaded values based on the timestamps given
         self.test_data = [] 
         self.kde_data = [] 
         self.residuals_id=[]                   # the variables known that will be used for the model training, defining each mso to look for repeated ones
         self.models=[]                         # the set of objects that contain the information and the trained models 
         self.mso_set = []                      # the mso_set that is found to identify all possible faults
         self.sensor_eqs=sensor_eqs             # the equations that represent the known variables
         self.sensors_lookup=sensors_lookup     # the loopup table to use as intemediary between sensor_eqs and sensors
         self.preferent=preferent               # the prefered goals to be predicted
         self.priori = self.get_priors_even()   # the initial prior probabilities     
         
         self.FSSM={}                           # the matrix to evaluate how likely is a fault to be triggered by each residual
         self.FSOM={}                           # a matrix representing the orders of activation acording to FSSM
         
         self.aggSeconds=aggS                   # it can be fed initially I supose
         self.filt_parameter=filt_parameter     # In the UC devices UnitStatus 
         self.filt_value=float(filt_value)      # In the UC devices 9.0
         self.filt_delay_cap=filt_delay_cap     # The maximum time to consider that a transition might be affecting the measures
         self.max_ca_jump=max_ca_jump           # The units are jump with (in measured database units) divided per second
         self.main_ca=main_ca                   # list with the name of the variables that are to be combined to identify transitions
         self.data_creation=datetime.datetime.now()
         self.cont_cond=cont_cond
         self.device=machine
         #self.ES_manager=elastic_manager(host,machine)                # Elastic search manager -->> DISABLED FOR THE DOCKER COMPOSE SERVICE
     def get_priors_even(self):
         priori = [] 
         for j in range(len(self.faults)):
             priori.append(1/len(self.faults))
         return priori
    
     def Save(self,folder,file):
        #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        #self.ES_manager.client=[]
        for i in self.mso_set:
            self.models[i].save(folder,i)
        variables=self.__dict__
        file = open(file, 'wb')
        pickle.dump(variables, file)
        file.close()
        #file = open(self.file_name, 'wb') 
        #pickle.dump(self, file)
        
        
     def Load(self,folder,file):
        #self.ES_manager.connect()
        filehandler = open(file, 'rb') 
        variables = pickle.load(filehandler)
        filehandler.close()
        for att in variables:
            if att!='models' and att!='residuals_id':
                setattr(self,att,variables[att])
        self.MSO_residuals()
        for i in self.mso_set:
            self.models[i].load(folder,i)
        
        
     def read_msos(self):      
        f = open(self.msos_path, "r")
        for x in f:
            a=x.split()
            l=[]
            for new in a[2:]:
                l.append(int(new))
            self.msos.append(l)
        f.close()
        
     def get_sensor_names(self,index):
        names=[]
        for i in index:
            names.append(self.sensors[i])
            
        return names
         
     # WITH DOCKER COMPOSE: Just meant to give the data required by the elastic client
     def get_data_names(self,option='training',times=[]):
        #data={}
        times_b=self.time_bands
        if option!='training':
            times_b=times
        
        names=['UnitStatus']  
        for i in self.sensors:
            names.append(self.sensors[i])
        return names, times_b
    
     def get_dic_entry(self,mso):
        entry='MSO_'+str(mso)
        return entry
###################################### Data Filtering ########################################
     # get the difference in secconds between two dates in format: "YYYY-MM-DDTHH:MM:SS.000Z"
     def diff_dates(self,date_start,date_end):
         start=datetime.datetime(year=int(date_start[0:4]), month=int(date_start[5:7]), day=int(date_start[8:10]), hour=int(date_start[11:13]),  minute=int(date_start[14:16]), second=int(date_start[17:19]), microsecond=1000)
         end=datetime.datetime(year=int(date_end[0:4]), month=int(date_end[5:7]), day=int(date_end[8:10]), hour=int(date_end[11:13]),  minute=int(date_end[14:16]), second=int(date_end[17:19]), microsecond=1000)
         diff=end-start
         return diff.total_seconds()
     
     # we must filter out the transition data after the machine was switched of
     def filter_samples(self,samples,filt_name=[],filt_value=[],filt_delay_cap=[],date_field='timestamp'):
        if filt_name==[]:
            filt_name=self.filt_parameter#"InvInfoCirc1.Info_MotPwr"
            filt_value=self.filt_value#0.0
            filt_delay_cap=self.filt_delay_cap/2
        reorder=samples.sort_values(by=[date_field],ascending=True)
        sorted_s=reorder.reset_index(drop=True)
        keep=sorted_s.loc[:,filt_name]==filt_value # IS THIS OK?
        keep[0]=False
        prev=True
        waiting=False
        date_1=sorted_s.loc[0,date_field]
        for i in range(len(keep)):
            if waiting:
                if self.diff_dates(date_1,sorted_s.loc[i,date_field])>filt_delay_cap:
                    waiting=False
                else:
                    keep[i]=False
            elif prev==False:
                if keep[i]:
                    date_1=sorted_s.loc[i,date_field]
                    keep[i]=False
                    waiting=True
            prev=keep[i]
        return sorted_s[keep]
    
    # LINKED TO PM_MANAGER CODE --> These filtering functions return a bool list that can be used easily by the pm_manager, being prepared ad hoc
    # we also want to locate the transitions in general so we will provide for a given sample (Dataframe) the bool list of values above limits
     def find_transitions(self,data,date_field='timestamp',ma_size=10,max_ca_jump=[],main_ca=[]):
         #filter by the filter parameters
        df=copy.deepcopy(data)
        df = df.loc[df[self.filt_parameter] == self.filt_value]
        # ma_size : THIS SHOULD BE CHANGED TO BE TIME DEPENDANT -- use Agg Seconds 
        if max_ca_jump==[]:
            max_ca_jump=self.max_ca_jump
            main_ca=self.main_ca
        if self.device!=71471:
            ma_size=int(ma_size/self.aggSeconds)
        else:
            ma_size=int(ma_size/0.5)
        if ma_size<2:
            ma_size=2
        # max_ca_jump : change of 0.3kw in 10s gap, change of 10l/min in 10s gap
        df['t_step']=pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        df['t_step']=df['t_step'].diff()
        df['t_step']=df.loc[:,'t_step'].apply(lambda x: x.total_seconds())
        main_steps=[]
        for n in main_ca:
            name=n+'_step'
            main_steps.append(name)
            df[name]=df[n].diff()
            #df.loc[df.index[0]][name]=0
        date=df.iloc[0][date_field]
        df=df.drop(df.index[0])
        data=data.drop(df.index[0])
        df_mvavg={}
        names_ma=[]
        for n in main_steps: 
            name=n+'_movavg'
            names_ma.append(name)
            df_mvavg[name]=[] 
        for i in range(df.shape[0]-ma_size):
            for n in range(len(main_steps)):
                j=i+ma_size
                df_mvavg[names_ma[n]].append((df.iloc[i:j][main_steps[n]].sum())/(df.iloc[i:j]['t_step'].sum()))
        # we fill the last 20 with the last mean obtained
        for i in range(ma_size):
            for name in names_ma:
                df_mvavg[name].append(df_mvavg[name][df.shape[0]-ma_size-1])
        for name in names_ma:
            df[name]=df_mvavg[name]
        triggers=[] # to compensate for the initial drop 
        active=False
        prev_date="1937-09-21T00:00:00.001Z"
        for j in range(df.shape[0]):
            i=-1
            flag=False
            if active and j>ma_size:
                if self.diff_dates(prev_date,df.iloc[j][date_field])<self.filt_delay_cap:
                    flag=True
                else:
                    active=False
                    flag=False
            for n in names_ma:
                i=i+1
                if abs(df.iloc[j][n])>max_ca_jump[i]:
                    flag=True
                    active=True
                    prev_date=df.iloc[j][date_field]
            triggers.append(flag)
        df['transition_trigger']=triggers
        return df,date
    
    # with the result of gradients and the model confidences this method discards all samples with errors that might be due to transitions
     def filter_transitions(self,data,activations,times,date_field='timestamp'):
         data,date=self.find_transitions(data)
         transition_filtering={date:False} # we discard the first element that in the function self.find_transitions is dropped
         j=-1
         for i in range(data.shape[0]):
             if not(times[times.isin([data.iloc[i][date_field]])].empty):
                 j=j+1
                 if data.iloc[i]['transition_trigger']:
                     activation=False
                     for mso in self.mso_set:
                         entry=self.get_dic_entry(mso)
                         if activations[entry][j]==1:
                             activation=True
                     transition_filtering[data.iloc[i][date_field]]=not(activation)
                 else:
                     transition_filtering[data.iloc[i][date_field]]=True
                 
         return transition_filtering
         
###################################### ML - TF ###############################################   
     # OBSOLETE --> just use get_data_names
     def get_data_frame(self):
        data, names = self.get_data_names()
        i=-1
        di={}
        for name in names:
            i=i+1
            di[name]=data[i]
            #get data in a dataframe structure from Pandas: https://www.geeksforgeeks.org/python-pandas-dataframe/
        self.training_data=pd.DataFrame(di) 
        
                
    
###################################### ANFIS ###############################################   
     def get_data_matrix(self,out_var,in_var):
         data, names = self.get_data_names(out_var,in_var)
         matrix= np.transpose(np.matrix(data))
         self.training_data=matrix
            
###################################### Get Residuals #######################################
                 
     def MSO_residuals(self):
         repeated=0
         n=0
         o=-1
         for mso in self.msos:
             n=n+1
             known=[] 
             variables=[]
             faults=[]
             equations=[]
             for eq in mso:
                 equations.append(eq)
                 if eq in self.faults:
                     if self.faults[eq] not in faults:
                         faults.append(self.faults[eq])
                 eq_vars=self.str_matrix[(eq-1)]
                 i=0
                 for new in eq_vars:
                     i=i+1
                     if (i not in variables) and (new==1):
                         variables.append(i)
                         if i in self.sensors:
                             if self.sensor_eqs[self.sensors_lookup[i]] in mso:
                                 known.append(i)
                             
             if known not in self.residuals_id:
                 o=o+1
                 self.residuals_id.append(known)
                 self.models.append(residual(n,o,known,variables,faults,equations,self.sensors,self.main_ca))
             else:
                 repeated=repeated+1
         #print(len(self.models))
            
     def fault_signature_matrix(self):
        self.fault_signature_matrix=[]
        n=-1
        for fault in self.faults:
            i=0
            n=n+1
            self.fault_signature_matrix.append([])
            for mso in self.mso_set:
                i=i+1
                #print("The mso selected is "+str(mso))
                
                if (self.faults[fault] in self.models[mso].faults):
                    self.fault_signature_matrix[n].append(1)
                else:
                    self.fault_signature_matrix[n].append(0)
                    
                    
     def train_mso(self,mso,predictor,option2,folder,return_dic,cont_cond):
         print('[D] Inside paralel process to train mso'+str(mso))
         not_done=True
         variables=self.models[mso].known
         names=self.get_sensor_names(variables)
         i=-1
         y=0
         for name in names:
             i=i+1
             if name in self.preferent:
                 y=i
         goal=names[y]
         names.remove(names[y])
         source=names
         while not_done:
             n=self.models[mso].mso_index
             o=self.models[mso].mso_reduced_index
             known=self.models[mso].known
             variables=self.models[mso].variables
             faults=self.models[mso].faults
             equations=self.models[mso].equations
             to_train=residual(n,o,known,variables,faults,equations,self.sensors,cont_cond)
             print('[D] Inside train mso function for mso #'+str(mso))
             return_dic[self.get_dic_entry(mso)]=to_train.train(self.training_data,self.test_data,self.kde_data,source,goal,cont_cond)
             to_train.save(folder,mso)
             #self.models[mso].load(folder,mso)
             self.models[mso]=to_train
             not_done=return_dic[self.get_dic_entry(mso)]['not_done']
             
     def train_residuals(self,folder,file,cont_cond,predictor='NN',outlayers='No',option2='PCA'):
         manager = multiprocessing.Manager()
         return_dic = manager.dict()
         jobs = []
         print('[D] Inside train residual function')
         for mso in self.mso_set: 
             print('[D] About to start mso'+str(mso))
             p = multiprocessing.Process(target=self.train_mso, args=(mso,predictor,option2,folder,return_dic,cont_cond))
             p.start()
             jobs.append(p)
         for proc in jobs:
             proc.join()
         for mso in self.mso_set:
             self.models[mso].load(folder,mso)
         self.Save(folder,file)
         return return_dic
            
        # This option would check all the dates to make sure that they do not provide outlayers or a faulty behaviour
        #if outlayers=='Yes':
            #self.check_outlayers()
          
     # OBSOLETE !!!!
     def check_outlayers(self):
         time_outlayers=[]
         i=-1
         for time in self.time_bands:
             i=i+1
             print(('Time Band #'+str(i)))
             result=[]
             data,names=self.get_data_names(option='PrimoVictoria',times=[time])
             #try:
                 #data = data.loc[data['UnitStatus'] == 9.0]
             #except:
                 #print('[!] No Filtering Parameter')
             data = data.drop(['timestamp'],axis=1)
             data = data.astype(float)
             for mso in self.mso_set:
                 error=self.models[mso].predict(data,plot='No')
                 diff=abs((self.models[mso].kde_stats.loc['Errors','mean']-error.mean())/self.models[mso].kde_stats.loc['Errors','std'])
                 print(('    MSO #'+str(mso)+ '--> Diff: '+str(diff)))
                 if diff>1:
                     result.append(True)
                 else:
                     result.append(False)
             time_outlayers.append(result)
         return time_outlayers
            
                
            
     def get_predictions(self,data):
        errors=[]
        for mso in self.mso_set:
            errors.append(self.models[mso].predict(data))
        return errors
    

     # Function meant to help the paralelizing of the analysis from pm_manager
     def evaluate_mso(self,data,mso,recover_dic,name=[],option='PCA'):
         if name==[]:
             recover_dic[self.get_dic_entry(mso)],q=self.models[mso].evaluate_performance(data,option1=option)
         else:
             recover_dic[name],q=self.models[mso].evaluate_performance(data,option1=option)
         return q  
     # pass dates as in the format of the training data: [[Start_1,End_1],[Start_2,End_2]...]
     #Work is now paralelized
     def evaluate_data(self,dates=[],manual_data=[],option='Zonotope'):
        if len(manual_data)<3:
            data, names = self.get_data_names(option='evaluation',times=dates)
        else:
            data=copy.deepcopy(manual_data)
        return_dic = {}
        forgets=[]
        for mso in self.mso_set: 
            tt=self.evaluate_mso(data,mso,return_dic,option=option)
            for t in tt:
                forgets.append(t)
        return return_dic,forgets
            
            
     def prior_update(self, activations, confidences, priori=[],alpha=[0.5],k_factor=0.25,option='GG'):#'SensitivityWeight'
         if priori==[]:
             priori=self.priori
         prior_evolution=[]
         for j in range(len(self.faults)):
             prior_evolution.append([priori[j]])
         test_weights=[]            
         for i in range(len(activations[0])):
             p_phi=[]
             fault=False
             for l in range(len(self.mso_set)):
                 if activations[l][i]==1:
                     fault=True
             #if an anomally is detected ...  
             if fault:
                 #first you get the probability of this marking for each fault
                 for j in range(len(self.faults)):
                     p_phi.append(0)
                     zvf=1
                     tot=0
                     for l in range(len(self.mso_set)):
                         if ((self.fault_signature_matrix[j][l]==0) and (activations[l][i]==1)):
                             zvf=0
                         if self.fault_signature_matrix[j][l]==1:
                             tot=tot+1
                             p_phi[j]=p_phi[j]+confidences[l][i]
                     if zvf==1:
                         p_phi[j]=p_phi[j]/tot 
                     elif tot>0:
                         p_phi[j]=p_phi[j]*0.5/tot 
                         #print([j,tot,p_phi[j]])
                 #then you compute the posterior probabilities (being the new prior probabilities)
                 try:
                     base=0
                     for j in range(len(self.faults)):
                         base=base+prior_evolution[j][i]*p_phi[j]
                     to_weight=[]
                     for j in range(len(self.faults)):
                         s=(p_phi[j]*prior_evolution[j][i])/base
                         to_weight.append(s)
                     # We pass the information to the function to evaluate the probabilities using the FSSM (and maybe FSOM)
                     to_activate=[]
                     for l in range(len(self.mso_set)):
                         to_activate.append(activations[l][i])
                     test_weights.append(to_weight)
                     if option=='SensitivityWeight':
                         prior=self.sensitivity_weight(to_weight,to_activate)
                         for j in range(len(self.faults)): 
                             prior_evolution[j].append(prior[j])
                     else:
                         s=[]
                         base=0
                         for j in range(len(self.faults)):
                             s.append((k_factor*to_weight[j]+(1-k_factor)*prior_evolution[j][i])/2)
                             base=base+s[j]
                         for j in range(len(self.faults)):     
                             prior_evolution[j].append(s[j]/base)
                 except:
                     #traceback.print_exc()
                     print('  [!] Error preparing sample for Prior Evolution')
                     fault=False
             # in case no anomally is detected get the prior probabilities close to the original
             if fault==False:
                 base=0
                 s=[]
                 for j in range(len(self.faults)):
                     s.append((k_factor*prior_evolution[j][i]+(1-k_factor)*prior_evolution[j][0])/2)
                     base=base+s[j]  
                 for j in range(len(self.faults)):     
                     prior_evolution[j].append(s[j]/base)
                     #print("Now This is a smooth reduction --> From: "+str(prior_evolution[j][len(prior_evolution[j])-2])+ "   To: " +str(s))
              
         return prior_evolution
                         
                  
     def plot_prior_evo(self,prior_evolution):
        s=-1
        end=np.floor(len(self.faults)/3)
        color=iter(CM.rainbow(np.linspace(0,1,(end-s))))
        fig, ax = plt.subplots()
        i=-1
        custom_lines=[]
        names=[]
        for fault in self.faults:
            i=i+1
            if i>s and i<=end:
                c=next(color)
                custom_lines.append(Line2D([0], [0], color=c, lw=4))
                names.append(self.faults[fault])
                ax.plot(prior_evolution[i],c=c,linewidth=0.25)
        plt.xlabel('Samples')
        plt.ylabel('Fault probabilities')
        plt.title("Fault Probability Evolution")
        plt.legend(custom_lines,names)
        #plt.show() 
        
        s=np.floor(len(self.faults)/3)
        end=np.floor(len(self.faults)*2/3)
        color=iter(CM.rainbow(np.linspace(0,1,(end-s))))
        fig, ax = plt.subplots()
        i=-1
        custom_lines=[]
        names=[]
        for fault in self.faults:
            i=i+1
            
            if i>s and i<=end:
                c=next(color)
                custom_lines.append(Line2D([0], [0], color=c, lw=4))
                names.append(self.faults[fault])
                ax.plot(prior_evolution[i],c=c,linewidth=0.25)
        plt.xlabel('Samples')
        plt.ylabel('Fault probabilities')
        plt.title("Fault Probability Evolution")
        plt.legend(custom_lines,names)
        #plt.show() 
        
        s=np.floor(len(self.faults)*2/3)
        end=len(self.faults)
        color=iter(CM.rainbow(np.linspace(0,1,(end-s))))
        fig, ax = plt.subplots()
        i=-1
        custom_lines=[]
        names=[]
        for fault in self.faults:
            i=i+1
            if i>s and i<=end:
                c=next(color)
                custom_lines.append(Line2D([0], [0], color=c, lw=4))
                names.append(self.faults[fault])
                ax.plot(prior_evolution[i],c=c,linewidth=0.25)
        plt.xlabel('Samples')
        plt.ylabel('Fault probabilities')
        plt.title("Fault Probability Evolution")
        plt.legend(custom_lines,names)
        #plt.show() 
        
        s=np.floor(len(self.faults)*2/3)
        end=len(self.faults)
        color=iter(CM.rainbow(np.linspace(0,1,(4))))
        fig, ax = plt.subplots()
        i=-1
        custom_lines=[]
        names=[]
        for fault in self.faults:
            i=i+1
            if (i==5) or (i==17) or (i==9) or (i==19):
                c=next(color)
                custom_lines.append(Line2D([0], [0], color=c, lw=4))
                names.append(self.faults[fault])
                ax.plot(prior_evolution[i],c=c,linewidth=0.45)
        plt.xlabel('Samples')
        plt.ylabel('Fault probabilities')
        plt.title("Fault Probability Evolution")
        plt.legend(custom_lines,names)
        #plt.show()
             
     # with the confidence obtained from each evaluation, a decision making must be made
     # in a first attempt, sensors accuracy will be (1 - 1/10 of the std)
     def identify_faults(self,data,priori=[],option='PCA'):
        activations=[]
        confidences=[]
        return_dic=self.evaluate_data(data)
        for mso in self.mso_set:
            activations.append(return_dic[self.get_dic_entry(mso)]['phi'])
            confidences.append(return_dic[self.get_dic_entry(mso)]['alpha'])
        prior_evolution=self.prior_update(activations, confidences)
        #self.plot_prior_evo(prior_evolution)
        
        return [prior_evolution,activations,confidences]
    
#################################################################################################################
#formula for Shannon Entropy from a vector of prob p
     def H_s(self,p,cond=[]):
         p_f = p[(p!=0)]
         
         if len(cond)==0:
             array=p_f*np.log2(p_f)
         else:
             cond_f = cond[(cond!=0)]
             array=p_f*np.log2(cond_f)
         filters = ~np.isnan(array)
         updated_array = array[filters]     
         return -sum(updated_array)
    
     def is_causal(self,entropy):
         direction=0
         if (entropy['Hx']>1.5*entropy['Hy']) and (1.5*entropy['Hx_p_y']<entropy['Hy_p_x']):
             direction=1
         if (entropy['Hy']>1.5*entropy['Hx']) and (1.5*entropy['Hy_p_x']<entropy['Hx_p_y']):
             direction=2
         return direction
     
     def check_pair_entropy(self,x,y,names,name_x,name_y):
         # Hypotheses --> a is CAUSE, b is Effect
         # provide dataset to train and test, and names 
         a=x
         b=y
         train_dataset, test = train_test_split(self.training_data, test_size=0.4)
         #try:
             #train_dataset, test = train_test_split(self.training_data.loc[self.training_data['UnitStatus'] == 9.0], test_size=0.4)
         #except:
             #train_dataset, test = train_test_split(self.training_data, test_size=0.4)
             #print('[!] No Filtering Parameter')
          #.loc[self.training_data['UnitStatus'] == '9.0']
         #print('Taking the UNITSTATUS==9:')
         #print(train_dataset.shape[0])
         name_a='PDF_'+name_x
         name_b='PDF_'+name_y
         if name_a in self.pdf_H:
             Px=self.pdf_H[name_a]
         else:
             Px=PDF_multivariable()
             Px.create_P(train_dataset[[names[a]]])
             self.pdf_H[name_a]=Px
        
         if name_b in self.pdf_H:
             Py=self.pdf_H[name_b]
         else:
             Py=PDF_multivariable()
             Py.create_P(train_dataset[[names[b]]])         
             self.pdf_H[name_b]=Py
         Pxy=PDF_multivariable()
         Pxy.create_P(train_dataset[[names[a],names[b]]])
         
         # entropy calculation according to shannon's
         pr_xy=Pxy.gen_probabilities(test[[names[a],names[b]]])
         Hxy=self.H_s(pr_xy)
         Hx_p_y=self.H_s(pr_xy,cond=Pxy.gen_conditionals(names[a],test[[names[a],names[b]]]))
         Hy_p_x=self.H_s(pr_xy,cond=Pxy.gen_conditionals(names[b],test[[names[a],names[b]]]))
         Hx=self.H_s(Px.gen_probabilities(test[[names[a]]]))
         Hy=self.H_s(Py.gen_probabilities(test[[names[b]]]))
        
         # CONDITIONAL ENTOPY: interpreted Hy_prior_x as the uncertainty of Y given a observation of X
         Hy_prior_x=Hxy-Hx
         Hx_prior_y=Hxy-Hy
         Ixy=Hx+Hy-Hxy
        
         # if Is_X_casue is bigger than 0 then we cant accept X as a cause --> K(Pc)+K(Pe_prior_c)<=K(Pe)+K(Pc_prior_e) pg 61 Scholkopf ... but since Pe_prior_c are both using the same values...
         Is_X_casue=(Hy_p_x+Hx)-(Hx_p_y+Hy)
         Is_Y_casue=-(Hy_p_x+Hx)+(Hx_p_y+Hy)
        
         causal_reasoning={'Hxy':Hxy,'Hx':Hx, 'Hy':Hy, 'Hx_p_y':Hx_p_y, 'Hy_p_x':Hy_p_x, 'Ixy':Ixy, 'Is_X_casue':Is_X_casue, 'Is_Y_casue':Is_Y_casue}
        
         return causal_reasoning
             
   
     # By using the data available from the training set we study each variable to see what  
     def load_entropy(self):
         edges=[]
         self.pdf_H={}
         self.entropy={}
         for eq in self.str_matrix:
             set_var=[]
             i=0
             for var in eq:
                 i=i+1
                 if var==1:
                    set_var.append(i)
             for n in range(len(set_var)-1):
                 for m in range(n+1,len(set_var)):
                     new=True
                     for edge in edges:
                         if (set_var[n] in edge) and (set_var[m] in edge):
                             new=False
                     if new:
                         if (set_var[n] in self.sensors) and (set_var[m] in self.sensors) and (self.sensors[set_var[m]]!=self.sensors[set_var[n]]):
                             entry=self.sensors_lookup[set_var[n]]+'-'+self.sensors_lookup[set_var[m]]
                             self.entropy[entry]=self.check_pair_entropy(set_var[n],set_var[m],self.sensors,self.sensors_lookup[set_var[n]],self.sensors_lookup[set_var[m]])
      
     # this method defines how the sensitivity is calculated acording to the    
     def detect_timeout(self,errors, high, low, max_noise):  
         #print('Detecting Time Out - SENSITIVITY')
         threshold=0.35 # at least 35% of the last "window" samples were erratic
         fcount=[]
         t=0
         finished=0
         window=np.ceil(2*len(errors)/(100*max_noise)) #2% error increase window
         sens=0.01 # to avoid 0 since it is supposed to be in the model, but would need stronger noise
         while t<len(errors) and t<len(low) and t<len(high) and finished!=1:
             if errors[t]>high[t] or errors[t]<low[t]:
                 fcount.append(1)
             else:
                 fcount.append(0)
             if (len(fcount)>window): 
                 from_n=int((len(fcount)-window))
                 to_n=int(len(fcount))
                 #print('Last ' + str(window) + ' samples had a count of errors: ' + str(sum(fcount[from_n:to_n])))
                 if ((sum(fcount[from_n:to_n])/window)>threshold):
                     sens=1-t/len(errors)
                     #print(sens)
                     finished=1
             t=t+1
         return sens

         
     # create a matrix based on the FSM where each position evaluates the sensitivity of each residual to each variable
     # this will be the base to build the FSSM                  
     def create_SM(self,samples=200,option='PCA'): 
         max_noise=0.4
         sensitivity={}
         for mso in self.mso_set:
             sensor_set={}
             for sens in self.sensors_lookup:
                 sensor_set[sens]=0
             sensitivity[mso]=sensor_set
         # DATA MUST HAVE BEEN FILTERED
         #try:
             #data=self.training_data.loc[self.training_data['UnitStatus'] == 9.0]
         #except:
             #data=self.training_data  
         data=self.training_data 
         fraction=samples/len(data)
         data=pd.DataFrame(data.sample(frac=fraction,random_state=np.random.RandomState(1234)))
         #manager = multiprocessing.Manager()
         return_dic = {}
         #jobs = []
         for mso in self.mso_set:
             for var in self.sensors:
                 ############# CAREFUL WITH THE DEFINITION OF THESE NAMES ##################
                 name=self.sensors[var]
                 # only do it if it is contained in the known variables of the mso
                 if var in self.models[mso].known:
                     d=copy.deepcopy(data)
                     noise=[]
                     for i in range(len(d)):
                         # ONLY POSSITIVE DEVIATIONS??
                         noise.append(np.random.normal(max_noise*self.models[mso].train_stats.loc[name,'mean']*i/len(data),self.models[mso].train_stats.loc[name,'std']/50))
                     d = d.drop(['timestamp'],axis=1)
                     d = d.astype(float)
                     d[name]=d[name]+noise
                     print('For the variable '+self.sensors_lookup[var]+ ' the sensitivity analysis is:')
                     name='MSO_'+str(mso)+'_var_'+str(var)
                     self.evaluate_mso(data,mso,return_dic,name=name)
                     #p = multiprocessing.Process(target=self.evaluate_mso, args=(data,mso,return_dic,name))
                     #p.start()
                     #jobs.append(p)
                     
                 else:
                     sensitivity[mso][var]=0
         #for proc in jobs:
             #proc.join()
         for mso in self.mso_set: 
             for var in self.sensors:
                 if var in self.models[mso].known:
                     name='MSO_'+str(mso)+'_var_'+str(var)
                     sensitivity[mso][var]=return_dic[name]
         return sensitivity   


     # based on the FSSM, the FSOM is built
     def create_FSSM(self,SM,option='PCA'): 
         #SM=self.create_SM(option)
         #self.load_entropy()
         
         for fau in self.faults:
             sensor_set={}
             for mso in self.mso_set:
                 sensor_set[mso]=0
             self.FSSM[fau]=sensor_set
             
         for fau in self.faults:
             eq=self.str_matrix[fau-1]
             i=0
             list_vars=[]
             for v in eq:
                 i=i+1
                 if v==1 and (i in self.sensors_lookup):
                     list_vars.append(i)
             # classify each fault to find the sensitivy needed
             # it is required to store all the variables interpreted as clear causes of the fault ... with their weighting
             if len(list_vars)==0:
                 print('The fault #'+str(fau)+' is linked to a variable not measured')
             else:   
                 fault_trigger={list_vars[0]:1}
                 if len(list_vars)!=1:
                     type_C=True
                     for v in list_vars:
                         fault_trigger[v]=0
                     for v1 in range(len(list_vars)-1):
                         for v2 in range(v1+1,len(list_vars)):
                             entry=(self.sensors_lookup[list_vars[v1]]+'-'+self.sensors_lookup[list_vars[v2]])
                             entry_b=(self.sensors_lookup[list_vars[v2]]+'-'+self.sensors_lookup[list_vars[v1]])
                             if entry in self.entropy:
                                 direction=self.is_causal(self.entropy[entry])
                                 if direction==1:
                                     type_C=False
                                     fault_trigger[list_vars[v1]]=fault_trigger[list_vars[v1]]+1
                                 if direction==2:
                                     type_C=False
                                     fault_trigger[list_vars[v2]]=fault_trigger[list_vars[v2]]+1
                             elif entry_b in self.entropy:
                                 direction=self.is_causal(self.entropy[entry_b])
                                 if direction==1:
                                     type_C=False
                                     fault_trigger[list_vars[v2]]=fault_trigger[list_vars[v2]]+1
                                 if direction==2:
                                     type_C=False
                                     fault_trigger[list_vars[v1]]=fault_trigger[list_vars[v1]]+1
                     # it has been checked if any variable presents a strong causal behaviour
                     if type_C:
                         print('FAULT '+str(fau)+': Type C')
                         # in case not, the mean value among all the variable sensitivities for each residual is used
                         # WEIGHTED Average NOT Implemented
                         
                         for mso in self.mso_set:
                             if self.faults[fau] in self.models[mso].faults:
                                 fi_list=[]
                                 for v in list_vars:
                                     fi_list.append(SM[mso][v])
                                 self.FSSM[fau][mso]=sum(fi_list)/len(fi_list) 
                             else:
                                 self.FSSM[fau][mso]=0
                     else:
                         print('FAULT '+str(fau)+': Type B')
                         # in case yes,  that variable sensitivity is taken as the actuator and it represents the faults
                         # WEIGHTED implemented ONLY with the number of causal links associated to it
                         causals=[]
                         total_causal=0
                         for v in list_vars:
                             # reevaluate this part ... other non causal variables can be linked aswell 
                             if fault_trigger[v]>0:
                                 causals.append(v)
                                 total_causal=total_causal+fault_trigger[v]
                         for mso in self.mso_set:
                             if self.faults[fau] in self.models[mso].faults:
                                 fi_list=[]
                                 for v in list_vars:
                                     if v in causals:
                                         fi_list.append(fault_trigger[v]*SM[mso][v]/total_causal)
                                     #else:
                                         
                                 self.FSSM[fau][mso]=sum(fi_list)/len(fi_list)  
                             else:
                                 self.FSSM[fau][mso]=0
                 else:
                     print('FAULT '+str(fau)+': Type A')
                     # case it is a sensor, only one variable is included and therefore 
                     for mso in self.mso_set:
                         self.FSSM[fau][mso]=SM[mso][list_vars[0]]
                     
             # normalize the sensitivity of each fault
             tot=0
             for mso in self.mso_set:
                 tot=tot+self.FSSM[fau][mso]
             for mso in self.mso_set:
                 if tot>0:
                     self.FSSM[fau][mso]=self.FSSM[fau][mso]/tot

    
     # based on the FSSM, the FSOM is built
     '''
     def create_FSOM(self): 
         for fau in self.faults:
             sensor_set={}
             for mso in self.mso_set:
                 sensor_set[mso]=0
             self.FSOM[fau]=sensor_set
             
         for fau in self.FSSM:
             first=True
             for sens in fau:
                 if first:
                     False
                     dic={1:[fau[sens]]}
                 else:
                     not_finished=True
                     i=0
                     while not_finished:
                         i=i+1
                         if fau[sens]>dic[i][0]:
                             
                         elif fau[sens]<dic[i][0]:
                             
                         else:'''
                         
     def sensitivity_weight(self,to_weight,to_activate): 
         # using the weights A and B
         A=1
         B=1
         #print('Given the activation vector:')
         #print(to_activate)
         # we check those residuals not activated
         onh=abs(np.array(to_activate)-1)
         i=-1
         ms=[]
         tms=0
         for f in self.faults:
             i=i+1
             if to_weight[i]>0:
                 j=-1
                 ms.append(0)
                 for mso_s in self.FSSM[f]:
                     j=j+1
                     if self.faults[f] in self.models[mso_s].faults:
                         ms[i]=ms[i]+onh[j]*self.FSSM[f][mso_s]
                         tms=tms+onh[j]*self.FSSM[f][mso_s]
                 #print('For '+self.faults[f]+' the MS is '+str(ms[i]))
             else: 
                 ms.append(0)
                 
         #print('And the TMS is: '+str(tms))
         Pf=[]
         total=0.001
         i=-1
         for n in ms:
             i=i+1
             A=2
             B=1
             
             pfi=to_weight[i]*tms*A/(1+ms[i]*B)
             Pf.append(pfi)
             total=total+pfi
         for i in range(len(Pf)):
             try:
                 Pf[i]=Pf[i]/total
             except:
                 print('Error in Sensitivity Weight normalization: ')   
                 print('     Pf[i] = '+str(Pf[i]))
                 print('     total = '+str(total))
                 Pf[i]=0
         return Pf
     
                         
     def forecast_Brown(self,X,k,w=4,N=50): 
         # Given a time series X, forecast of k steps ahead wanted and an evaluation window w (this must be even number)
         # Using Brown's double exponential smoothing in two steps:
         # STEP 1: alpha parameter selection --> testing N options
         s_1={}
         s_2={}
         F={}
         E={}
         alpha={}
         a={}
         b={}
         ids=[]
         for j in range(N-1):
             name="alpha_"+str((j+1)/N)
             alpha[name]=(j+1)/N
             ids.append(name)
             s_1j=[X[0]]
             s_2j=[X[0]]
             for i in range(1,len(X)):
                 s1t=alpha[name]*X[i]+(1-alpha[name])*s_1j[(i-1)]
                 s2t=alpha[name]*s1t+(1-alpha[name])*s_2j[(i-1)]
                 s_1j.append(s1t)
                 s_2j.append(s2t)
             s_1[name]=s_1j
             s_2[name]=s_2j
         
         for name in ids:
             a[name]=2*s_1[name][len(X)-k]-s_2[name][len(X)-k]
             b[name]=alpha[name]/(1-alpha[name])*(s_1[name][len(X)-k]-s_2[name][len(X)-k])
             ej=0
             F[name]=[]
             for i in range(k):
                 l=len(X)-k+i
                 next_f=a[name]+(i+1)*b[name]
                 F[name].append(next_f)
                 ej=ej+(X[l]-next_f)**2
             E[name]=math.sqrt(ej)
             
         Ew={}
         for i in range(N-1):
             if i<=(w/2):
                 s=(w/2)+1+i
                 em=0
                 for q in range(int(s)):
                     em=em+E[ids[q]]
                 Ew[ids[i]]=em/s
             elif (N-i-1)<=(w/2):
                 s=(w/2)+1+(N-i-1)
                 em=0
                 for q in range(int(s)):
                     em=em+E[ids[N-q-2]]
                 Ew[ids[i]]=em/s
             else:
                 em=0
                 for q in range(int(w+1)):
                     em=em+E[ids[i-2+q]]
                 Ew[ids[i]]=em/(w+1)
         
         #plt.bar(E.keys(), E.values(), color='b')
         #plt.show()
         #plt.bar(Ew.keys(), Ew.values(), color='g')
         #plt.show()
         
         # STEP 2: Generate a forecas with the best alpha
         min_e=100000
         for i in ids:
             if Ew[i]<min_e:
                 min_e=Ew[i]
                 selected=i
         
         f=[]
         x_f=[]
         for i in range(k):
             x_f.append(i+len(X)-1)
             f.append(a[selected]+(i)*b[selected])
             
         
         #x_original=range(len(X)) 
         #x_validation=range((len(X)-k),(len(X)))
         
         #plt.figure()
         #plt.xlabel('Epoch')
         #plt.ylabel('Error')
         #plt.title(selected)
         #plt.plot(x_f, f,label='Forecast')
         #plt.plot(x_original, X,label = 'Model Error')

         #plt.plot(x_validation,F[selected],label = 'Validation Forecast')
         #plt.legend()
         #plt.show()
         
         return Ew[selected],f,alpha[selected]
         
                         
     def boundary_forecast(self,error, high, low): 
         bound='no_activation'
         n=len(error)
         if (error[1]-error[0])>0:
             #Pb=PDF_multivariable()
             #hi=pd.DataFrame({'error_forecast':high})
             #Pb.create_P(hi)
             if error[len(error)-1]>np.mean(high):
                 bound='high'
                 n=0
                 test=True
                 top=np.mean(high)
                 while test and n<(len(error)-1):
                     n=n+1
                     if error[n]>top:
                         test=False
                 
         elif (error[1]-error[0])<0:
             #Pb=PDF_multivariable()
             #lo=pd.DataFrame({'error_forecast':low})
             #Pb.create_P(lo)
             if error[len(error)-1]<np.mean(low):
                 bound='low'
                 n=0
                 test=True
                 bot=np.mean(low)
                 while test and n<(len(error)-1):
                     n=n+1
                     if error[n]<bot:
                         test=False
         #er=pd.DataFrame({'error_forecast':error[n:]})
         #probs=Pb.gen_probabilities(er)
         return n, bound
         
         
                 
             
                 
         
                         
                 
                  
                        
                              

                
         
            