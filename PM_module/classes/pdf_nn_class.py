# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:31:16 2020

@author: sega01

Creating a class to obtain the multivariate PDF
"""

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
import random 
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.core import Dense, Activation, Dropout
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time
import pickle

#################################################################################################################
class PDF_multivariable:
    def __init__(self):
        self.P=[]
        self.stats=[]
        self.bins=[]
        self.replace=[]
        self.max_edge=4
        
    def norm(self,x,train_stats):
        for name in x.columns:
            x[name]=x[name].apply(lambda num: (num-train_stats.loc[name,'mean'])/train_stats.loc[name,'std'])
        x=x.dropna()
        return x

    def init_discretize(self, bins, data):
        # do the discretization and find the edges of the bins that will be used --> over normed data
         edges=[-1000, -self.max_edge]
         replace={0:(-self.max_edge-2*self.max_edge/(bins-2))}
         for i in range(1,bins-1):
             edges.append(edges[i]+2*self.max_edge/(bins-2))
             replace[i]=(edges[i]+edges[i+1])/2
         replace[bins-1]=(self.max_edge+2*self.max_edge/(bins-2))
         edges.append(1000)
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
        normed_data = self.norm(data,self.stats)
        disc_data = self.discretize(normed_data)
        for row in disc_data.iterrows():
            ind=()
            for name in self.names:
                ind= ind+((row[1][name]),)
            prob.append(self.P[ind])
        return np.array(prob)
    
    def gen_conditionals(self,cond_var,data):
        # obtain the conditional probabilities for "name" given certain data --> variable xi conditional to all others
        # only for dim 2
        prob=[]
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
            prob.append(self.P[ind]/tot[cond])
        return np.array(prob)        
                
    

#################################################################################################################
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='') 

#################################################################################################################
# Class where a NN is trained to learn the probabilistic distribution of M variables 
class PDF_NN:
    def __init__(self, dims, acc, filename, bounds=[], confidence=0.95, std_dev=2):          
        self.model=[]                     # the NN used to model the PDF
        self.dims=dims                    # the number of dimensions considered   
        self.bounds=bounds                # the upper and lower limits for the random sampling 
        self.confidence=confidence        # the confidence taken in each sample
        self.std_dev=std_dev              # the amount of standard deviations within the upper and lower bounds given by the sensors - 4 means 95% confidence
        self.data=[]                      # the list of samples, in first column is the vector of the sample and in the last is the std deviation, 2nd and 3rd are upper and lower limits
        self.training_data=[]             # the original set of samples
        self.accuracy=acc                 # +- deviation in measure units for each variable: [[+-d1],[+-d2],...] - include dependence on FS ???
        self.total=[]                     # the total amount of samples multiplied by the confidence
        self.names=[]                     # reference to know the names of the variables modelled
        self.shape_nn=[]                  # necessary to load the NN after saving it
        self.epochs = 1000
        self.data_trained=[]
        self.cummulative={}               # the cummunlative probability for each dimmension, to use the cummnulative sampling
        self.probabilities={}
        self.df=[]                        # to store the samples obtained from MC approach
        self.filename=filename

    # Load the data and rearrange it to be in 4 columns 
    def load_data(self, training_data):        
        dic={'Vector':[],'Upper_Bound':[],'Lower_Bound':[],'std':[]}
        tot=0
        self.training_data=training_data
        self.names=training_data.columns
        for row in training_data.iterrows():
            vec=[]
            sup=[]
            inf=[]
            std=[]
            i=-1
            for name in self.names:
                i=i+1
                vec.append(row[1][name])
                sup.append(row[1][name]+self.accuracy[i])
                inf.append(row[1][name]-self.accuracy[i])
                std.append(self.accuracy[i]/self.std_dev)
            dic['Vector'].append(vec)
            dic['Upper_Bound'].append(sup)
            dic['Lower_Bound'].append(inf)
            dic['std'].append(std)
            tot=tot+self.confidence
        self.total=tot
        self.data=pd.DataFrame(dic)
    
    # for a given vector, get in which samples it falls
    def retrieve_matchings(self, vector):        
        filt=self.training_data
        for i in range(len(vector)):
            filt=filt[filt[self.names[i]]>(vector[i]-self.accuracy[i])]
            filt=filt[filt[self.names[i]]<(vector[i]+self.accuracy[i])]
        return filt
    
    # compute the accumulative probability for all the matching samples
    def get_total_prob(self,df,vector):
        p=0
        #print(('The sample: '+str(vector)))
        #print(('    Has '+str(df.shape[0])+' matching points'))
        for index, row in df.iterrows():
            # 3 vectors expected
            p=p+multivariate_normal.pdf(vector,row['Vector'],row['std'])
        #print(('    Has a probability of '+str(p)))
        return p/self.total
  
    # to complement get_pseudo_randoms we get from this function the positions betweeen the continuous value is placed
    def get_position(self,rand,dist):
        before=True
        ratio=[]
        if rand<dist[0]:
            low=0
            high=0    
            ratio=[1,0]
        elif rand>dist[len(dist)-1]: 
            low=1
            high=1 
            ratio=[0,1]
        else:
            for i in range(1,len(dist)):
                if before:
                    if rand<dist[i]:
                        before=False
                        low=dist[i-1]
                        high=dist[i]
            ratio.append((rand-low)/(high-low))
            ratio.append((high-rand)/(high-low))
        return ratio, low, high
                        
    # From the database used initially obtain the discrete prob distribution so that the cummulative can be obtained
    # A random value from 0 to 1 can be interpolated to get back a single possition included in the distribution 
    # the pseudorandom necessary for monte carlo ???
    def get_pseudo_randoms(self,name,samples):
        if name not in self.cummulative:
            Px=PDF_multivariable()
            Px.create_P(pd.DataFrame(self.training_data[name]))
            self.probabilities[name]=Px
            tot=0
            dic={} # this dictionary will allow to find the values of the variable given the cummulative prob
            #the limits are +inf and -inf, so we must avoid them  while doing this
            for i in range(len(Px.P)-1):
                tot=tot+Px.P[i]
                ind=(Px.edge_list[i+1]*Px.stats.loc[name,'std'])+Px.stats.loc[name,'mean']
                dic[tot]=ind
            self.cummulative[name]=dic
            
        # to generate # of random samples
        generated=[]
        for i in range(samples):
            try:
                rand=random.uniform(0, 1)
                ratio, low, high=self.get_position(rand,list(self.cummulative[name].keys()))
            except Exception as e:
                print('ERROR GENERATING NEW POINT')
                print(e)
                print(rand)

                ratio=[0,1]
                low=0
                high=0

            if low==0:
                generated.append(((-self.probabilities[name].max_edge-random.uniform(0, 1))*self.probabilities[name].stats.loc[name,'std'])+self.probabilities[name].stats.loc[name,'mean'])
            elif high==1:
                generated.append(((self.probabilities[name].max_edge+random.uniform(0, 1))*self.probabilities[name].stats.loc[name,'std'])+self.probabilities[name].stats.loc[name,'mean'])
            else:
                generated.append((ratio[0]*self.cummulative[name][low]+ratio[1]*self.cummulative[name][high]))
        return generated
        
    # create a random sampling within the bounds --> pseudorandom generation!!!! Dimensionality makes it impossible otherwise
    # use PDF_multivariable to get univariate PDFs and from there bias the distribution
    # use the given samples to randomize from "seeds"
    def create_training_set(self,n_samples):
        if self.bounds==[]:
            self.bounds={}
            for name in self.names:
                self.bounds[name]=[self.training_data[name].max(),self.training_data[name].min()] # should it be broader? does it even make sense?
        # type of sampling? can we ensure at least a grid sampling ? Now is just random independent samples
        randoms={}
        for name in self.names:
            randoms[name]=[]
        if self.df==[]: 
            self.df={'Vector':[],'Probability':[]}
        i=len(self.df['Vector'])
        k=0
        cons=0
        #debug={'get_pseudo_randoms':[],'get_matchings':[],'get_total_prob':[]}
        time0=time.time()
        tot_len=0
        while i<n_samples:
            v=[]
            k=k+1
            #t0=time.time()
            for name in self.names:
                r=self.get_pseudo_randoms(name,1)
                v.append(r[0])
            #t1=time.time()
            #debug['get_pseudo_randoms'].append(t1-t0)             
            #t2=time.time()
            fil=self.retrieve_matchings(v)
            #t3=time.time()
            #debug['get_matchings'].append(t3-t2)
            
            if len(fil)>0:
                cons=0
                tot_len=tot_len+len(fil)
                #filtered=self.data.loc[fil.index]
                #t4=time.time()
                p=self.get_total_prob(self.data,v) # It is already filtered
                #t5=time.time()
                #debug['get_total_prob'].append(t5-t4)
            else:
                cons=cons+1
                p=0
                #if cons>10:
                    #print('well see ...')
            if cons<2:
                i=i+1
                tot_len=tot_len+len(fil)
                if (i%500)==0:
                #if len(fil)>0:
                    print(' --> Already '+str(len(self.df['Vector']))+' samples')
                    time1=time.time()
                    print('       Time for the last 500 samples in s:'+str(time1-time0))
                    print('       The average # of hitted samples for 500 records is: '+str(tot_len/500))
                    tot_len=0
                    time0=time1
                    #print('       Number of hitted samples:'+str(len(fil)))
                    #print('       Time to retrieve sample in s:'+str(debug['get_total_prob'][(len(debug['get_total_prob'])-1)]))
                    file = open(self.filename, 'wb')
                    pickle.dump(self, file)
                    file.close()
                self.df['Vector'].append(v)
                self.df['Probability'].append(p)
            
        return pd.DataFrame(self.df)

    #####################################################
    # NN Model created with size proportional to the # of variables
    def build_model_nn(self,shape): # best result in models.txt
        size=self.dims
        if size>2:
            self.model = keras.Sequential([
                layers.Dense(round((25*size)), activation=tf.nn.relu, input_shape=[shape]),
                layers.Dropout(0.25),
                layers.Dense(round((20*size)), activation=tf.nn.relu),
                layers.Dropout(0.2),
                layers.Dense(round((15*size)), activation=tf.nn.relu),
                layers.Dropout(0.15),
                layers.Dense(round((15*size)), activation=tf.nn.relu),
                layers.Dropout(0.15),
                layers.Dense(round(10*size), activation=tf.nn.relu),  
                layers.Dropout(0.1),
                layers.Dense(round(10*size), activation=tf.nn.relu),  
                layers.Dropout(0.1),
                layers.Dense(round(10*size), activation=tf.nn.relu),  
                layers.Dropout(0.1),
                layers.Dense(round(5*size), activation=tf.nn.relu), 
                layers.Dropout(0.05),
                layers.Dense(round(5*size), activation=tf.nn.relu), 
                layers.Dropout(0.05),
                layers.Dense(round(5*size), activation=tf.nn.relu), 
                layers.Dropout(0.05),
                layers.Dense(round(4*size), activation=tf.nn.relu), 
                layers.Dense(round(4*size), activation=tf.nn.relu), 
                layers.Dense(round(3*size), activation=tf.nn.relu), 
                layers.Dense(round(2*size), activation=tf.nn.relu), 
                layers.Dense(round(size), activation=tf.nn.relu), 
                layers.Dense(round(size), activation=tf.nn.relu),  
                layers.Dense(round(size), activation=tf.nn.relu),  
                layers.Dense(round(size), activation=tf.nn.relu), 
                layers.Dense(1)])  
            optimizer = tf.keras.optimizers.Adam(0.005)
            self.model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mean_absolute_error', 'mean_squared_error'])     
         
        else:
            print('These simple cases ... Never work')
            self.model = keras.Sequential([
                layers.Dense(round((5)), activation=tf.nn.relu, input_shape=[shape]),
                layers.Dense(round(3), activation=tf.nn.relu),  
                layers.Dense(2, activation=tf.nn.relu),
                layers.Dense(1)])  
            optimizer = tf.keras.optimizers.Adam(0.002)
            self.model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mean_absolute_error', 'mean_squared_error'])  
    
    # function to normalize the data before training
    def norm(self,x,train_stats,prob=False):
        if prob:
            x=x.apply(lambda num: (num-train_stats.loc['mean'])/train_stats.loc['std'])
        else:
            for name in x.columns:
                x[name]=x[name].apply(lambda num: (num-train_stats.loc[name,'mean'])/train_stats.loc[name,'std'])
        return x 
    
    def dropna(self,test,train,series=False):
        nas=train.isna()
        delete=[]
        if series:
            for index, row in nas.iteritems():
                isna=False
                #print('The row problem is:')
                #print(type(row))
                #print(row)
                if row:
                    isna=True
                if isna:
                    delete.append(index)
        else:
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
        plt.show()
         
    # here we take create a nn and prepare the training set from        
    def train_nn(self,train_proportion=0.8):
        df=pd.DataFrame(self.df)
        d={}
        n=-1
        for name in self.names:
            n=n+1
            d[name]=[]
            for i in range(len(self.df['Vector'])):
                s=df['Vector'][i]
                d[name].append(s[n])
        df
        data=pd.DataFrame(d)
        train_dataset = data.sample(frac=train_proportion,random_state=np.random.RandomState(1234))
        drops=train_dataset.index
        test_dataset = data.drop(drops) 
        
        train_stats = data.describe()
        self.train_stats = train_stats.transpose()
        normed_train_data = self.norm(train_dataset,self.train_stats)
        normed_test_data = self.norm(test_dataset,self.train_stats)
        
        test_stats = df['Probability'].describe()
        self.test_stats = test_stats.transpose()
        train_labels= df['Probability'].loc[drops]
        test_labels= df['Probability'].drop(drops)
        normed_train_label = self.norm(train_labels,self.test_stats,prob=True)
        normed_test_label = self.norm(test_labels,self.test_stats,prob=True)
        
        self.shape_nn=len(train_dataset.keys())
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.build_model_nn(self.shape_nn)
        normed_train_label,normed_train_data=self.dropna(normed_train_label,normed_train_data)
        normed_train_data,normed_train_label=self.dropna(normed_train_data,normed_train_label,series=True)
        history = self.model.fit(normed_train_data, normed_train_label, epochs=self.epochs,validation_split = 0.15, verbose=0, callbacks=[early_stop, PrintDot()])
        self.plot_history(history)     
            
        test_predictions = self.model.predict(normed_test_data).flatten()
        error = test_predictions - normed_test_label
        
        x=[]
        for i in range(len(test_labels)):
            x.append(i)
            
        fig, ax = plt.subplots()
        ax.hist(error, bins = 25)
        plt.xlabel("Prediction Error")
        plt.title("Errors for PDF NN")
        plt.ylabel("Count")
        plt.show()      
        fig, ax = plt.subplots()
        ax.plot(x[:200],test_predictions[:200],'r',linewidth=0.35,label='Predicted Values')
        ax.plot(x[:200],normed_test_label[:200],'b',linewidth=0.35,label='Measured Values')
        plt.legend()
        plt.title("Prediction PDF NN")
        plt.xlabel('Samples')
        plt.ylabel('Probability')    
        plt.show() 
        return normed_test_label,test_predictions
        
    def predict(self,new_data,plot='Yes'):
        
        normed_predict_data = self.norm(new_data,self.train_stats)
        predictions = self.model.predict(normed_predict_data).flatten()

        if plot=='Yes':
            x=[]
            fig, ax = plt.subplots()
            ax.plot(x[:600],predictions[:600],'r',linewidth=0.35,label='Predicted Values')
            plt.title("Prediction PDF NN")
            plt.xlabel('Samples')
            plt.ylabel(self.objective)           
            plt.legend()
            plt.show()  
        return predictions
