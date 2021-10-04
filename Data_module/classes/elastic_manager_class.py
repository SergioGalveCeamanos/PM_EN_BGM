# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:25:47 2020

@author: sega01
"""
from elasticsearch import Elasticsearch, helpers
import pandas as pd
import json
import traceback

class elastic_manager:
     def __init__(self, host, device,s=100,ag_time=5):
        self.host=host                 # The address for the elastic cluster, currently: '52.169.220.43:9200'
        self.device=device             # the machine SN that will be used for filtering, for example the 69823
        self.root_index='telemetry_'   # The base to form all the names of other roots, as for example: telemetry_122020 (December of 2020)
        self.client=[]                 # Variable to store the client that has the methods of the ES library
        self.size=s                    # number of samples per batch of the scroll 
        self.agg_seconds=ag_time       # the aggregation time to filter the read inputs
        self.names_analysis=['models_error', 'low_bounds', 'high_bounds', 'activations', 'confidence','timestamp'] # the name of the main fields loaded appart from device and timestamp (??)
         
     def connect(self):
        self.client = Elasticsearch(hosts=[self.host])
        #print(self.client.info())
         
     # in order to retrieve the right cluster, given the nomenclature as: telemetry_XXYYYY (XX=month, YYYY=year)
     # Timestap format (aka date)= "2019-12-30T09:25:20.000Z"
     def get_index(self,date):
         index=self.root_index+date[5:7]+date[0:4]
         return index 
     # to obtain the data from a single variable in a single time stamp interval
     # will return a dictionary 
     def get_one(self,var,dates):
        response = self.client.search(
            index=self.get_index(dates[0]),
            body={
                  "query": {
                    "bool": {
                      # Also: filter, must_not, should
                      "must": [ 
                        {
                          "match": {
                            "deviceId": self.device
                          }
                        },
                        {
                          "match": {
                            "aggregationSeconds": self.agg_seconds
                          }
                        },
                        {
                          "match": {
                            # name of the variable from sensors
                            "param": var 
                          }
                        },
                        {
                        "range": {
                        # Timestap format= "2019-12-30T09:25:20.000Z"
                        "timestamp": { 
                        "gt": dates[0], # Date Format in Fault Manager: '2019-05-02 08:00:10'
                        "lt": dates[1] 
                                     }
                                 }
                        }
                      ],
                      "must_not": [],
                      "should": []
                    }
                  },
                  "from": 0,
                  "size": self.size,
                  "sort": [{ "timestamp" : {"order" : "asc"}}],
                  "aggs": {}
                },
            scroll='5m'
        )


        data={}
        for hit in response['hits']['hits']:
            #print(hit)
            data[hit['_source']['timestamp']]=hit['_source']['avg']
            
        sc_id=response['_scroll_id']
        more=True
        while more:
            sc=self.client.scroll(scroll_id=sc_id,scroll='2m') # ,scroll='1m'
            #sc_id=response['_scroll_id']
            if len(sc['hits']['hits'])==0: #|| total>20
                more=False
            else:
                for hit in sc['hits']['hits']:
                    data[hit['_source']['timestamp']]=hit['_source']['avg']
                    
        return data
        
     # get all variables in list, from the given time_stamp
     def get_variables(self,names,dates):
        data={'timestamp':[]}
        for name in names:
            data[name]=[]
        d=[]
        n=0
        tot=len(names)
        for name in names:
            n=n+1
            not_done=True
            while not_done:
                try:
                    self.connect()
                    d.append(self.get_one(name,dates))
                    print('% of variables collected: '+str(n/tot))
                    not_done=False
                except:
                    print('Failed attempt to connect with ES')    

        n=0
        tot=len(d[0])
        for i in d[0]:
            n=n+1
            all_in=True
            for j in range(len(d)):
                if i not in d[j]:
                    all_in=False
            if all_in:
                data['timestamp'].append(i)
                for j in range(len(d)):
                    data[names[j]].append(d[j][i])
            if (n%1000)==0:
                print('% of samples processed: '+str(n/tot))

        return data
    
     # get all variables in list, from the given list of time_stamps
     def get_set(self,names,dates):
        first=True
        tot=len(dates)
        k=0
        for time in dates:
            k=k+1
            self.connect()
            print('Evaluating timeband #'+str(k)+' out of '+str(tot))
            if first:
                first=False
                data=self.get_variables(names,time)
            else:
                nexty=self.get_variables(names,time)
                for n in data:
                    data[n]=data[n]+nexty[n]
            
        return data
###############################################################################
     # in order to retrieve the right cluster, given the nomenclature as: analytics_XXYYYY (XX=month, YYYY=year)
     # Timestap format (aka date)= "2019-12-30T09:25:20.000Z"
     def get_index_analytics(self,date,ty):
         index=ty+date[5:7]+date[0:4]
         return index 
     
     def get_ids(self,date,ty,sn):
         index=ty+sn+'_'+date
         return index 
     
     # write a new document in an Analytics_ cluster --> will try to overwrite if already exists
     def create_new_doc(self,data):
         if 'group_prob' in data[0]:
             ty='pm_bgm_data_'
             doc_class='diagnosis_bgm'
         else:
             ty='pm_data_'
             doc_class='diagnosis'
         self.connect()
         response=True
         ind=self.get_index_analytics(data[0]['timestamp'],ty)
         for d in range(len(data)):
             model=str(data[d]['device'])+data[d]['trained_version']
             data[d]['_id']=self.get_ids(data[d]['timestamp'],ty,model)
         #b=json.dumps(data)
         try:
             helpers.bulk(self.client, data, index=ind,doc_type=doc_class)
         except:
             response=False
             print(' [!] Error uploading pm_data')
             traceback.print_exc()
         return response
     
     # retrieve the data for forecasts/probabilities 
     def get_analytics(self,time_start, time_stop,version,probs):
         self.connect()
         if probs:
             ty='pm_bgm_data_'
             names_analysis=self.names_analysis
             names_analysis.append('group_prob')
             #doc_class='diagnosis_bgm'
         else:
             ty='pm_data_'
         ind=self.get_index_analytics(time_start,ty) 
         response = self.client.search(
            index=ind,
            body={
                  "query": {
                    "bool": {
                      # Also: filter, must_not, should
                      "must": [ 
                        {
                          "match": {
                            "device": self.device
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
                  "size": self.size,
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
                 n_msos=len(hit['_source'][self.names_analysis[0]])
                 for i in range(n_msos):
                     new_mso={}
                     for name in self.names_analysis:
                         field=hit['_source'][name]
                         if name=='timestamp':
                             new_mso[name]=[field]
                         else:
                             new_mso[name]=[field[i]]
                     data.append(new_mso)
             else: 
                 for i in range(n_msos):
                     for name in self.names_analysis:
                         field=hit['_source'][name]
                         if name=='timestamp':
                             data[i][name].append(field)
                         else:
                             data[i][name].append(field[i])

            
         sc_id=response['_scroll_id']
         more=True
         while more:
             sc=self.client.scroll(scroll_id=sc_id,scroll='2m') # ,scroll='1m'
             #sc_id=response['_scroll_id']
             if len(sc['hits']['hits'])==0: #|| total>20
                 more=False
             else:
                 for hit in sc['hits']['hits']:
                     if len(hit['_source'][self.names_analysis[0]])==n_msos:
                         for i in range(n_msos):
                             for name in self.names_analysis:
                                 field=hit['_source'][name]
                                 if name=='timestamp':
                                     data[i][name].append(field)
                                 else:
                                     data[i][name].append(field[i])
                     else:
                         print('  [!] WARNING: The gathered analysis data might come from different models, two sizes of MSO_SET: '+str(len(hit['_source'][self.names_analysis[0]]))+', '+str(n_msos)+'  | timestamp: '+hit['_source']['timestamp'])
                    
         return data
    
     #LOAD Forecast: Using bulk function and where data is a list of dictionaries (each doc to be uploaded)
     def load_forecast(self,data):
         ty='fore_'
         self.connect()
         response=True
         ind=self.get_index_analytics(data[0]['timestamp'],ty)
         for d in range(len(data)):
             model=str(data[d]['device'])+data[d]['trained_version']
             data[d]['_id']=self.get_ids(data[d]['timestamp'],(ty+data[d]['mso_name']),model)
         #b=json.dumps(data)
         try:
             helpers.bulk(self.client, data, index=ind,doc_type='forecast')
         except:
             response=False
             print(' [!] Error uploading forecast')
             traceback.print_exc()
         return response
     
     def load_probability(self,data):
         ty='baye_'
         self.connect()
         response=True
         ind=self.get_index_analytics(data[0]['timestamp'],ty)
         for d in range(len(data)):
             model=str(data[d]['device'])+data[d]['trained_version']
             data[d]['_id']=self.get_ids(data[d]['timestamp'],(ty+data[d]['fault']),model)
         #b=json.dumps(data)
         try:
             helpers.bulk(self.client, data, index=ind,doc_type='bayesian')
         except:
             response=False
             print(' [!] Error uploading probability')
             traceback.print_exc()
         return response
     
     #LOAD Configuration: Resulting from each forecast computation 
     def load_configuration(self,data):
         ty='conf_'
         self.connect()
         ind=self.get_index_analytics(data['timestamp'],ty)
         model=str(data['device'])+data['trained_version']
         id_es=self.get_ids(data['timestamp'],ty,model)
         b=json.dumps(data)
         response=False
         try:
             response=self.client.create(index=ind,id=id_es,body=b,doc_type='pm_configuration')
         except:
             try:
                 print('[I] Trying to delete old document - Configuration.')
                 response=self.client.delete(index=ind,id=id_es,doc_type='pm_configuration')
                 response=self.client.create(index=ind,id=id_es,body=b,doc_type='pm_configuration')
             except:
                 print(' [!] Error uploading configuration')
                 traceback.print_exc()
         return response
     
     #LOAD Report: Resulting from each summary report
     def load_report(self,data):
         ty='rep_'
         self.connect()
         ind=self.get_index_analytics(data['timestamp'],ty)
         model=str(data['device'])+data['trained_version']
         id_es=self.get_ids(data['timestamp'],ty,model)
         if data['critical_time']=="":
             data['critical_time']="1969-07-20T20:17:40.000Z"
         b=json.dumps(data)
         response=False
         try:
             response=self.client.create(index=ind,id=id_es,body=b,doc_type='report')
         except:
             try:
                 print('[I] Trying to delete old document - Report.')
                 response=self.client.delete(index=ind,id=id_es,doc_type='report')
                 response=self.client.create(index=ind,id=id_es,body=b,doc_type='report')
             except:
                 print(' [!] Error uploading Report')
                 traceback.print_exc()
         return response
     
     #LOAD Report: Resulting from each summary report
     def load_report(self,data):
         ty='notif_'
         self.connect()
         ind=self.get_index_analytics(data['timestamp'],ty)
         model=str(data['device'])+data['trained_version']
         id_es=self.get_ids(data['timestamp'],ty,model)
         b=json.dumps(data)
         response=False
         try:
             response=self.client.create(index=ind,id=id_es,body=b,doc_type='notification')
         except:
             try:
                 print('[I] Trying to delete old document - notification.')
                 response=self.client.delete(index=ind,id=id_es,doc_type='notification')
                 response=self.client.create(index=ind,id=id_es,body=b,doc_type='notification')
             except:
                 print(' [!] Error uploading notification')
                 traceback.print_exc()
         return response
     
     #DOWNLOAD Configuration: Resulting from each forecast computation 
     def get_configuration(self,times,version):
         '''{
                           "distance_feature": {
                             "field": "timestamp",
                             "pivot": "2d",
                             "origin": time
                           }
                        }'''
        #How to make a search for the "closest date" --> CURRENT: interval from Hiroshima drop up to the latest date in the interval evaluated, from there sorted 
         ty='conf_'
         self.connect()
         ind=self.get_index_analytics(times[1],ty) 
         response = self.client.search(
            index=ind,
            body={
                  "query": {
                    "bool": {
                      # Also: filter, must_not, should
                      "must": [ 
                        {
                          "match": {
                            "device": self.device
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
                        "gt": times[0], # Date Format in Fault Manager: '2019-05-02 08:00:10'
                        "lt": times[1] 
                                     }
                                 }
                        }
                      ],
    
                      "must_not": [],
                      "should": []
                  }
                },
                  "from": 0,
                  "size": 5,
                  "sort": [{ "timestamp" : {"order" : "asc"}}],
                  "aggs": {}
                },
            scroll='5m'
         )
    
         start=True
         data=[]
         for hit in response['hits']['hits']:
             #print(hit)
             if start:
                 data=hit['_source']
                 start=False
                    
         return data
     
     def get_last_priors(self,times,faults,version,length=200):
         '''{
                           "distance_feature": {
                             "field": "timestamp",
                             "pivot": "2d",
                             "origin": time
                           }
                        }'''
        #How to make a search for the "closest date" --> CURRENT: interval from Hiroshima drop up to the latest date in the interval evaluated, from there sorted 
         ty='baye_'
         self.connect()
         ind=self.get_index_analytics(times[1],ty) 
         response = self.client.search(
            index=ind,
            body={
                  "query": {
                    "bool": {
                      # Also: filter, must_not, should
                      "must": [ 
                        {
                          "match": {
                            "device": self.device
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
                        "gt": times[0], # Date Format in Fault Manager: '2019-05-02 08:00:10'
                        "lt": times[1] 
                                     }
                                 }
                        }
                      ],
    
                      "must_not": [],
                      "should": []
                  }
                },
                  "from": 0,
                  "size": length,
                  "sort": [{ "timestamp" : {"order" : "asc"}}],
                  "aggs": {}
                },
            scroll='5m'
         )
    
         got_it={}
         for hit in response['hits']['hits']:
             #print(hit)
             if hit['_source']['fault'] not in got_it:
                 got_it[hit['_source']['fault']]=hit['_source']['probability']
         
         priors=[]
         for i in faults:
             priors.append(got_it[faults[i]])
         
         return priors
     
        
     def get_reports(self,times,faults,version,length=200):
         ty='rep_'
         ind=self.get_index_analytics(times[1],ty) 
         response = self.client.search(
            index=ind,
            body={
                  "query": {
                    "bool": {
                      # Also: filter, must_not, should
                      "must": [ 
                        {
                          "match": {
                            "device": self.device
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
                        "gt": times[0], # Date Format in Fault Manager: '2019-05-02 08:00:10'
                        "lt": times[1] 
                                     }
                                 }
                        }
                      ],
    
                      "must_not": [],
                      "should": []
                  }
                },
                  "from": 0,
                  "size": length,
                  "sort": [{ "timestamp" : {"order" : "asc"}}],
                  "aggs": {}
                },
            scroll='5m'
         )
    
         data={}
         for hit in response['hits']['hits']:
             if hit['_source']['timestamp'] not in data:
                 data[hit['_source']['timestamp']]=[hit['_source']]
             else:
                 data[hit['_source']['timestamp']].append(hit['_source'])
        
         sc_id=response['_scroll_id']
         more=True
         while more:
             sc=self.client.scroll(scroll_id=sc_id,scroll='2m') # ,scroll='1m'
             #sc_id=response['_scroll_id']
             if len(sc['hits']['hits'])==0: #|| total>20
                 more=False
             else:
                 for hit in sc['hits']['hits']:
                     if hit['_source']['timestamp'] not in data:
                         data[hit['_source']['timestamp']]=[hit['_source']]
                     else:
                         data[hit['_source']['timestamp']].append(hit['_source'])
                    
         return data
 
         