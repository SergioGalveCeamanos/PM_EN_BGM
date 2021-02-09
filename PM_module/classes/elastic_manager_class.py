# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:25:47 2020

@author: sega01

Updated to include creation of messages
"""
from elasticsearch import Elasticsearch
import pandas as pd


class elastic_manager:
     def __init__(self, host, device,s=100,ag_time=5):
        self.host=host                 # The address for the elastic cluster, currently: '52.169.220.43:9200'
        self.device=device             # the machine SN that will be used for filtering, for example the 69823
        self.root_index='telemetry_'   # The base to form all the names of other roots, as for example: telemetry_122020 (December of 2020)
        self.client=[]                 # Variable to store the client that has the methods of the ES library
        self.size=s                    # number of samples per batch of the scroll 
        self.agg_seconds=ag_time       # the aggregation time to filter the read inputs
         
     def connect(self):
        self.client = Elasticsearch(hosts=[self.host])
         
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
            d.append(self.get_one(name,dates))
            print('% of variables collected: '+str(n/tot))

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
     def get_index_analytics(self,date):
         index="analytics_"+date[5:7]+date[0:4]
         return index 
     
     # write a new document in an Analytics_ cluster 
     def create_new_doc(self,time_start, time_stop,errors,probability):
         ind=self.get_index_analytics(time_start)
         b={"device":self.device,"time_start":time_start,"time_stop":time_stop,"model_errors":errors,"probability_evolution":probability}
         response=self.client.create(index=ind,body=b)
         return response
     
     # retrieve the data for analytics?
     def get_analytics(self,time_start, time_stop):
         a=1
        
            
            
        

 
         