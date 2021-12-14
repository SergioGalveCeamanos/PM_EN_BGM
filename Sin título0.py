# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:33:05 2021

@author: sega01
"""
if True:
    import os, requests, uuid, json, pickle
    from os import path
    import numpy as np
    import pandas as pd
    from classes.conditional_analysis_methods import get_joint_tables,get_cond_activ_mtrs,get_mean_std_mtrs,moving_average_variables,corrected_matrices,final_selection,launch_report_generation
    from classes.pm_manager import dif_dates,upload_results,get_analysis,set_new_model,update_model,get_available_models,get_forecast, get_probability,generate_report,load_unit_data,notification_trigger,get_data_batch,generate_report
    from classes.fault_detector_class_ES import fault_detector
    from classes.pm_manager import load_model,file_location,homo_sampling,conditional_analysis,dif_dates,key_strokes
    from classes.MSO_selector_GA import find_set_v2
    from classes.test_cross_var_exam import launch_analysis, fix_dict
    import datetime 
    import traceback
    import copy
    import multiprocessing
    device=74124
    length=24
    ma=[5,7,9,12]
    space_between=12
    version='_test_III_NotifSyst_181021'#"_test_II_NoFStab_150721_v1"#'_test_I_NoFStab_130721'#•'_test_I_ClustSegm_180621_v1'#_test_II_Redo_100621,_test_I_Redo_100621,_test_VI_StabilityFilt_090621,_test_V_StabilityFilt_040621,_test_IV_StabilityFilt_040621,#_test_I_260521,_test_II_StabilityFilt_260521,_test_XII_NewSelec_120421'#"_test_XI_NewSelec_080421"#"_test_X_NewSelec_070421"
    time_start="2021-05-26T07:00:00.000Z"
    time_stop="2021-07-08T07:30:00.000Z"
    aggSeconds=1
    NEW_TIMEOUT=600
    file, folder = file_location(device,version)
    f=load_model(file, folder)
    mso=0
    t=0
    fm=f.models[f.mso_set[mso]]
    new_data=f.training_data
    normed_predict_data=fm.get_prediction_input(new_data)
    measured_value=normed_predict_data[fm.objective]
    source_value=normed_predict_data[fm.source]
    contour_cond=normed_predict_data[fm.cont_cond]
    groups=fm.bgm.predict(contour_cond.values)
    predictions=np.zeros(source_value.shape[0]) # not that useful now
    probs=np.zeros([source_value.shape[0],len(fm.regions)])
    
    locats=np.where(groups == t)[0]
    selection=source_value.iloc[locats]
    cont_selec=contour_cond.iloc[locats]
    reference=measured_value.iloc[locats]
    if selection.shape[0]!=0:
        forecasted=fm.model[t]['model'].predict(selection)
        error = reference - forecasted  
        
#define optimization problem
if True:
    def toVector(w, z):
        assert w.shape == (2, 4)
        assert z.shape == (2, 4)
        return np.hstack([w.flatten(), z.flatten()])

    def toWZ(vec):
        assert vec.shape == (2*2*4,)
        return vec[:2*4].reshape(2,4), vec[2*4:].reshape(2,4)  
 
    def doOptimization(f_of_w_z, w0, z0):
        def f(x): 
            w, z = toWZ(x)
            return f_of_w_z(w, z)
        result = minimize(f, toVec(w0, z0))
        # Different optimize functions return their
        # vector result differently. In this case it's result.x:
        result.x = toWZ(result.x) 
        return result
    #https://stackoverflow.com/questions/31292374/how-do-i-put-2-matrix-into-scipy-optimize-minimize
    nor=np.linalg.norm(telem.dot(fm.hos[t]),ord=1)
    
    