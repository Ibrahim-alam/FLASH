#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 10:20:39 2022

@author: papuska
"""
#%% Importing Modules

import time
start_time=time.time()
import os
import csv
import random
import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor # Random Forest
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier # SimpleCART
# from rotation_forest import RotationForestClassifier # Rotation Forest
from sklearn.linear_model import LogisticRegression # Logistic
# from sklearn.naive_bayes import GaussianNB # CategoricalNB did not work -- DTNB?? Not exacly, DTNB is a hybrid NB
from sklearn.neural_network import MLPClassifier # Multi-Layer perceptron : used in the Learners Set
import xgboost as xgb
import lightgbm as lgbm

from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_validate

# from optimizer import HyperoptOptimizer

import pickle as pkl

time_import=time.time()-start_time
import warnings
warnings.filterwarnings("ignore")

#%% Functions
def Clf_param_api(Clf):
    if Clf=='RandomForestClassifier':
        space = {'n_estimators': hp.uniformint("n_estimators", 50, 300), #random.randint(100,1000), #default 100
                    # 'criterion':hp.choice("criterion", ["gini", "entropy"]),  #default gini
                    'max_depth' :hp.uniformint("max_depth", 4, 20), #random.randint(1,10),      # as much as it is needed (may be default is better)
                    'min_samples_split' : hp.uniformint('min_samples_split', 2, 6),  #random.randint(2,10), #default 2
                    'min_samples_leaf' : hp.uniformint('min_samples_leaf', 1, 3) #random.randint(1,3), #default 1
                    #'max_features' : ['log2', 'sqrt'][random.randint(0,1)],
                    #'bootstrap' : [True, False][random.randint(0,1)],
                    }
    elif Clf=='GradientBoostingClassifier':
        space = {'learning_rate':hp.uniform('learning_rate', 1e-1,0.5),# 0.1, 
                    'n_estimators': hp.uniformint("n_estimators", 50, 300), #random.randint(100,1000), #default 100
                    'max_depth' :hp.uniformint("max_depth", 4, 20), #random.randint(1,10),      # as much as it is needed (may be default is better)
                    'min_samples_split' : hp.uniformint('min_samples_split', 2, 6), #random.randint(2,10), #default 2
                    'min_samples_leaf' : hp.uniformint('min_samples_leaf', 1, 3) #random.randint(1,3), #default 1
                    #'max_features' : ['log2', 'sqrt'][random.randint(0,1)],
                    #'bootstrap' : [True, False][random.randint(0,1)],
                    }
    elif Clf=='DecisionTreeClassifier':
        space = {'max_depth' : hp.uniformint("max_depth", 4, 20),#random.randint(1,10),      # as much as it is needed (may be default is better)
                    'min_samples_split' : hp.uniformint('min_samples_split', 2, 6), #random.randint(2,10), #default 2
                    'min_samples_leaf' : hp.uniformint('min_samples_leaf', 1, 3)  #random.randint(1,3), #default 1
                    #'max_features' : ['log2', 'sqrt'][random.randint(0,1)],
                    #'bootstrap' : [True, False][random.randint(0,1)],
                    }
    elif Clf=='ExtraTreeClassifier':
        space = {'max_depth' : hp.uniformint("max_depth", 4, 20), #random.randint(1,10),      # as much as it is needed (may be default is better)
                    'min_samples_split' : hp.uniformint('min_samples_split', 2, 6), #random.randint(2,10), #default 2
                    'min_samples_leaf' : hp.uniformint('min_samples_leaf', 1, 3) #random.randint(1,3), #default 1
                    #'max_features' : ['log2', 'sqrt'][random.randint(0,1)],
                    #'bootstrap' : [True, False][random.randint(0,1)],
                    }
    elif Clf=='LogisticRegression':
        space = {'tol':hp.uniform('tol', 1e-4, 1e-3), #1e-3 loguniform
                 'C': hp.uniform("C", 0.2, 1.0), # greater than 0
                 'max_iter' : hp.uniformint('max_iter', 80, 100),#100
                    }
    elif Clf=='xgb.XGBClassifier':
        space={'eta':hp.uniform('eta', 0.1, 0.3), #0.3
               'min_child_weight': hp.uniformint('min_child_weight', 1, 3),
               'max_depth' :hp.uniformint("max_depth", 3, 12), #3-10
                    } # Rest of the parameters are too complex or not much used
    elif Clf=='lgbm.LGBMClassifier':
        space={'bagging_fraction':hp.uniform('bagging_fraction', 0.5, 1.0),#(0.5, 0.8),
               'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0), #(0.5, 0.8),
               'max_depth' :hp.uniformint("max_depth", 3, 12), #(10, 13),
                'min_data_in_leaf': hp.uniformint('min_data_in_leaf', 15, 30) ,#(20),
                'num_leaves': hp.uniformint('num_leaves', 20, 80) #(31),
                    }
    elif Clf=='MLPClassifier':
        space={'hidden_layer_sizes':hp.uniformint('hidden_layer_sizes', 50, 200),#(0.5, 0.8),
                    'alpha': hp.loguniform('alpha', -5*np.log(10), 1*np.log(10)), #(0.5, 0.8),
                    'learning_rate_init' :hp.loguniform('learning_rate_init',-5*np.log(10), -1*np.log(10)) #(10, 13),
                    }
    
    return space

def Clf_parameter_names(Clf):
    if Clf=='RandomForestClassifier':
        lst_hps={'n_estimators':'int',
                 'max_depth':'int', 
                 'min_samples_split': 'int',
                 'min_samples_leaf':'int'}
    elif Clf=='GradientBoostingClassifier':
        lst_hps={'learning_rate':'float',
                 'n_estimators':'int',
                 'max_depth':'int',
                 'min_samples_split':'int',
                 'min_samples_leaf':'int'}
    elif Clf=='DecisionTreeClassifier':
        lst_hps={'max_depth':'int', 
                 'min_samples_split': 'int',
                 'min_samples_leaf': 'int'}
    elif Clf=='ExtraTreeClassifier':
        lst_hps={'max_depth': 'int',
                 'min_samples_split': 'int',
                 'min_samples_leaf': 'int'}
    elif Clf=='LogisticRegression':
        lst_hps={'tol':'float',
                 'C':'float',
                 'max_iter': 'int'}
    elif Clf=='xgb.XGBClassifier':
        lst_hps={'eta':'float',
                 'min_child_weight': 'int',
                 'max_depth': 'int'}
    elif Clf=='lgbm.LGBMClassifier':
        lst_hps={'bagging_fraction':'float',
                 'feature_fraction':'float',
                 'max_depth': 'int',
                 'min_data_in_leaf': 'int',
                 'num_leaves': 'int'}
    elif Clf=='MLPClassifier':
        lst_hps={'hidden_layer_sizes':'int',
                 'alpha':'float',
                 'learning_rate_init': 'float'}
    return lst_hps 

def Clf_parameters_assign(Clf, param):
    
    if Clf=='RandomForestClassifier':
        Clf_with_para=RandomForestClassifier(**param, n_jobs=-1)
    elif Clf=='GradientBoostingClassifier':
        Clf_with_para=GradientBoostingClassifier(**param)
    elif Clf=='DecisionTreeClassifier':
        Clf_with_para=DecisionTreeClassifier(**param)
    elif Clf=='ExtraTreeClassifier':
        Clf_with_para=ExtraTreeClassifier(**param)
    elif Clf=='LogisticRegression':
        Clf_with_para=LogisticRegression(**param) #n_jobs=-1)
    elif Clf=='xgb.XGBClassifier':
        Clf_with_para=xgb.XGBClassifier(**param, n_jobs=-1)
    elif Clf=='lgbm.LGBMClassifier':
        Clf_with_para=lgbm.LGBMClassifier(**param, n_jobs=-1)
    elif Clf=='MLPClassifier':
        Clf_with_para=MLPClassifier(**param)
    return Clf_with_para

def Clf_param(Clf):
    if Clf=='RandomForestClassifier':
        param = {'n_estimators': random.randint(50,300), #default 100
                # 'criterion':['gini','entropy'][random.randint(0,1)], #default gini
                'max_depth' : random.randint(1,25),      # as much as it is needed (may be default is better)
                'min_samples_split' : random.randint(2,6), #default 2
                'min_samples_leaf' : random.randint(1,3) #default 1
                #'max_features' : ['log2', 'sqrt'][random.randint(0,1)],
                #'bootstrap' : [True, False][random.randint(0,1)],
                    }
    elif Clf=='GradientBoostingClassifier':
        param = {'learning_rate': 0.1, 
                'n_estimators': random.randint(50,300), #default 100
                'max_depth' : random.randint(1,25),      # as much as it is needed (may be default is better)
                'min_samples_split' : random.randint(2,6), #default 2
                'min_samples_leaf' : random.randint(1,3) #default 1
                #'max_features' : ['log2', 'sqrt'][random.randint(0,1)],
                #'bootstrap' : [True, False][random.randint(0,1)],
                    }
    elif Clf=='DecisionTreeClassifier':
        param = {'max_depth' : random.randint(1,25),      # as much as it is needed (may be default is better)
                'min_samples_split' : random.randint(2,6), #default 2
                'min_samples_leaf' : random.randint(1,3) #default 1
                }
    elif Clf=='ExtraTreeClassifier':
        param = {'max_depth' : random.randint(1,25),      # as much as it is needed (may be default is better)
                'min_samples_split' : random.randint(2,6), #default 2
                'min_samples_leaf' : random.randint(1,3) #default 1
                }
    elif Clf=='LogisticRegression':
        param = {'tol' : 1e-5*random.randint(1,100),      # as much as it is needed (may be default is better)
                'C' : 1e-2*random.randint(20,100), #default 1
                'max_iter' : random.randint(80,100) #default 100
                }
    elif Clf=='xgb.XGBClassifier':
        param = {'eta':3*10e-5*random.randint(100,1000),
                 'min_child_weight': random.randint(1,3),
                 'max_depth': random.randint(3,12) 
                 }
    elif Clf=='lgbm.LGBMClassifier':
        param = {'bagging_fraction':5*10e-4*random.randint(100,200),
                 'feature_fraction':5*10e-4*random.randint(100,200),
                 'max_depth': random.randint(5,12), 
                 'min_data_in_leaf': random.randint(15,30),
                 'num_leaves': random.randint(20,80)
                }
    elif Clf=='MLPClassifier':
        param = {'hidden_layer_sizes':random.randint(50, 200),#(0.5, 0.8),
                 'alpha': np.exp(np.random.uniform(-5*np.log(10), 1*np.log(10))), #(0.5, 0.8),
                 'learning_rate_init' :np.exp(np.random.uniform(-5*np.log(10), -1*np.log(10)))
                 }
    return param


def clean_data(directory, file):
    df_raw = pd.read_csv(directory+ file['DataSet_name'].iloc[0]+'.csv')
    # Deleting the column of df_raw which is redundant/extra in the csv
    if ~file['remove'].isnull().iloc[0]:
        del df_raw[file['remove'].iloc[0]]
    
    #reading column names /headers after the removal
    col_names=list(df_raw.columns)
    char_to_rmv=[',','.',':','/','!','@','#','$','%','^','&','*','(',')']
    for m in range(len(col_names)):
        for n in range(len(char_to_rmv)):
            if char_to_rmv[n] in col_names[m]:
                col_names[m]=col_names[m].replace(char_to_rmv[n], '')
    
    # Imputing and coding data to fill missing data and ready for classifiers/regressors
    impt=SimpleImputer(strategy = 'constant')
    impt.fit(df_raw)
    v1=impt.transform(df_raw)
    df_impt = pd.DataFrame(v1, columns = col_names)
    df_impt = df_impt.astype(str)
    
    enc=OrdinalEncoder()
    enc.fit(df_impt)
    v2=enc.transform(df_impt)
    df_enc = pd.DataFrame(v2, columns = col_names)
    
    df = df_enc
    # changing target column name to match previous change
    targt=file['Target_Class'].iloc[0]
    for m in range(len(char_to_rmv)):
        if char_to_rmv[m] in targt:
            targt=targt.replace(char_to_rmv[m], '')
    
    # popping up target column to put it in first column
    column_to_move = df.pop(targt)
    df.insert(0, targt, column_to_move)
    return df

def fetch_data(directory, client): #you may think of cleaning the data once and not for all iteration
    df = pd.read_csv(directory+ '/Client_'+str(client+1)+'.csv')
    # train, test, holdout division
    train, test = train_test_split(df, test_size=0.3, random_state=Data_Seed, stratify=(df.iloc[:,0])) #  'class' for Electricity, 'Class' for EEG
    train_only, holdout = train_test_split(train, test_size=1/7, random_state=Data_Seed, stratify=(train.iloc[:,0])) # 'class' for Electricity, 'Class' for EEG

    return train, train_only, holdout, test

def HPO_per_client(Clfrs, i, j, N_i, HPO_ITER, target_dir, file):
    train, train_only, holdout, test=fetch_data(target_dir, j)
    
    if N_i[i][-1]<1:
        train_new, train_later = train_test_split(train, train_size=N_i[i][-1], random_state=Data_Seed, stratify=train.iloc[:,0])
    else:
        train_new=train #make these two to train to train_only if test with holdout
    
    LL=len(train.columns)
    x_train = train_new.iloc[:,1:LL]
    y_train = train_new.iloc[:,0:1]
    
    # x_test=holdout.iloc[:,1:LL]
    # y_test=holdout.iloc[:,0:1]
    space=Clf_param_api(Clfrs[i])
    trials = Trials()
    def hyperparameter_tuning(params):
        clf = Clf_parameters_assign(Clfrs[i], params)
        # clf.fit(x_train, y_train.values.ravel())
        # acc = - balanced_accuracy_score(clf.predict(x_test), y_test)
        cv_results = cross_validate(clf, x_train, y_train.values.ravel(), scoring=('balanced_accuracy'), cv=10, n_jobs=-1) #, return_train_score=True)
        acc = - np.mean(cv_results['test_score'])
        return {"loss": acc, "status": STATUS_OK, 'eval_time': time.time()}
    
    rstate=np.random.default_rng(HPO_Seed)
    time_HPO_start=time.time()
    best = fmin(
        fn=hyperparameter_tuning,
        space = space, 
        algo=tpe.suggest, 
        max_evals=HPO_ITER[i], 
        trials=trials,
        rstate=rstate
    )
    
    print("Best: {}".format(best))
        
    HPs=pd.DataFrame.from_dict(trials.vals, orient='columns', dtype=None, columns=None)
    HPs['Accuracy'] = trials.losses()
    
    temp=[]
    Time_temp=[time_HPO_start]
    for k in range(HPO_ITER[i]):
        Time_temp.append(trials.results[k].get('eval_time'))
        temp.append(Time_temp[k+1]-Time_temp[k])
    HPs['time_duration']=temp
    HPs=HPs.sort_values(by=['Accuracy'],ascending=True).reset_index(drop=True)
    temp=list(range(0, HPO_ITER[i]))
    HPs.insert(0,'indx',temp)
    HPs.insert(0, 'Client', 'Client '+str(j+1))
    HPs.insert(0, 'Algorithm', Clfrs[i])
    HPs['datasize']=N_i[i][-1]
    return HPs


def Top_Rand_HP_score (i, Top_Rand_HPs_dict, N_i):
    Top_Rand_HPs_score_df=pd.DataFrame()
    
    for j in range(N_Clients):
        train, train_only, holdout, test = fetch_data(target_dir, j)
    
        if N_i[i][-1]<1:
            train_new, train_later = train_test_split(train, train_size=N_i[i][-1], random_state=Data_Seed, stratify=train.iloc[:,0]) 
        else:
            train_new=train
        
        LL=len(train.columns)
        x_train = train_new.iloc[:,1:LL]
        y_train = train_new.iloc[:,0:1]
        
        # x_test=holdout.iloc[:,1:LL]
        # y_test=holdout.iloc[:,0:1]
        
        Top_Rand_HPs_score=[]
        for m in range(len(Top_Rand_HPs_dict)):
            clf_model = Clf_parameters_assign(Clfrs[i],Top_Rand_HPs_dict[m])
            # clf_model.fit(x_train, y_train.values.ravel())
            cv_results = cross_validate(clf_model, x_train, y_train.values.ravel(), scoring=('balanced_accuracy'), cv=10, n_jobs=-1) #, return_train_score=True)
            Top_Rand_HPs_score.append(np.mean(cv_results['test_score']))
        Top_Rand_HPs_score_df['client '+str(j+1)+' score']=Top_Rand_HPs_score
    return Top_Rand_HPs_score_df
    
    
def main_func_FLASH():
    main_start_time=time.time()
    
    N_i=[ [] for _ in range(N_Clsfrs)] # fraction of data that will be given to Algorithm from train data
    Clsfr_acc=[ [] for _ in range(N_Clsfrs)]
    Full_performance=[ [] for _ in range(N_Clsfrs)]
    Top_HPs_all_clsfr=[ [] for _ in range(N_Clsfrs)]
    ub=[ [] for _ in range(len(Clfrs))]
    ub_list=[ [] for _ in range(len(Clfrs))]
    p=1
    itr=0
    List_i=list(range(0, N_Clsfrs))
    while(p>0):
        if itr>2:
            for i in range(len(Clfrs)):
                if len(ub_list[i])<len(N_i[i])-2:
                    v1=len(Clsfr_acc[i])
                    xx=N_i[i][v1-3:v1]
                    yy=Clsfr_acc[i][v1-3:v1]
                    v2, v3 = np.polyfit(xx, yy, 1)
                    ub[i]=Clsfr_acc[i][-1]+(0.8-N_i[i][-1])*v2
                    ub_list[i].append(ub[i])
            v4=max(ub)
            i=ub.index(v4)
            v6=r*N_i[i][-1]
            if v6>1 and N_i[i][-1]>0.98:
                p=0
                break
            elif v6>1 and N_i[i][-1]<.98:     
                N_i[i].append(1)
            else:
                N_i[i].append(v6)
            List_i=[i]
        else:
            for i in range(N_Clsfrs):
                N_i[i].append(a*r**itr)
            if a==1: # checking if N-client per dataset is small or not
                p=0
        
        for i in List_i:
            Full_performance_Alg=pd.DataFrame()
            #%% HPO for each client
            for j in range(N_Clients):
                HPs=HPO_per_client(Clfrs,i,j,N_i,HPO_ITER,target_dir, file) # Calling the function to calculate HPs of HPO
                Full_performance_Alg=pd.concat([HPs, Full_performance_Alg], ignore_index = True, axis = 0)
              
            Full_performance[i].append(Full_performance_Alg)
            HPs_temp_df=Full_performance_Alg.copy()
            HPs_temp_df.drop(['Algorithm', 'Client','datasize','time_duration'], inplace=True, axis = 1)
            print('HPO calculation for Algorithm '+str(i+1)+' is done')
            
            #%%
            N_Top_HP=1 # we are taking the top from each client
            Top_HPs_of_client=HPs_temp_df.loc[HPs_temp_df.indx<N_Top_HP] #  IMPORTANT if this to indx needed
            # scores.append(np.mean(N_Top_pairs['Accuracy']))
            Aggr_HP=Top_HPs_of_client.drop(['Accuracy','indx'], axis = 1)
            # y_regressor=Top_HPs_of_client['Accuracy']
            HP_names=list(Aggr_HP.columns)
            Top_HPs_dict_list=[]
            Aggr_HP_ensemble=pd.DataFrame(columns = HP_names, dtype=object)
            Aggr_HP_ensemble=Aggr_HP.sum(axis=0)/len(Aggr_HP)
            api_config=Clf_parameter_names(Clfrs[i])
            for j in range(len(HP_names)):
                if api_config[HP_names[j]]=='int':
                    Aggr_HP_ensemble[j]=int(round(Aggr_HP_ensemble[j]))
            Aggr_HP.loc[len(Aggr_HP)]=Aggr_HP_ensemble
            Aggr_HP_dict={}
            for k in range(len(HP_names)):
                if api_config[HP_names[k]]=='int':
                    Aggr_HP_dict[HP_names[k]]= int(Aggr_HP[HP_names[k]].iloc[0])
                else:
                    Aggr_HP_dict[HP_names[k]]= Aggr_HP[HP_names[k]].iloc[0]
            Top_HPs_dict_list.append(Aggr_HP_dict)
                
            
            Top_Rand_HPs_score_df= Top_Rand_HP_score(i, Top_HPs_dict_list, N_i) # Another function
            print('Top Rand HP determination for Algorithm '+str(i+1)+' is done')
            #%% Performance Analysis for the N_Top_HPs
            
            Top_Rand_HPs_avg_score=(Top_Rand_HPs_score_df.sum(axis=1)).to_frame(name='Score')
            temp=list(range(0, len(Top_Rand_HPs_avg_score)))
            Top_Rand_HPs_avg_score['indx']=temp
            Top_Rand_HPs_avg_score=Top_Rand_HPs_avg_score.sort_values(by=['Score'],ascending=False).reset_index(drop=True)
            for m in range(len(Top_HPs_dict_list)):
                Top_HPs_all_clsfr[i].append(Top_HPs_dict_list[Top_Rand_HPs_avg_score['indx'].iloc[m]]) # saving top 10 HPs for each iteration
            temp=Top_Rand_HPs_avg_score.iloc[0,0]
            temp2=temp/N_Clients
            
            #%% Storing the best score (accuracy)
            if len(Clsfr_acc[i])>0:
                if temp2>=Clsfr_acc[i][-1]:
                    Clsfr_acc[i].append(temp2)
                else:
                    temp3=0.5*(Clsfr_acc[i][-1]+temp2)
                    Clsfr_acc[i].append(temp3)
            else:
                Clsfr_acc[i].append(temp2)
            print('Algorithm '+str(i+1)+' best HP calculation is done')
        print('FULL RUN SUCCESSFUL for ITERATION *** '+str(itr+1)+' ***')
        itr=itr+1
    
    
    #%%
    Final_training=[]
    Final_Test=[]
    
    for i in range(len(Clfrs)):
        Final_training.append(Clsfr_acc[i][-1])
        train_FL=pd.DataFrame()
        test_FL=pd.DataFrame()
        for j in range(N_Clients):
            train, train_only, holdout, test=fetch_data(target_dir, j)
            train = pd.concat([train_only, holdout], ignore_index = True, axis = 0)
            train_FL=pd.concat([train, train_FL], ignore_index = True, axis = 0)
            test_FL = pd.concat([test, test_FL], ignore_index = True, axis = 0)
            
        LL=len(train_FL.columns)
        x_train_FL = train_FL.iloc[:,1:LL]
        y_train_FL = train_FL.iloc[:,0:1]
        
        x_test_FL=test_FL.iloc[:,1:LL]
        y_test_FL=test_FL.iloc[:,0:1]
        
        Top_HP_index=len(Top_HPs_all_clsfr[i])-len(Top_HPs_dict_list)
        clf_model = Clf_parameters_assign(Clfrs[i], Top_HPs_all_clsfr[i][Top_HP_index]) # instead, taking a average of the best HPs may perform better
        clf_model.fit(x_train_FL, y_train_FL.values.ravel())
        Final_Test.append(balanced_accuracy_score(clf_model.predict(x_test_FL), y_test_FL))
    
    Time_full_run=time.time()-main_start_time
    return Full_performance, Top_HPs_all_clsfr, Clsfr_acc, Final_training, Final_Test
   


                                # ********** --------------- ************   

#%% Reading the file name and target info

current_dir=os.getcwd()

file = pd.read_csv(current_dir+'/Dataset_to_run.csv')

dataset_directory = current_dir+'/Datasets/'
dataset_main = clean_data(dataset_directory, file)
CV=1

Clfrs=['RandomForestClassifier', 'GradientBoostingClassifier',
       'DecisionTreeClassifier','ExtraTreeClassifier','LogisticRegression',
       'xgb.XGBClassifier','lgbm.LGBMClassifier','MLPClassifier'
       ]

N_Clsfrs=len(Clfrs)

#%%
Seeds=[0 ,1, 42, 1234, 10]
N_Seed=len(Seeds)

HPO_ITER = [50]*N_Clsfrs
    
N_Clients = 3 # total number of clients
N_data_client=len(dataset_main)/N_Clients
if N_data_client>5000:
    a=max(0.05,250/N_data_client) # taking max perc of 5% or 250 data points
else:
    a=1
r=(N_data_client/4000)**0.5 # the ratio is higher for large dataset.


#%% Loop to generate results for different seed
main_loop_start_time=time.time()

All_Results_combined=[]
Full_Clsfr_Acc_DF=pd.DataFrame()
for out_itr in range(N_Seed): #FIXME
    Data_Seed=Seeds[0]
    HPO_Seed=Seeds[out_itr]
    
    #%% Creating Clients
    dataset = dataset_main.copy()
    client_data_directory='Clients_of_'+file['DataSet_name'].iloc[0]
    target_dir=current_dir+client_data_directory
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for i in range(N_Clients-1):
        X1, X2 = train_test_split(dataset, train_size=1/(N_Clients-i), random_state=Data_Seed, stratify=dataset[file['Target_Class'].iloc[0]]) ## check
        dataset=X2
        X1.to_csv(target_dir+'/Client_'+str(i+1)+'.csv', index=False)
    X2.to_csv(target_dir+'/Client_'+str(i+2)+'.csv', index=False)
    
    #%% Saving Results
    
    Full_performance, Top_HPs_all_clsfr, Clsfr_acc, Final_training, Final_Test = main_func_FLASH()
    
    All_Results=[[],[],[]]
    All_Results[0]=Full_performance
    All_Results[1]=Top_HPs_all_clsfr
    All_Results[2]=Clsfr_acc
    All_Results_combined.append(All_Results)
    
    Clsfr_Acc_DF=pd.DataFrame()
    Clsfr_Acc_DF['Classifiers']=['RF','GB','DT','ET','LR','XGB','LGBM','MLP']
    Clsfr_Acc_DF['HPO - Train']=Final_training
    Clsfr_Acc_DF['HPO - Test']=Final_Test
    Clsfr_Acc_DF.insert(0, 'HPO-Seed', HPO_Seed)
    Clsfr_Acc_DF.insert(0, 'Data-Seed', Data_Seed)
    Full_Clsfr_Acc_DF=pd.concat([Clsfr_Acc_DF, Full_Clsfr_Acc_DF])

pkl.dump(All_Results_combined,open(file['DataSet_name'].iloc[0]+"_N_"+str(N_Clients)+"_FLASH_D3_All_HPO_Data_"+str(Data_Seed)+".p", "wb" ))
Full_Clsfr_Acc_DF.to_csv(file['DataSet_name'].iloc[0]+"_N_"+str(N_Clients)+'_FLASH_D3_Accuracy_'+str(Data_Seed)+'.csv')
main_loop_end_time=time.time()-main_loop_start_time
