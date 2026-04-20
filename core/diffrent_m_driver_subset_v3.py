#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
import scipy.io as sio
from scipy.io import loadmat
import os
from sklearn import preprocessing
import pandas as pd
import copy as cp
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pycatch22
from sklearn.linear_model import LinearRegression
def get_weights(X, y):
    
    logit_model = LinearRegression()
    logit_model.fit(X, y)
    logit_weights = np.abs(logit_model.coef_)

    return logit_weights

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()  
        self.log.flush()

    def flush(self):
        pass




### C22MP start

def cal_additional_features(x):
    return [np.min(x), np.std(x)]

def cal_catch22(x, cal_addtional_features= False):
    transformed_data =pycatch22.catch22_all(x)['values']
    if cal_addtional_features:
        additional_features = cal_additional_features(x)
        transformed_data.extend(additional_features)
    return transformed_data

def cal_feature_profiles(ts,win, step = 1 ,cal_addtional_features = False):
    transformed_seq = [] 
    for i in np.arange(0,len(ts)-win+1, step):
        new_subseq = ts[i:i+win]
        tr_new_subseq = cal_catch22(new_subseq, cal_addtional_features=cal_addtional_features)
        transformed_seq.append(tr_new_subseq)
    transformed_seq = np.asarray(transformed_seq) 
    return transformed_seq

def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def bfill(arr): 
    return ffill(arr[:, ::-1])[:, ::-1]


def eudis1(v1, v2):
    return np.linalg.norm(v1-v2)
def left_c22_mp(X,  
                start_idx, 
                exclude_zone, 
                earlly_abaondon = True, 
                verbose=False, 
                get_c22_mp_idxs =False,
                you_can_do_better = False,
                dynamic_bsf_update =False,
                look_back_window_factor = 2,
                weights = None,
                ):
  
    """Left_C22_MP.

    Parameters
    ----------
    X : array
      the feature profile array to calculate its left_c22_MP. 

    start_idx : int 
      The index of the first sample in test data. 
      note- since we use all the samples before start_idx as traing data, so this number also tells us the size of the traing data.

    exclude_zone : int
      It is used to avoid trivial matches. 

    earlly_abaondon : boolean
      Switch to set if we want to have earlly abondon or not. 

    verbose : boolean
      Verbosity mode.

    get_c22_mp_idxs : boolean
      if set true it returns the left_c22_mp indexs. the index of the subsequnce that its distance is used to fill left_c22_mp.

    you_can_do_better : boolean
      if set true, it forces to: 
      for the current subsequnc we compare it atleast with n previouse subsequnces. 
      by default n is set to be eqaul to the size of the training data.  

    dynamic_bsf_update : boolean
      if set false it updates the bsf as follow: 
      if for the current subsequence we had to go back all the way to the begining to find a neighbor for it that
      has distance less than bsf, then we use the value of the 1NN of curr subsequnec to fill left_c22_mp and update bsf. 
      if set true it updates the bsf dynamically:
      if for the current subsequence we had to go back all the way to the begining to find a neighbor for it that
      has distance less than bsf, then we use the value of the 1NN of curr subsequnec to only fill left_c22_mp.
      We use the median of the n recent values in the left_c22_mp that we have made so far to update the bsf.
      the n can be set with by look_back_window_factor.

    look_back_window_factor : int
      it is used to set the look back back window when dynamic_bsf_update is used.
      note - set look_back_window_factor to 0 if you want to use the median of all the values in the left_c22_mp to update bsf.

    weights : array
      the array used to weight features.

      
    Returns
    -------
    left_C22_MP : list (1D array)
      left_c22_mp of the give feature profiles.

    anomaly_idxs : list (1D array)
      indexes of the subsequneces that we had to go back all the way to begining to find a neighbor for them that has distance less than bsf.
      note -  we consider these subsequnse anomaly as we could not find easily/fast a neighbor for them.

    n_skipped : int
      number of times that we had earlly abondon. 
      or in other world the number of subsequnces that we could find a neighbor for them that has distance less than bsf without the need to 
      go back all the way to begining. 

    n_full_back : int
      number of times that we did not have earlly abondon. 
      or in other world the number of subsequnces that we had to go all the way to begining to find a neighbor for them which has distance less bsf.

    bsf_mem : list
      a list containing all the values of bsfs used during the left_c22_mp calculation. 

    left_c22_mp_idxs : list (1D array)
      the left_c22_mp indexs. the index of the subsequnce that its distance is used to fill left_c22_mp.

    bsf_mem_idx: list
      a list containing the index of subsquences that made the algorithm to update the bsf value.

    """

    if weights is not None:
      if np.ndim(weights) == 1:
        not_important_features_idx = []
        for i, w in enumerate(weights):
          if np.isclose(w, 0.):
            not_important_features_idx.append(i)
        important_features_idx = np.arange(weights.shape[0])
        important_features_idx = np.delete(important_features_idx, not_important_features_idx)
        X = np.multiply(X[:,important_features_idx], weights[important_features_idx])
      else:
        X = np.multiply(X, weights)
      
    n_subseq = len(X)
    bsf =  0 #learn_bsf(xx, start_idx, exclude_zone)
    if verbose: 
        print('init bsf',bsf)
    left_C22_MP =np.zeros((X.shape[0],))
    n_skipped = 0
    n_full_back = 0
    anomaly_idxs ={}
    bsf_mem =[]
    bsf_mem_idx = []
    len_train = (n_subseq-start_idx)
    left_c22_mp_idxs = np.zeros((X.shape[0],))
    #print(X.shape)
    if earlly_abaondon:    
        for i in np.arange(start_idx, n_subseq):
            curr = X[i]
            lbsf = np.inf
            lbsf_idx = None
            for j in np.arange(i-1-exclude_zone, -1, -1):
                d = eudis1(curr, X[j])

                if d< lbsf:
                    lbsf = d
                    lbsf_idx = j
                

                if d < bsf and j > 0:
                    n_skipped+=1
                    if you_can_do_better and j>i-(start_idx):
                      continue
                    break

                elif d >= bsf and j > 0:
                    continue 

                elif d >= bsf and j == 0:
                    n_full_back+=1
                    if i>start_idx+ 2 and dynamic_bsf_update: # we check if lenght left_C22_MP that we have  made so far is bigger than 2 then we calculate its median 
                      temp = np.median(left_C22_MP[-start_idx*look_back_window_factor:])
                      if temp < np.min(bsf_mem):#temp<np.median(bsf_mem)
                        pass
                      else:
                        bsf = temp
                    else:
                      bsf =lbsf 
                    anomaly_idxs[i] =j
                    bsf_mem.append(bsf)
                    bsf_mem_idx.append(i)
                    if verbose:
                        print('found anomaly at {}, update bsf to:{}'.format(i,bsf))

              
            left_C22_MP[i]=lbsf
            if get_c22_mp_idxs:
                left_c22_mp_idxs[i] = lbsf_idx 


    else:
        for i in np.arange(start_idx, n_subseq):
            curr = X[i]
            lbsf = np.inf
            for j in np.arange(i-1-exclude_zone, -1, -1):
                d = eudis1(curr, X[j])
                if d < lbsf:
                    lbsf=d
                    if get_c22_mp_idxs:
                        left_c22_mp_idxs[i]=j
                    
            left_C22_MP[i]=lbsf

    anomaly_idxs = np.asarray([[k,v] for k, v in anomaly_idxs.items()])
    return left_C22_MP, anomaly_idxs, n_skipped , n_full_back, bsf_mem, left_c22_mp_idxs, bsf_mem_idx

### C22MP end
def scoring_function(loc, anomaly_start, anomaly_end):
    L = anomaly_end - anomaly_start + 1
    lower_bound = min(anomaly_start-L , anomaly_start-100)
    upper_bound = max(anomaly_end+L, anomaly_end+100)
    if lower_bound < loc and loc < upper_bound:
        return True
    else:
        return False
    
sys.stdout = Logger('./log_manually_selected_m.log', sys.stdout)
#sys.stderr = Logger('./log.log', sys.stderr)

# Get M for 250 dataset
M = loadmat("DAMP_M.mat")['M'][0]
#M = loadmat("period.mat")['results'][0]
currentPath = os.getcwd().replace('\\','/')
# get file name for 1 Data collected from a healthy bearing
path = currentPath+"/dataset"
files = os.listdir(path)
files = sorted(files)
success_num = 0
success_vec = []
left_c22_mp_times = 0
total_time = 0
i = 0
for file in files:
    
    print(file)
    tmp = file.split("_")
    if len(tmp) < 7:
        continue
    i = i+1
    file_index = int(tmp[0])  
    start_pos = int(tmp[4])
    anomaly_start = int(tmp[5])
    anomaly_end = int(tmp[6].split(".")[0])
   

    
    # Load data
    ts = np.array(pd.read_csv(path+"/"+files[file_index]))
    
    print("Time series length:",len(ts))
    SubsequenceLength = M[file_index-1]
    #SubsequenceLength = (anomaly_end+10)-(anomaly_start-10)
    print("Cycle length:", SubsequenceLength)
    
    
    # Run C22MP
    # Calculate C22MP
    cal_addtional_features = False # set to true if you want to calculate addition features : min, std
    additinal_feature_names =['min', 'std']
    step = 1 # step size to take when calculating catchh 22 features by moving window 
    win = SubsequenceLength # subsequence size
    
    start = time.time()
    feature_profiles = cal_feature_profiles(ts,win, step = step ,cal_addtional_features = cal_addtional_features)
    end = time.time()
    AA = end-start
    print('feature profiles calculation time : {:.2f}s'.format(AA))
    
    ## uncomment and run this if you wish to have same order for the features as the one for MATLAB catch 22 library
    # feature_profiles =reorder(feature_profiles, id_mapping) 
    
    start = time.time()
    feature_profiles = ffill(feature_profiles)
    feature_profiles = bfill(feature_profiles)
    end = time.time()
    BB = end-start
    print('filling None values time : {:.2f}s'.format(BB))
    
    
    
    start = time.time()
    # feature_profiles_normalized = StandardScaler().fit_transform(feature_profiles)
    feature_profiles_normalized = MinMaxScaler().fit_transform(feature_profiles)
    end = time.time()
    CC = end-start
    print('data scaling time : {:.2f}s'.format(CC))
    
    
    
    exclude_zone = win//2
    start_idx = start_pos
    earlly_abaondon = True
    get_c22_mp_idxs = False
    dynamic_bsf_update = False
    look_back_window_factor = 2
    you_can_do_bMetter=False
    verbose=False
    

    '''
    C22MP - weighted features
    '''  

    start = time.time()
    weights = None #get_weights(feature_profiles_normalized, ts[:ts.shape[0]-win+1])
    end = time.time()
    DD = end-start
    print('Feature weight calculation time : {:.2f}s'.format(DD))
    
    start = time.time()
    left_C22_MP, anomaly_idxs, n_skipped , n_full_back, bsf_mem, left_c22_mp_idxs, bsf_mem_idx = left_c22_mp(
                                                                                                          feature_profiles_normalized,
                                                                                                          start_idx, 
                                                                                                          exclude_zone, 
                                                                                                          earlly_abaondon,
                                                                                                          get_c22_mp_idxs =get_c22_mp_idxs,
                                                                                                          dynamic_bsf_update=dynamic_bsf_update,
                                                                                                          look_back_window_factor = look_back_window_factor,
                                                                                                          weights=weights
                                                                                                          )
    left_C22_MP = np.pad(left_C22_MP,(win//2,0), 'constant')
    end = time.time()
    left_c22_mp_time = end-start
    print('left_c22_mp calculation time : {:.2f}s'.format(left_c22_mp_time))
    
    loc = np.where(left_C22_MP==np.max(left_C22_MP))[0][0]
    print("Predicted anomaly location:",loc)
    
    if scoring_function(loc, anomaly_start, anomaly_end): 
        success_num = success_num + 1
        success_vec.append(1)
        print('Success')
    else:
        success_vec.append(0)
        print('Fail')
    
    print("Total number of successes for",i,"runs:",success_num)
    left_c22_mp_times = left_c22_mp_times + left_c22_mp_time 
    print("Total left_c22_mp calculation time",left_c22_mp_times)
    total_time = total_time + left_c22_mp_time + AA + BB + CC + DD
    print("Total time",total_time )
    print("----------------------------------------------\n")
    #scipy.io.savemat('vec_c22mp_m_subset.mat',mdict={'success_vec': success_vec})