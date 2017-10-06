import pandas as pd
import os
import numpy as np
import csv

pd.set_option('display.width', 1000)

import datetime
import logging
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
acc_filelogger = logging.getLogger('accuracy')
acc_filelogger.setLevel(logging.DEBUG)
acc_fh = logging.FileHandler('log/accuracy2.log')
acc_fh.setLevel(logging.DEBUG)
acc_filelogger.addHandler(acc_fh) 


def log_accuracy(name, detect_info_df):
    detect_info_nparray = detect_info_df.values
    true_match=0
    nones = 0
    nones_similarity = []
    false_match=0
    acc_filelogger.debug("******* TEST %s ******", name)
    for row in detect_info_nparray:
        gt = row[10]
        r = str(row[11])
        rname = r.split(' ')[0]
        rsim = float(r.split(' ')[1])
        if gt == rname:
            true_match=true_match+1
        elif gt == 'none' :
            nones = nones+1
            nones_similarity.append(rsim)
        else:
           false_match=false_match+1
           
        #acc_filelogger.debug('row: %s, gr: %s <-> %s ', row[0], gt, rname)
    
    tot = len(detect_info_nparray)
    mean_nones_sim = 0
    mean_nones_sim_max = 0
    mean_nones_sim_min = 0
    if(len(nones_similarity) > 0):
        nones_similarity = np.array(nones_similarity)
        mean_nones_sim = nones_similarity.mean()
        mean_nones_sim_max = np.amax(nones_similarity)
        mean_nones_sim_min = np.amin(nones_similarity)
    acc_filelogger.debug('tot: %i, true_match: %i, nones: %i, false_match: %i', tot, true_match, nones, false_match)
    acc_filelogger.debug('true_match: %f, nones: %f, false_match: %f', true_match/tot, nones/tot, false_match/tot)
    acc_filelogger.debug('true_match_without nones: %f, tot - nones: %f', true_match/(tot - nones), (tot - nones))
    acc_filelogger.debug('mean_nones_sim: %f, mean_nones_sim_max: %f, mean_nones_sim_min: %f',  mean_nones_sim, mean_nones_sim_max, mean_nones_sim_min)
    
    

if(__name__ == '__main__'):
    
    video_name = 'cvpr10_tud_stadtmitte'
    detect_info_df = pd.read_csv('video_data/cvpr10_tud_stadtmitte_dict/cvpr10_tud_stadtmitte_gt.csv')
    print (detect_info_df.head())
    log_accuracy(video_name, detect_info_df)
    
    """
    video_name = 'ex_02'
    detect_info_df = pd.read_csv('video_data/ex_02_dict/ex_02_gt.csv')
    log_accuracy(video_name, detect_info_df)
    """
    
    """
    detect_info_df = pd.read_csv('video_data/cvpr10_tud_stadtmitte_dict/tud_stadtmitte_gt.txt')
    print (detect_info_df.values)
    
    counter={}
    for r in detect_info_df.values:
        f = r[0]
        a = "FRAME_" +str(f)
        if a in counter:
            counter[a]=counter[a]+1
        else:
            counter[a] = 1
    print(counter)
    #res = pd.DataFrame.from_dict(counter)
    #with open('mycsvfile.csv','wb') as fi:
    #w = csv.writer(fi)
    for key, value in counter.items():
        print(value)      
            
    #print (res.head())    
    """
    

    

    