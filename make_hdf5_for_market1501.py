# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import h5py
import numpy as np
from PIL import Image

def get_list_positive_index_market1501(train_or_test, data_dir):
    path_list = get_image_path_list(train_or_test, data_dir)
    print (data_dir+ ' path_list -> '+str(len(path_list)))
    index = []
    i = 0
    while i < len(path_list):
        j = i + 1
        while j < len(path_list) and path_list[j][0:4] == path_list[i][0:4]:
            if path_list[j][6] != path_list[i][6]:
                index.append([path_list[i],path_list[j]])
                index.append([path_list[j],path_list[i]])
                #print (index)
                #quit()
                #break
                #print(len(index))
            j += 1
        #break
        i += 1
    #print ('transforming the list to the numpy array......')
    index = np.array(index)
    #print ('shuffling the numpy array......')
    np.random.shuffle(index)
    return index

def get_list_negative_index_market1501(train_or_test, data_dir):
    path_list = get_image_path_list(train_or_test, data_dir)
    print (data_dir+ ' path_list -> '+str(len(path_list)))
    index = []
    i = 0
    while i < len(path_list):
        j = i + 1
        while j < len(path_list) and path_list[j][0:4] != path_list[i][0:4]:
            if path_list[j][6] != path_list[i][6]:
                index.append([path_list[i],path_list[j]])
                index.append([path_list[j],path_list[i]])
                #print (index)
                #quit()
                #break
                #print(len(index))
            j += 1
        #break
        i += 1
    #print ('transforming the list to the numpy array......')
    index = np.array(index)
    #print ('shuffling the numpy array......')
    np.random.shuffle(index)
    return index
    
def make_positive_index_market1501(train_or_test, data_dir):
    f = h5py.File('market1501_positive_index.h5')
    index = get_list_positive_index_market1501(train_or_test, data_dir)
    print ('storing the array to HDF5 file......')
    f.create_dataset(train_or_test,data = index)

def get_image_path_list(train_or_test, data_dir):
    if train_or_test == 'train':
        folder_path = data_dir + '/bounding_box_train'
    elif train_or_test == 'test':
        folder_path = data_dir +  '/bounding_box_test'
    elif train_or_test == 'query':
        folder_path = data_dir +  '/query'
    print (folder_path)
    assert os.path.isdir(folder_path)
    if train_or_test == 'train' or train_or_test == 'query':
        return sorted(os.listdir(folder_path))
    elif train_or_test == 'test':
        return sorted(os.listdir(folder_path))[6617:]


"""
# FOR TESTING
if __name__ == '__main__':

    #make_positive_index_market1501('train', 'dataset/market-1501')
    make_positive_index_market1501('test', 'dataset/market-1501')
"""