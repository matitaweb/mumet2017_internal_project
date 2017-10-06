# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1217)
import h5py
import tensorflow as tf
#tf.python.control_flow_ops = tf
from PIL import Image
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing import image as pre_image
from make_hdf5_for_market1501 import get_image_path_list, get_list_positive_index_market1501, get_list_negative_index_market1501
from model_for_market1501 import model_def, compiler_def
import pandas as pd
from utils import mkdir_p
import os

import datetime
import logging
logging.basicConfig(level=logging.ERROR)
test_model_logger = logging.getLogger(__name__)
test_model_filelogger = logging.getLogger('test_model')
test_model_filelogger.setLevel(logging.DEBUG)
test_model_fh = logging.FileHandler('log/test_model.log')
test_model_fh.setLevel(logging.DEBUG)
test_model_filelogger.addHandler(test_model_fh) 

def random_select_pos(f, data_dir, num):
    
    indexs = list(np.random.choice(range(f.shape[0]),num))
    A = []
    B = []
    for index in indexs:
        path1 = f[index,0]
        path2 = f[index,1]
        #print (str(path1[0:7]) + "(" + path1 + ") - " + str(path2[0:7] + "(" + path2 + ")"))
        A.append(np.array(Image.open(data_dir + '/bounding_box_test/' + path1)))
        B.append(np.array(Image.open(data_dir + '/bounding_box_test/' + path2)))
        
    return np.array(A)/255.,np.array(B)/255.
    
def single_test(model, path1, path2):
    
    A = []
    B = []
    #A.append(np.array(Image.open(path1).convert('RGB')))
    #B.append(np.array(Image.open(path2).convert('RGB')))
    A.append(np.array(Image.open(path1))[:, :, :3]) #remove alpha
    B.append(np.array(Image.open(path2))[:, :, :3]) #remove alpha
    #print (A[0].shape)
    #print (B[0].shape)
    A = np.array(A)/255.
    B = np.array(B)/255.
    
    return model.predict([A,B],batch_size = 100)[:,1]


def reload_model(weight_path):
    
    test_model_logger.debug('default dim order is:' + K.image_dim_ordering())
    model = model_def()
    test_model_logger.debug ('model definition done.')
    model = compiler_def(model)
    test_model_logger.debug ('model compile done.')
    
    model.load_weights(weight_path)
    return model

def add_reid_data_annotation(model, detect_info_df, dictionary_test_path, out_video_path):
    columns=['FRAME', 'FRAME_FILE_PATH', 'CROP_FILE_PATH', 'CROP_NUM', 'xA', 'yA', 'xB', 'yB', 'ID', 'INFO']
    
    detect_info_nparray = detect_info_df.values
    for idx, row  in detect_info_df.iterrows():
        
        #detect_row = [frame_num, frame_file_path, crop_file_path, crop_num, xA, yA, xB, yB, id_tag, info]
        better_pred=None
        better_dict=None
        prediction_info = ""
        for d in dictionary_test_path:
            #test_model_logger.error('compare ' + str(detect_info_nparray[idx][3]))
            pred = single_test(model, detect_info_nparray[idx][3], d)
            prediction_info+= str(os.path.splitext(os.path.basename(d))[0]) + ":" + " {0:.5f}".format(pred[0])+", "
            if better_pred is None or better_pred < pred:
                better_pred = pred[0]
                better_dict =os.path.splitext(os.path.basename(d))[0]
                
        detect_info_nparray[idx][9] = str(better_dict) + " {0:.2f}".format(better_pred)
        detect_info_nparray[idx][10] = prediction_info
        #print(str(detect_info_nparray[idx][1]) + " --> " +  str(detect_info_nparray[idx][9]))

    result = pd.DataFrame(detect_info_nparray, index=detect_info_df.index)
    mkdir_p(out_video_path)
    result.to_csv(out_video_path + '/frame_detection_annotations.csv')
    return result



"""
only for testing
"""
if __name__ == '__main__':
    
    weight_path = 'weights/weights_on_market1501_0_0_0.h5'
    model = reload_model(weight_path)
    
    data_dir = 'dataset/market-1501'
    training_set_positive_index_market1501 = get_list_positive_index_market1501('train', data_dir)
    test_set_positive_index_market1501 = get_list_positive_index_market1501('test', data_dir)
    test_set_negative_index_market1501 = get_list_negative_index_market1501('test', data_dir)
    num = 2000
    
    A,B = random_select_pos(test_set_positive_index_market1501, data_dir, num)
    pred_pos =  model.predict([A,B],batch_size = 100)[:,1]
    print("pred_pos: " + str(np.mean(pred_pos)))
    
    A,B = random_select_pos(test_set_negative_index_market1501, data_dir, num)
    pred_neg =  model.predict([A,B],batch_size = 100)[:,1]
    print("pred_neg: " + str(np.mean(pred_neg)))
    
    
    path1 = 'dataset/market-1501/bounding_box_test/0194_c6s1_059901_04.jpg'
    path2 = 'dataset/market-1501/bounding_box_test/1148_c4s5_025904_02.jpg'
    pred = single_test(model, path1, path2)
    print("pedestrian diversi " +  str(pred))
    
    dictionary_test = ['01.jpg', '02.jpg'] 
    path_test = 'dataset/pedestrian-reidentification-model-trainer/test/01_002.jpg'
    path_dict = 'dataset/pedestrian-reidentification-model-trainer/dictionary/'
    test_set = ['01_001.jpg' , '01_002.jpg',  '01_003.jpg',  '02_001.jpg',  '02_002.jpg',  '02_003.jpg',  '99_001.jpg',  '99_002.jpg',  '99_003.jpg']
    
    for t in test_set:
        path_img_test = 'dataset/pedestrian-reidentification-model-trainer/test/'+ t
        for d in dictionary_test:
            path_img_dict = 'dataset/pedestrian-reidentification-model-trainer/dictionary/'+d
            pred = single_test(model, path_img_test, path_img_dict)
            print(t,  d,  str(pred), type(pred[0]))
        print ("----------------------------")
    
    
    
    
    
    
    
