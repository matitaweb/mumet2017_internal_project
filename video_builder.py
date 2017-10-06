# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
import argparse
import cv2
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import errno
from utils import mkdir_p
import sys

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
filelogger = logging.getLogger('test_result')
filelogger.setLevel(logging.DEBUG)
fh = logging.FileHandler('test_result.log')
fh.setLevel(logging.DEBUG)
filelogger.addHandler(fh) 


def put_marker_box(frame, xA, yA, xB, yB, label_text, font):
    cv2.rectangle(frame,(xA-1,yA-18),(xB+1,yA),(0,0,0),-1)
    cv2.rectangle(frame,(xA,yA),(xB,yB),(0,0,0),2)
    cv2.putText(frame,label_text,(xA+3,yA-6), font, 0.4, (0,255,255),1,cv2.LINE_AA)

def put_marker_box_list(frame, font, detect_info_nparray_frame):
    for f in detect_info_nparray_frame:
        xA = f[5]
        yA = f[6]
        xB = f[7]
        yB = f[8]
        label_text = str(f[9]) #label
        put_marker_box(frame, xA, yA, xB, yB, label_text, font)
    
def rebuild_frame_video(input_video_path, out_video_path, detect_info_nparray, slack_frame_max, frame_video_debug_path):
    
    #fourcc = cv2.VideoWriter_fourcc('MJPG')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')

    out = None
    size = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # loop over the video
    cap = cv2.VideoCapture(input_video_path)
    
    detect_info_nparray_frame_bkp = None
    slack_frame_counter = 0;
    
    ii=0
    while cap.isOpened():
    	ret,frame = cap.read()
    	if(ret == 0):
    		break
    	
    	if out is None:
            if size is None:
                size = frame.shape[1], frame.shape[0]
            out = cv2.VideoWriter(out_video_path + '/output.mp4', fourcc, 23.0, size, True)
    	
    	detect_info_nparray_frame = detect_info_nparray [(ii==detect_info_nparray[:,1])]
    	
    	if (detect_info_nparray_frame is None or len(detect_info_nparray_frame) == 0):
            if(detect_info_nparray_frame_bkp is not None and slack_frame_counter < slack_frame_max):
                put_marker_box_list(frame, font, detect_info_nparray_frame_bkp)
                slack_frame_counter=slack_frame_counter+1
                #print(slack_frame_counter)
    	else:
            put_marker_box_list(frame, font, detect_info_nparray_frame) 
            detect_info_nparray_frame_bkp = detect_info_nparray_frame
            slack_frame_counter=0

    	cv2.imwrite(frame_video_debug_path + '/' + str(ii)+'.jpg', frame)
    	out.write(frame)
    	ii = ii+1
    
    cap.release()
    out.release()
    
def rebuild_frame_dir(input_video_path, out_video_path, detect_info_nparray, slack_frame_max, frame_video_debug_path):
    
    #fourcc = cv2.VideoWriter_fourcc('MJPG')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')

    out = None
    size = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # loop over the video
    cap = cv2.VideoCapture(input_video_path)
    
    detect_info_nparray_frame_bkp = None
    slack_frame_counter = 0;
    
    flist = [p for p in pathlib.Path(input_video_path).iterdir() if p.is_file()]
    ii=0
    for f in flist:
        frame = cv2.imread(str(f))
        if out is None:
            if size is None:
                size = frame.shape[1], frame.shape[0]
            out = cv2.VideoWriter(out_video_path + '/output.mp4', fourcc, 23.0, size, True)
        detect_info_nparray_frame = detect_info_nparray [(ii==detect_info_nparray[:,1])]
        if (detect_info_nparray_frame is None or len(detect_info_nparray_frame) == 0) :
    	    if(detect_info_nparray_frame_bkp is not None and slack_frame_counter < slack_frame_max):
                put_marker_box_list(frame, font, detect_info_nparray_frame_bkp)
                slack_frame_counter=slack_frame_counter+1
        else:
            put_marker_box_list(frame, font, detect_info_nparray_frame) 
            detect_info_nparray_frame_bkp = detect_info_nparray_frame
            slack_frame_counter=0
        cv2.imwrite(frame_video_debug_path + '/' + str(ii)+'.jpg', frame)
        out.write(frame)
        ii = ii+1
    
    cap.release()
    out.release()

def build_video(input_video_path, output_info_path, out_video_path, frame_video_debug_path, detect_info_df, slack_frame_max=100,isVideo=True):

    mkdir_p(out_video_path)
    mkdir_p(frame_video_debug_path)
	
    detect_info_nparray = detect_info_df.values
 
    if isVideo:
        rebuild_frame_video(input_video_path, out_video_path, detect_info_nparray, slack_frame_max, frame_video_debug_path)
    else:
        rebuild_frame_dir(input_video_path, out_video_path, detect_info_nparray, slack_frame_max, frame_video_debug_path)
    
    
    


if(__name__ == '__main__'):


	video_name = 'ex_01'
	input_video_path = 'video_data/'+video_name+'.mp4' 	
	
	output_info_path = 'video_result'
	output_info_path = 'video_result/'+video_name+'/video_builder' 
	out_video_path = output_info_path +'/video_builder_out'
	frame_video_debug_path = output_info_path+'/video_builder_debug'
	
	
	detect_info_df = pd.read_csv(output_info_path + '/detect_info.csv')
	build_video(input_video_path, output_info_path, out_video_path, frame_video_debug_path, detect_info_df)