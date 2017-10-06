# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
import argparse
import cv2
from PIL import Image
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from utils import mkdir_p
from utils import non_max_suppression_fast
import errno    
import datetime



import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
filelogger = logging.getLogger('detect_pedestrian')
filelogger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/detect_pedestrian.log')
fh.setLevel(logging.DEBUG)
filelogger.addHandler(fh) 


def resize(img, basewidth, hsize):

	img = img.resize((basewidth,hsize), Image.ANTIALIAS)
	img.save('sompic.jpg') 
	
def analize_frame(frame, hog, overlapThresh, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, resize, frame_debug_dir, detection_box_dir, ii, detect_info_mx):
	frame_gray = frame.copy()
	frame_copy = frame.copy()

	# detect people in the image 
	(rects, weights) = hog.detectMultiScale(frame_gray, winStride=winStride, padding=padding, 
						scale=scale, hitThreshold=hitThreshold, finalThreshold=finalThreshold, useMeanshiftGrouping=useMeanshiftGrouping)

	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	
	pick = non_max_suppression_fast(rects, overlapThresh)
	#pick = rects

	frame_file_path = frame_debug_dir + '/frame_'+str(ii)+'.jpg'
	
	# draw the final bounding boxes and crop
	jj=0
	for (xA, yA, xB, yB) in pick:
		
		frame_num = ii
		crop_file_path = detection_box_dir+'/frame_'+str(ii)+'_crop_'+str(jj)+'.jpg'
		crop_num = jj
		id_tag = ''
		info =''
		
		crop=frame[(yA+tkline_size):(yB-tkline_size), (xA+tkline_size):(xB-tkline_size)]
		resized_crop = cv2.resize(crop, resize) 
		
		cv2.imwrite(crop_file_path, resized_crop)
		cv2.rectangle(frame_copy, (xA, yA), (xB, yB), (0, 255, 0), tkline_size)
		
		detect_row = [frame_num, frame_file_path, crop_file_path, crop_num, xA, yA, xB, yB, id_tag, info]
		detect_info_mx.append(detect_row)
		
		jj = jj+1
		
		
	# show some information on the number of bounding boxes
	logging.debug("[INFO] frame {}: {} boxes, {} supressed".format(str(ii), len(rects), len(pick)))
		
	cv2.imwrite(frame_file_path, frame_copy)
	
def analize_frame_dir(input_video_path, overlapThresh, snapshot_rate, hog, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, resize, frame_debug_dir, detection_box_dir):
	flist = [p for p in pathlib.Path(input_video_path).iterdir() if p.is_file()]
	ii=0
	detect_info_mx = []
	for f in flist:
		print(str(f))
		#quit()
		if(ii%snapshot_rate != 0):
			ii = ii+1
			continue
		
		frame = cv2.imread(str(f))
		
		analize_frame(frame, hog, overlapThresh, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, resize, frame_debug_dir, detection_box_dir, ii, detect_info_mx)

		ii = ii+1
		
	return detect_info_mx
	
def analize_frame_video(input_video_path, overlapThresh, snapshot_rate, hog, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, resize, frame_debug_dir, detection_box_dir):
	detect_info_mx = []
	# loop over the video
	cap = cv2.VideoCapture(input_video_path)
	
	ii=0
	while cap.isOpened():
		ret,frame = cap.read()
		if(ret == 0):
			break
		
		if(ii%snapshot_rate != 0):
			ii = ii+1
			continue
		
		analize_frame(frame, hog, overlapThresh, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, resize, frame_debug_dir, detection_box_dir, ii, detect_info_mx)

		ii = ii+1
		
	cap.release()
	return detect_info_mx

def detect_pedestrian(input_video_path, snapshot_rate, output_info_path, detection_box_dir, frame_debug_dir, 
						winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, overlapThresh, resize, isVideo=True):
	
	#create dir to save data
	mkdir_p(output_info_path)
	mkdir_p(detection_box_dir)
	mkdir_p(frame_debug_dir)
	
	# initialize the HOG descriptor/person detector http://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	if isVideo:
		detect_info_mx = analize_frame_video(input_video_path, overlapThresh, snapshot_rate, hog, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, resize, frame_debug_dir, detection_box_dir)
	else:
		detect_info_mx = analize_frame_dir(input_video_path, overlapThresh, snapshot_rate, hog, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, resize, frame_debug_dir, detection_box_dir)
		
	
	
	detect_info_df=pd.DataFrame(detect_info_mx, columns=['FRAME', 'FRAME_FILE_PATH', 'CROP_FILE_PATH', 'CROP_NUM', 'xA', 'yA', 'xB', 'yB', 'ID', 'INFO'])
	detect_info_df.to_csv(output_info_path + '/frame_detection_annotations.csv')
	return detect_info_df


if(__name__ == '__main__'):

	#input_video_path = 'video_data/WalkByShop1cor.mpg' 
	video_name = 'ex_01'
	input_video_path = 'video_data/'+video_name+'.mp4' 
	#input_video_path = 'video_data/2.mp4'

	output_info_path = 'video_result/'+video_name+'/frame_detection' 
	detection_box_dir = output_info_path+'/frame_detection_crop'
	frame_debug_dir = output_info_path + '/frame_detection_debug'
	
	#params
	overlap_thresh=0.55
	snapshot_rate = 24

	
	winStride=(4, 4)
	padding=(2, 2)
	scale=1.05
	hitThreshold=1
	finalThreshold=2.0
	useMeanshiftGrouping=False
	
	tkline_size = 2
	overlapThresh = 0.55
	resize = (64, 128)
	
	t1 = datetime.datetime.now()
	detect_info_df = detect_pedestrian(input_video_path, snapshot_rate, output_info_path, detection_box_dir, 
			frame_debug_dir, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, overlapThresh, resize)
	
	filelogger.debug("[DETECT PEDESTIAN] video: %s, shape: %s, %f sec", input_video_path, str(detect_info_df.shape), (datetime.datetime.now()-t1).total_seconds())
	
	
	
	



