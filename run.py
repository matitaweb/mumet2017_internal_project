from detect_pedestrian import detect_pedestrian
from video_builder import build_video
from test_model_for_market1501 import add_reid_data_annotation
from test_model_for_market1501 import reload_model
import pandas as pd
import os

pd.set_option('display.width', 1000)

import datetime
import logging
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
run_filelogger = logging.getLogger('run')
run_filelogger.setLevel(logging.DEBUG)
run_fh = logging.FileHandler('log/run.log')
run_fh.setLevel(logging.DEBUG)
run_filelogger.addHandler(run_fh) 





def call_detect(video_name, snapshot_rate = 1, winStride=(4, 4), padding=(2, 2), scale=1.05, hitThreshold=1, finalThreshold=2.0, useMeanshiftGrouping=False, tkline_size = 2, overlapThresh = 0.55, resize = (64, 128), isVideo=True):
	
	input_video_path = 'video_data/'+video_name 
	if isVideo:
	    input_video_path = 'video_data/'+video_name+'.mp4' 
	
	output_info_path = 'video_result/'+video_name+'/frame_detection' 
	detection_box_dir = output_info_path+'/frame_detection_crop'
	frame_debug_dir = output_info_path + '/frame_detection_debug'
	
	t1 = datetime.datetime.now()
	detect_info_df = detect_pedestrian(input_video_path, snapshot_rate, output_info_path, detection_box_dir, 
			frame_debug_dir, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping, tkline_size, overlapThresh, resize, isVideo)
	
	run_filelogger.debug("[DETECT PEDESTIAN] video: %s, shape: %s, %f sec", input_video_path, str(detect_info_df.shape), (datetime.datetime.now()-t1).total_seconds())
	return detect_info_df
	
	
def call_build_video(video_name, detect_info_df, slack_frame_max=12, isVideo=True):

	input_video_path = 'video_data/'+video_name 
	if isVideo:
	    input_video_path = 'video_data/'+video_name+'.mp4' 
	
	output_info_path = 'video_result/'+video_name+'/video_builder' 
	out_video_path = output_info_path +'/video_builder_out'
	frame_video_debug_path = output_info_path+'/video_builder_debug'
	
	
	build_video(input_video_path, output_info_path, out_video_path, frame_video_debug_path, detect_info_df, slack_frame_max=slack_frame_max, isVideo=isVideo)
	
if(__name__ == '__main__'):
    
    
    weight_path = 'weights/weights_on_market1501_0_0_0.h5'
    model = reload_model(weight_path)

    """
    video_name = 'ex_01'
    detect_info_df = call_detect(video_name)
    detect_info_df = pd.read_csv('video_result/'+video_name+'/frame_detection' + '/frame_detection_annotations.csv')
    dictionary_test_path = ['video_data/ex_01_dict/andrea.jpg', 'video_data/ex_01_dict/sara.jpg'] 
    df_annotated = add_reid_data_annotation(model, detect_info_df, dictionary_test_path, 'video_result/'+video_name+'/video_builder')
    #print (df_annotated.head())
    call_build_video(video_name, df_annotated)
    logger.debug("DONE VIDEO %s", video_name)
    """
    
    
    video_name = 'ex_02'
    #detect_info_df = call_detect(video_name, snapshot_rate = 1, winStride=(2, 2), padding=(4, 4), scale=1.5, hitThreshold=0, finalThreshold=3.0, useMeanshiftGrouping=False, isVideo=True)
    detect_info_df = pd.read_csv('video_result/'+video_name+'/frame_detection' + '/frame_detection_annotations.csv')
    dictionary_test_path = ['video_data/ex_02_dict/leo.jpg', 'video_data/ex_02_dict/dave.jpg'] 
    df_annotated = add_reid_data_annotation(model, detect_info_df, dictionary_test_path, 'video_result/'+video_name+'/video_builder')
    #df_annotated = pd.read_csv( 'video_result/'+video_name+'/video_builder/frame_detection_annotations.csv')
    #df_annotated = df_annotated.drop('IDX', 1)
    call_build_video(video_name, df_annotated)
    logger.warn("DONE VIDEO %s", video_name)
    
    
    """
    video_name = 'ex_03'
    #detect_info_df = call_detect(video_name, snapshot_rate = 3, winStride=(2, 2), padding=(4, 4), scale=1.5, hitThreshold=0, finalThreshold=3.0, useMeanshiftGrouping=False, isVideo=True)
    detect_info_df = pd.read_csv('video_result/'+video_name+'/frame_detection' + '/frame_detection_annotations.csv')
    dictionary_test_path = ['video_data/ex_03_dict/leonard.jpg', 'video_data/ex_03_dict/nick.jpg'] 
    df_annotated = add_reid_data_annotation(model, detect_info_df, dictionary_test_path, 'video_result/'+video_name+'/video_builder')
    call_build_video(video_name, df_annotated)
    logger.debug("DONE VIDEO %s", video_name)


    video_name = 'ex_06'
    #detect_info_df = call_detect(video_name, snapshot_rate = 1, isVideo=True)
    detect_info_df = pd.read_csv('video_result/'+video_name+'/frame_detection' + '/frame_detection_annotations.csv')
    dictionary_test_path = ['video_data/ex_03_dict/leonard.jpg', 'video_data/ex_03_dict/nick.jpg'] 
    df_annotated = add_reid_data_annotation(model, detect_info_df, dictionary_test_path, 'video_result/'+video_name+'/video_builder')
    call_build_video(video_name, df_annotated)
    logger.debug("DONE VIDEO %s", video_name)

    
    video_name = 'cvpr10_tud_stadtmitte'
    #detect_info_df = call_detect(video_name, snapshot_rate = 1, winStride=(2, 2), padding=(4, 4), scale=1.5, hitThreshold=0, finalThreshold=3.0, useMeanshiftGrouping=False, isVideo=False)
    detect_info_df = pd.read_csv('video_result/'+video_name+'/frame_detection' + '/frame_detection_annotations.csv')
    dictionary_test_path = ['video_data/cvpr10_tud_stadtmitte_dict/albert.jpg', 'video_data/cvpr10_tud_stadtmitte_dict/anna.jpg', 'video_data/cvpr10_tud_stadtmitte_dict/bob.jpg', 'video_data/cvpr10_tud_stadtmitte_dict/carla.jpg'] 
    df_annotated = add_reid_data_annotation(model, detect_info_df, dictionary_test_path, 'video_result/'+video_name+'/video_builder')
    call_build_video(video_name, df_annotated, slack_frame_max=-1, isVideo=False)
    logger.debug("DONE VIDEO %s", video_name)
    
    
    video_name = '3depes_seq_1'
    #detect_info_df = call_detect(video_name, snapshot_rate = 1, winStride=(2, 2), padding=(4, 4), scale=1.5, hitThreshold=0, finalThreshold=3.0, useMeanshiftGrouping=False, isVideo=False)
    detect_info_df = pd.read_csv('video_result/'+video_name+'/frame_detection' + '/frame_detection_annotations.csv')
    dictionary_test_path = ['video_data/3depes_seq_1_dict/alf.jpg', 'video_data/3depes_seq_1_dict/andy.jpg', 'video_data/3depes_seq_1_dict/bob.jpg', 'video_data/3depes_seq_1_dict/rob.jpg', 'video_data/3depes_seq_1_dict/sophie.jpg'] 
    df_annotated = add_reid_data_annotation(model, detect_info_df, dictionary_test_path, 'video_result/'+video_name+'/video_builder')
    call_build_video(video_name, df_annotated, slack_frame_max=-1, isVideo=False)
    logger.debug("DONE VIDEO %s", video_name)
    """
    
    """
    video_name = '3depes_seq_3'
    detect_info_df = call_detect(video_name, snapshot_rate = 1, winStride=(2, 2), padding=(4, 4), scale=1.5, hitThreshold=0, finalThreshold=3.0, useMeanshiftGrouping=False, isVideo=False)
    quit()
    detect_info_df = pd.read_csv('video_result/'+video_name+'/frame_detection' + '/frame_detection_annotations.csv')
    dictionary_test_path = ['video_data/3depes_seq_1_dict/alf.jpg', 'video_data/3depes_seq_1_dict/andy.jpg', 'video_data/3depes_seq_1_dict/bob.jpg', 'video_data/3depes_seq_1_dict/rob.jpg', 'video_data/3depes_seq_1_dict/sophie.jpg'] 
    df_annotated = add_reid_data_annotation(model, detect_info_df, dictionary_test_path, 'video_result/'+video_name+'/video_builder')
    call_build_video(video_name, df_annotated, slack_frame_max=-1, isVideo=False)
    logger.debug("DONE VIDEO %s", video_name)
    """