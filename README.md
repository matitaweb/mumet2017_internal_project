# RE-IDENTIFICATION OF PEOPLE IN OUTDOOR SURVEILLANCE VIDEOS (#mumet2017_internal_project)
  
>"The system is capable of reidentify a given (query) person. 
>The process annotate videos detecting pedestrian using a simple HOG detector and than with a "Siamese Network" try to reidentify people by similarity ranking."  
  
 


### FIRST SOME RESULTS...
A fixed camera and 2 people to recognize....
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/RipYW9D15fs/0.jpg)](http://www.youtube.com/watch?v=RipYW9D15fs)

### SOME PROJECT ASSUMPTIONS....
To complete the project I need to consider some constraints as:

- Static camera or simple move as pan or tilt.
- We have a set of pedestrian image for the reidentification.
- Good light condition
- No crowd or animals and moving objects.


### ARCHITECTURE
The purpose is to extract pedestrian images from a video frame and recognize their identity, 
to resolve this problem first we try to all detetect pedestrian in a frame, than we try to reidentify every detected person.


###### Detection phase
The first phase read a input video (or a frame sequence) and produce a croped people image and annotation using a simple HOG detection.

###### ReID phase
The detection phase output is the input for a Trained Siamese Net that try to reidentify pedestrian comparing 
all croped pedestrian image with a database with all known identity.<br/>
Now the output is the identity similarity rank added to the croped image with the person to be re-identified.


###### Video annotation phase
After the reidentification we take all people identity and position information computed for all frames, 
we use them to enrich the input video adding labeled bouding box that follows every detected pedestrian.


[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/architettura_soluzione.png)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_main)

<br/>
<br/>

### PEDESTRIAN DETECTION
We take a simple HOG detection that allow to crop a fram removing removing most of the background and reducing the image size passed at next phase.
For every detected pedestrian we produce a image containing the person and some annotation (eg. boundig box position and size) in a csv format.

[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/detection_phase.png)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_main)

Built-in feature used:
```sh
Python 3
OpenCV 3.2
Panda 0.20
```

###### STRENGHT
1. Easy to implement/deploy because Opencv resolve that issue in some code lines.
2. No training required and is possible to test the solution immediatelly.
3. Detecting pedestrian frame by frame and it works also with moving camera.

###### WEAKNESS
1. Not always reliable with false positive.
2. Some time the solution has many missings when figure is not complete or is in some particular positions.
3. The solution requires some tuning for every scenarios/light.
4. The code not so fast and some speedup requires a trade of with accuracy. 
5. This solution is no suitable for realtime.  



###### Hog detection part...
in file: "detect_pedestrian.py", using OpenCV builtin feature.

```python
# initialize the HOG descriptor/person detector
import cv2
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
(rects, weights) = hog.detectMultiScale(frame_gray, winStride=(2,2), padding=(4,4), scale=1.05, hitThreshold=1, finalThreshold=2.0, useMeanshiftGrouping=False)
```
###### Non maxima suppression...
```python
# non max suppression
from imutils.object_detection import non_max_suppression
pick = non_max_suppression(rects, probs=None, overlapThresh=overlap_thresh)
```

###### Pedestrian frame crop and saveing annotation in CSV...
```python
# save data in csv
import pandas as pd
detect_info_df=pd.DataFrame(detect_info_mx, columns=['FRAME', 'FRAME_FILE_PATH', 'CROP_FILE_PATH', 'CROP_NUM', 'xA', 'yA', 'xB', 'yB', 'ID'])
detect_info_df.to_csv(output_info_path + '/detect_info.csv')
```
<br/>
<br/>

### RE-IDENTIFICATION PHASE
To reidentify pedestian we compare every croped frame produced in previous detection 
phase with a identities database.<br/>
This database is a images set with the identities to reidentify.
So every crop image is compared by a trained convolutional neural network with all identities in DB and for every 
identity the system return a similarity percentage. 

[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/reid_phase.png)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_reideintification)

###### STRENGTHS
1. No tuning required in preduciont because we have metaparameter only in training phase.
2. Fast in prediction (getting similarity rank for every identity in the database)

###### WEAKNESS
1. The CNN Requires a lot of training with many example to be effective.
2. The input images must have a precide size, so we need to resize detected pedestrian before the reideinfication phase.
3. The CNN is not trained to recognize background, so it may try to recognize every bounding box detected in previous phase also if there are non predestrian in the image.
4. The Siamese Network assume that all people detected in previous phase are in identity database, if in one frame some people are missing the CNN wrong.

<br/>
<br/>

### SIAMESE NETWORK
A trained network to compare people similarity, the idea comes from the this paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf

[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/an_siamese_net_reid.jpg)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_siam_net)

Paired images are passed through the network. While initial layers extract features in the two views individually, higher layers compute relationships between them.<br/> 
The number and size of convolutional filters that must be learned are shown.<br/>
For example, in the first tied convolution layer, 5 x 5 x 3 -> 20 indicates that there are 20 convolutional features in the layer, each with a kernel size of 5 x 5 x 3. There are 2,308,147 learnable parameters in the
whole network.<br/>
Now we describe every layer.

<br/>

###### Tied Convolution
The first two layers are convolution layers, which we use to compute higher-order features on each input image separately in order to return coparable features accross the two images in later layers.<br/>
The weights are shared across the two views, to ensure that both views use the same filters to compute features.<br/>
We pass as input pairs of RGB images of size 60 x 160 -> 3 through 20 learned filters of size 5 x 5 -> 3. <br/>
The resulting feature maps are passed through a max-pooling kernel that halves the width and height of features.<br/>
These features are passed through another tied convolution layer that uses 25 learned filters of size 5 x 5 x 20, 
followed by a max-pooling layer that again decreases the width and height of the feature map by a factor of 2. 
At the end of these two feature computation layers, each input image is represented by 25 feature maps of size 12 x 37.
  
<br/>
  
###### Cross-Input Neighborhood Differences
The aim is computed a rough relationship among features from the two input images in the form of neighborhood difference maps.<br/>
This layer computes differences in feature values across the two views around a neighborhood of each feature location,
producing a set of 25 neighborhood difference maps.<br/>
in simple words the 5 x 5 matrix Ki(x, y) is the difference of two 5 x 5 matrices, in the first of which every element is a copy
of the scalar f(i)(x, y), and the second of which is the 5 x 5 neighborhood of g(i) centered at (x, y).
The differences in a neighborhood is done to add robustness to positional differences in corresponding features of the two input images.
This operation si asymmetric so this layed consider the reverse difference.
This produce 50 neighborhood difference maps (25 in one wqy, 25 in reverse way, each of which has size 12 x 5 x 37 x 5). 
We pass these neighborhood difference maps through a rectified linear unit (ReLu) to add non linearity to the network.


[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/cross-input-neighborhood-diff.png)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_siam_net_detail)


###### Patch Summary Features

The layer summarizes these neighborhood difference maps by producing a summary representation of the differences in each
5 x 5 block.<br/>
This is accomplished by convolving K with 25 filters of size 5 x 5 x 25, with a stride of 5.<br/>
By exactly matching the stride to the width of the square blocks, we ensure that the 25-dimensional 
feature vector at location (x, y) of L is computed only from the 25 blocks K(x, y).
At the end are passed through a rectified linear unit (ReLu).

<br/>

###### Across-Patch Features

So far we have obtained a high-level representation of differences within a local neighborhood, 
by computing neighborhood difference maps and then obtaining a highlevel local representation of these neighborhood difference maps.<br/>
Now we learn spatial relationships across neighborhood differences. <br/>
This is done by convolving L with 25 filters of size 3 x 3 x 25 with a stride of 1. <br/>
The resultant features are passed through a max pooling kernel to reduce the height and width by a factor of 2. <br/>
This yields 25 feature maps of size 5 x 18.<br/>

<br/>

###### Higher-Order Relationships

It is a fully connected layer after M and M'.<br/>
This captures higher-order relationships by:
- combining information from patches that are far from each other 
- combining information from M with information from M'.

The resultant feature vector of size 500 is passed through a ReLu non linearity.<br/>
These 500 outputs are then passed to another fully connected layer containing 2 softmax units,
which represent the probability that the two images in the pair are of the same person or different people.

<br/>
Some code to take in account
https://github.com/Ning-Ding/Implementation-CVPR2015-CNN-for-ReID
<br/>

[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/model_keras_tesorflow.png)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_siam_net_detail)

<br/>
<br/>

###### Visualization of Features

[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/siamese_net_visualiz.png)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_siam_net_detail)

Visualization of features learned by our architecture. <br/>
Initial layers learn image features that are important to distinguish between a positive and a negative pair. 
Deeper layers learn relationships across the two views so that classification performance is maximized.

<br/>

### HOW IS TRAINIED THE SIAMESE NETWORK

The re-identification problem as binary classi-fication.<br/>
Training data consist of image pairs labeled as positive (same) and negative (different).<br/>
The optimization objective is average loss over all pairs in the data set. <br/>
As the data set can be quite large, in practice we use a stochastic approximation of this objective. <br/>
Training data are randomly divided into mini-batches.<br/>
The model performs forward propagation on the current mini-batch and computes the output and loss. <br/>
Backpropagation is then used to compute the gradients on this batch, and network weights are updated. <br/>
In training phase is performed a **stochastic gradient descent** to perform weight updates. 


[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/dataset_market-1501.jpg)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_siam_net_train)

###### Market-1501 Dataset

http://www.liangzheng.org/Project/project_reid.html
- Collected in front of a supermarket in Tsinghua University. 
- A total of six cameras including 5 high-resolution cameras, and one low-resolution camera.
- Contains 32,668 annotated bounding boxes of 1,501 identities.
- Annotated identity is present in at least two cameras.

###### Training code is in model_for_market1501.py

```python
if __name__ == '__main__':
    # define model
    model = model_def(weight_decay=0.0005)
    
    # Before training configure learning process adding an optimizer (SGD), a loss function and a metrics (accuracy) 
    model = compiler_def(model)
    
    #training model
    train(model, batch_size=200)
    
    # training produce a .h5 file containing weight to reload in predicion phase
```

###### Make prediction and add labeled bounding box to the video with run.py

```python
# save data in csv
if __name__ == '__main__':
    # reading detection annotatins (previous phase...)
    detect_info_df = pd.read_csv('video_result/'+video_name+'/frame_detection' + '/frame_detection_annotations.csv')
    
    #define the people to reidentify dictionary 
    dictionary_test_path = ['video_data/ex_01_dict/andrea.jpg', 'video_data/ex_01_dict/sara.jpg'] 
    
    df_annotated = add_reid_data_annotation(model, detect_info_df, dictionary_test_path, 'video_result/'+video_name+'/video_builder')
    
    # rebuilding video with 
    call_build_video(video_name, df_annotated)
```


###### TRAINING AND TEST PERFORMACES

optimization method: **Stochastic gradient descent**<br/>
epoch train: 5000 <br/>
batch size: 200 imgs<br/>
accuracy: 0.8741 <br/> 
loss: 0.3382 <br/> 
training time: ~48h <br/>

Trained on a laptop:<br/>
2.5GHz Dual-core Intel i5, 8GB of 1600MHz DDR3 SDRAM, Intel HD Graphics 4000 

> UBUNTU 16.04 <br/>
> PYTHON 3.6.1 <br/>
> KERAS 2.0.7  <br/>
> TENSORFLOW 1.3.0 <br/>

<br/>
<br/>

### TESTING

Two types:
- **Test detection**: <br/>
For every frame we put in relation detected boundig box with boundig box annotated manually.

- **test reidentification**: <br/>
verify if identity is correct for every detected pedestian in pevious phase with annotated ID.


##### TUD MULTIVIEW PEDESTRIANS

http://www.d2.mpi-inf.mpg.de/node/428<br/>
The TUD Pedestrians dataset from Micha Andriluka, Stefan Roth and Bernt Schiele [AndrilukaCVPR2008] consists of 250 images with 311 fully visible people with significant variation in clothing and articulation. The dataset has 3 video, The dataset "TUD Multiview Pedestrians" was used in the project to evaluate single-frame people detection.
Videos are created with the msmpeg4v2 codec (from ffmpeg)

**Accuracy detection**: 0,634948097<br/>
> Tot annotated detection 1157 smpl.<br/>
> Some relevanti info in CONFUSION MATRIX<br/>
> TP (correct detection)	734 smpl.<br/>
> FN (missing detection)	422 smpl.<br/>
> FP (ghost detection)	0 smpl.

**Accuracy reidentification**:<br/>
> Tot. reidentification done: 728<br/>
> True match: 0.510989 (372 smpl.)<br/>
> impossible match: 0.337912 (246 smpl.)**<br/>
> False match: 0.151099 (110 smpl.)<br/><br/>
> ** people to reidentify are less than people in videos<br/>
> if we don't care impossible match accuracy is 0.771784<br/>
> With a threshold like 0.7 the accuracy grow a little bit (0.69 circa)

##### > MY DATASET (@ UNIMORE) <
My dataset at unimore available in dropbox https://www.dropbox.com/sh/9njwblsl12ihzou/AAC0qW74RN17IP456rTIs7o0a?dl=0
Data acquired from the Axis camera mounted at UNIMORE, with video, people and situation semi-constrained and controlled, usually there are 2 people video are acquire both in static and moving camera in outdoor in high brightness conditions. Videos are created with the msmpeg4v2 codec (from ffmpeg)

**Accuracy detection**: 0,619479049<br/>
> Tot annotated detection 883 smpl.<br/>
> Some relevanti info in CONFUSION MATRIX<br/>
> TP (correct detection)	547 smpl.<br/>
> FN (missing detection)	336 smpl.<br/>
> FP (ghost detection)	0 smpl.<br/>

**Accuracy reidentification**:<br/>
> Tot. reidentification done: 543 <br/>
> True match: 0.732965 (398 smpl.) <br/>
> False match: 0.267035 (145 smpl.)

The video contains only people to reidentify.

<br/>
<br/>

### CONCLUSION
1. Some missing in detection phase, need more tuning
2. Not suitable for realtime application, but good approach for batch services
3. Training performaces can be improved

<br/>
<br/>

### FUTURE WORKS
1. Create a service API to make project available to third party
2. Introduct a classificator that distiguish from background and not background to eliminate false positive
3. Porting siamese net in CAFFE and THEANO to compare speed and accuracy performances in training and prediction
4. Enhance reideinfication removing double id in the same frame

