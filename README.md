# RE-IDENTIFICATION OF PEOPLE IN OUTDOOR SURVEILLANCE VIDEOS (#mumet2017_internal_project)
  
>"The system is capable of reidentify a given (query) person. 
>The process annotate videos detecting pedestrian using a simple HOG detector and than with a "Siamese Network" try to reidentify people by a similarity ranking."  
  
 


### FIRST SOME RESULTS...
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/RipYW9D15fs/0.jpg)](http://www.youtube.com/watch?v=RipYW9D15fs)

### SOME PROJECT ASSUMPTIONS....
- Static camera or simple move as pan or tilt.
- We have a set of pedestrian image for the reidentification.
- Good light condition
- No crowd or animals and moving objects.

### ARCHITECTURE
###### Detection phase
Simple HOG detection producing croped people image and annotation.

###### ReID phase
A Trained Siamese Net. try reidentify pedestrian searching in a DB, the output is a similarity rank for everty peope to reidentify.

###### Video annotation phase
Adding to the video the reidentificatoin info.

[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/architettura_soluzione.png)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_main)


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
1. Easy to implement/deploy.
2. No training required.
3. It work also with moving camera.

###### WEAKNESS
1. Not always reliable with false positive.
2. Missings when people figure is not complete.
3. Requires tuning for every scenarios.
4. Not so Fast, speedup requires a trade of with accuracy. 
5. Requires some tuning for every kind of video.



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


### RE-IDENTIFICATION PHASE
To reidentify pedestian we compare every croped frame produced in previous detection 
phase with a set of given image that contains a set of people to reidentify.
So every crop image is compared by a trained convolutional neural network with all possible people that have an a identity 
and for every identity we save a similarity percentage. 
An than we assign 


###### STRENGTHS
1. No tuning required.
2. Fast in prediction.

###### WEAKNESS
1. Requires a lot of training with many example
2. The input images have a precide size, so we need to resize detected pedestrian.
3. Is not trained to recognize background, so it try to recognize every bounding box detected in previous phase.
4. The net assume that all people detected are in the people to recognize set.

[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/reid_phase.png)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_reideintification)

### SIAMESE NETWORK
A trained network to compare people similarity, the idea came from the this paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ahmed_An_Improved_Deep_2015_CVPR_paper.pdf

[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/an_siamese_net_reid.jpg)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_siam_net)

The Architecture: Paired images are passed through the network. While initial layers extract features in the
two views individually, higher layers compute relationships between them. The number and size of convolutional filters that
must be learned are shown. For example, in the first tied convolution layer, 5 x 5 x 3 -> 20 indicates that there are 20
convolutional features in the layer, each with a kernel size of 5 x 5 x 3. There are 2,308,147 learnable parameters in the
whole network.

###### Tied Convolution
The first two layers convolution layers, which we use to compute
higher-order features on each input image separately. 
In order for the features to be comparable across the two images
in later layers, our first two layers perform tied convolution,
in which weights are shared across the two views, to ensure
that both views use the same filters to compute features. 
We pass as input pairs of RGB images of size 60 x 160 -> 3 through 20 learned filters of size 5 x 5 -> 3. 
The resulting feature maps are passed through a max-pooling kernel that halves
  
###### Cross-Input Neighborhood Differences
  
 
###### Patch Summary Features

###### Across-Patch Features

###### Higher-Order Relationships


Some code to take in account

https://github.com/Ning-Ding/Implementation-CVPR2015-CNN-for-ReID

### HOW I TRAINIED THE SIAMESE NETWORK
[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/dataset_market-1501.jpg)](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_siam_net_train)

###### Market-1501 Dataset
http://www.liangzheng.org/Project/project_reid.html
- Collected in front of a supermarket in Tsinghua University. 
- A total of six cameras including 5 high-resolution cameras, and one low-resolution camera.
- Contains 32,668 annotated bounding boxes of 1,501 identities.
- Annotated identity is present in at least two cameras.

###### TRAINING AND TEST PERFORMACES
optimization method: Stochastic gradient descent
epoch train: 5000 | batch size: 200 imgs
accuracy: 0.8741 | loss: 0.3382 | 2/3 days

Trained on a laptop:
2.5GHz Dual-core Intel i5, 8GB of 1600MHz DDR3 SDRAM, Intel HD Graphics 4000 
> UBUNTU 16.04 | PYTHON 3.6.1 
> KERAS 2.0.7  |  TENSORFLOW 1.3.0

### TESTING

two types:
- **Test detection**: 
for every frame put in relation detected boundig box with boundig box annotated manually

- **test reidentification**:
verify if identity is correct for every detected pedestian in pevious phase with annotated ID.


###### TUD MULTIVIEW PEDESTRIANS

http://www.d2.mpi-inf.mpg.de/node/428
The TUD Pedestrians dataset from Micha Andriluka, Stefan Roth and Bernt Schiele [AndrilukaCVPR2008] consists of 250 images with 311 fully visible people with significant variation in clothing and articulation. The dataset has 3 video, The dataset "TUD Multiview Pedestrians" was used in the project to evaluate single-frame people detection.
Videos are created with the msmpeg4v2 codec (from ffmpeg)

- Accuracy detection: 0,634948097
Tot annotated detection 1157 smpl.
Some relevanti info in CONFUSION MATRIX
TP (correct detection)	734 smpl.
FN (missing detection)	422 smpl.
FP (ghost detection)	0 smpl.

- Accuracy reidentification:
Tot. reidentification done: 728
true match: 0.510989 (372 smpl.)
impossible match: 0.337912 (246 smpl.)**
false match: 0.151099 (110 smpl.)
** people to reidentify are less than people in videos
if we don't care impossible match accuracy is 0.771784
With a threshold like 0.7 the accuracy grow a little bit (0.69 circa)

###### > MY DATASET AT UNIMORE <
My dataset at unimore available in dropbox
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

### CONCLUSION
1. Some missing in detection phase, need more tuning
2. Not suitable for realtime application, but good approach for batch services
3. Training performaces can be improved



### FUTURE WORKS
1. Create a service API to make project available to third party
2. Introduct a classificator that distiguish from background and not background to eliminate false positive
3. Porting siamese net in CAFFE and THEANO to compare speed and accuracy performances in training and prediction
4. Enhance reideinfication removing double id in the same frame
5. 