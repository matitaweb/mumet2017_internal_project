# RE-IDENTIFICATION OF PEOPLE IN OUTDOOR SURVEILLANCE VIDEOS (#mumet2017_internal_project)

" The system is capable of reidentify a given (query) person. 
The process annotate videos detecting pedestrian using a simple HOG detector and than with a "Siamese Network" try to reidentify people by a similarity ranking. "

---

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
Built-in feature used:
Python 3, OpenCV 3.2, Panda 0.20

###### STRENGHT
1. Easy to implement/deploy.
2. No training required.
3. It work also if camera is moving.

###### WEAKNESS
1. Not always reliable with false positive.
2. Missings when people figure is not complete.
3. Requires tuning for every scenarios.
4. Not so Fast, speedup requires a trade of with accuracy. 
5. Requires some tuning for every kind of video.

[![N|Solid](https://matitaweb.github.io/mumet2017_internal_project/img/detection_phase.png){:style="float: right;margin-right: 7px;margin-top: 7px;"}](https://matitaweb.github.io/mumet2017_internal_project/index.html#/architecture_main)



---
