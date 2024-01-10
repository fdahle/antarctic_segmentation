# Antarctic segmentation
This repository contains the code for applying semantic segmentation on historical cryospheric imagery. For more information see the
- [ISPRS Open Journal of Photogrammetry and Remote Sensing](https://www.sciencedirect.com/journal/isprs-open-journal-of-photogrammetry-and-remote-sensing) paper <b>Revisiting the Past: A comparative study for semantic segmentation of historical images of Adelaide Island using U-nets</b> (download [here](https://www.sciencedirect.com/science/article/pii/S2667393223000273))

- [ISPRS](https://www.isprs2022-nice.com/) paper <b>Semantic segmentation of historical photographs of the Antarctic Peninsula</b> (download [here](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2022/237/2022/)).

![Example for segmentation](https://github.com/fdahle/antarctic_segmentation/blob/main/readme/segmentation_example.png?raw=true)

The images used in this project are [publicly available](https://www.pgc.umn.edu/data/aerial/) and can be downloaded [here](https://data.pgc.umn.edu/aerial/usgs/tma/photos/).

<h3>Datasets</h3>
Due to file size limitation, the training data and the segmented data of Adelaide Island are not on github. Instead, they can be downloaded at https://doi.org/10.4121/de8ea9d4-f986-41fc-9412-6765985a0c9c.
<h3>Technical information</h3>

There are four important main functions in this repository:

- apply_model.py:
In this function the model can be applied and visualized on unseen imagery.

- evaluate_model.py:
With this function a created model can be evaluated.

- segmentator.py:
A small tool with a gui, that can be used to improve segmented imagery (for example created by unsupervised segmentation).

- train_model.py:
This function is loading the data and initizalizing the training to create the models.

Even though this code is suited especially for my needs and my dataset, it is possible to adapt the code for your needs:
- Change the folder paths to the images and the labels. These are specified at the begin of each of the four main functions.
- Change your training parameters in train_params.py (E.g. model name, number of epochs, number of classes, etc..)


<h3>Background</h3>

This semantic segmentation is a part of [my](https://www.tudelft.nl/citg/over-faculteit/afdelingen/geoscience-remote-sensing/staff/phd-students/f-felix-dahle) PhD project. I am trying use historic imagery of the Antarctica to create 3D models with Structure From Motion. At this [repository](https://github.com/fdahle/Antarctic_TMA/) you can find more information about it (and more code).
