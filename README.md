# Antarctic segmentation
This repository contains the code for applying semantic segmentation on historical cryospheric imagery. For more information see the [ISPRS](https://www.isprs2022-nice.com/) paper <b>Semantic segmentation of historical photographs of the Antarctic Peninsula</b>.

![Example for segmentation](https://github.com/fdahle/antarctic_segmentation/blob/main/readme/segmentation_example.png?raw=true)

The images used in this project are [publicly available](https://www.pgc.umn.edu/data/aerial/) and can be downloaded [here](https://data.pgc.umn.edu/aerial/usgs/tma/photos/). 

<h3>Technical information</h3>

Note that in the currenet state the code is very specific for my project and cannot be use from others without adaptions. In near future this code will be changed to make it more usable by the public.

There are three important main functions in this repository:

- train_model.py:
This function is loading the data/model and initizalizing the training.

- evaluate_model.py:
With this code a created model can be evaluated.

- segmentator.py:
A small tool with a gui, that can be used to improve segmented imagery (for example created by unsupervised segmentation). 

Work in progress!

<h3>Background</h3>
Many archives of historical grayscale aerial imagery is existing, however these images often contain no semantic information at all, which prevents their usage in computer-driven analysis. However, most modern semantic segmentation algorithms fail on these historical images, due to varying image quality, lack of color information, and most crucially, due to artifacts in both imaging as well as scanning (e.g. Newton’s rings)
