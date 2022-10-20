# How to label image

## Step 1

Get *yolo.h5* and *coco_classes.txt* files and paste to */data*.

Copy all images need to be labelled into */images* folder.

## Step 2

Open CVAT, create new task with all images from */images*.

Open task then Choose export annonations, choose COCO v1.1.

Download *.zip* file, unzip and paste *anonations.xml* file here (same folder as this file). 

## Step 3

Run *python main.py*, you will get file results.xml, compress it to *.zip* file.

## Step 4

Open task in CVAT, choose import annonations, choose COCO v1.1 , choose the file you have just create.

Done.