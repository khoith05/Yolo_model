import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO
from bs4 import BeautifulSoup


yolo = YOLO(0.6, 0.5)
file = 'data/coco_classes.txt'

def get_xml_data(path):
    with open(path, 'r') as f:
        data = f.read()
    return BeautifulSoup(data, 'xml')

def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

def label_image(imageTag,data):
    path = "images/"+ imageTag["name"]
    image = cv2.imread(path)
    if(image is not None):
        pimage = process_image(image)
        boxes, classes, scores = yolo.predict(pimage, image.shape)
        all_classes = get_classes(file)
        if boxes is not None:
            write_xml(boxes, scores, classes, all_classes, imageTag, data,image)

def write_xml(boxes, scores, classes, all_classes, imageTag, data, image):
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        
        box_tag = new_box_label_tag(data,all_classes[cl],top, left, right, bottom)
        imageTag.append(box_tag)

        # cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        # cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
        #             (top, left - 6),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6, (0, 0, 255), 1,
        #             cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

def new_box_label_tag(bs_data,label,xtl,ytl,xbr,ybr):
    return bs_data.new_tag("box",label=label, xtl=xtl,ytl=ytl,xbr=xbr,ybr=ybr,source="yolov3",z_order=0,occluded=0)

xmlPath = "annotations.xml"
data = get_xml_data(xmlPath)
for imageTag in data.find_all("image"):
  label_image(imageTag,data)
with open("result.xml", "w") as file:
  file.write(str(data.prettify()))

