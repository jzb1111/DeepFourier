# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:29:20 2021

@author: asus
"""

import os
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt

objdict={
'aeroplane':0,
'bicycle':1,
'bird':2,
'boat':3,
'bottle':4,
'bus':5,
'car':6,
'cat':7,
'chair':8,
'cow':9,
'diningtable':10,
'dog':11,
'horse':12,
'motorbike':13,
'person':14,
'pottedplant':15,
'sheep':16,
'sofa':17,
'train':18,
'tvmonitor':19}

def load_path(dirname):
    result = []#所有的文件
    for maindir, subdir, file_name_list in os.walk(dirname):
        #print("1:",maindir) #当前主目录
        #print("2:",subdir) #当前主目录下的所有目录
        #print("3:",file_name_list)  #当前主目录下的所有文件
        for filename in file_name_list:
            #apath = os.path.join(maindir, filename)#合并成一个完整路径
            result.append(filename)
    return result

#res=load_path("./VOC2007/JPEGImages")

def load_raw_data_by_num(num):
    #image_paths=load_path("./VOC2007/JPEGImages")
    image_paths=[]
    segment_paths=[]
    annotation_paths=[]
    segments=load_path("./VOC2007/SegmentationObject")
    for i in segments:
        image_paths.append('./VOC2007/JPEGImages'+'/'+i.replace('png','jpg'))
        segment_paths.append('./VOC2007/SegmentationObject'+'/'+i)
        annotation_paths.append('./VOC2007/Annotations'+'/'+i.replace('png','xml'))
    #print(image_paths)
    #print(segment_paths)
    image=cv2.resize(cv2.imread(image_paths[num]),(256,256))
    segment=cv2.resize(cv2.imread(segment_paths[num]),(256,256))
    annotation=annotation_paths[num]
    return image,segment,annotation

def load_xml_list(annotation_path):
    DOMTree = xml.dom.minidom.parse(annotation_path)
    collection = DOMTree.documentElement
    size=collection.getElementsByTagName("size")[0]
    width=int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height=int(size.getElementsByTagName('height')[0].childNodes[0].data)
    widthratio=256/width
    heightratio=256/height
    objects = collection.getElementsByTagName("object")
    #print(objects)
    objlist=[]
    for obj in objects:
        box=[]
        name = obj.getElementsByTagName('name')[0].childNodes[0].data
        box = obj.getElementsByTagName('bndbox')[0]
        xmin=int(box.getElementsByTagName('xmin')[0].childNodes[0].data)*widthratio
        xmax=int(box.getElementsByTagName('xmax')[0].childNodes[0].data)*widthratio
        ymin=int(box.getElementsByTagName('ymin')[0].childNodes[0].data)*heightratio
        ymax=int(box.getElementsByTagName('ymax')[0].childNodes[0].data)*heightratio
        objlist.append([objdict[name],[int(xmin),int(ymin),int(xmax),int(ymax)]])
    return objlist

def trans_seg_to_list(seg,ann):
    objlist=load_xml_list(ann)
    objlinelist=[]
    grayseg=cv2.cvtColor(seg,cv2.COLOR_RGB2GRAY)
    binseg=cv2.threshold(grayseg,10,1,cv2.THRESH_BINARY)[1]
    zlis=[]
    for n in range(len(objlist)):
        xmin=objlist[n][1][0]
        ymin=objlist[n][1][1]
        xmax=objlist[n][1][2]
        ymax=objlist[n][1][3]
        
        zeroimg=np.zeros((len(binseg),len(binseg[0])))
        for i in range(ymax-ymin):
            for j in range(xmax-xmin):
                color=binseg[ymin+i][xmin+j]
                if color > 0.5:
                    zeroimg[ymin+i][xmin+j]=1
        plt.figure(n)
        plt.imshow(zeroimg)
        zlis.append(zeroimg)
    return zlis