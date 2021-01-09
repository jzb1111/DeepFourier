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

class_color={
0:[128,0,0],
1:[0,128,0],
2:[128,128,0],
3:[0,0,128],
4:[128,0,128],
5:[0,128,128],
6:[128,128,128],
7:[64,0,0],
8:[192,0,0],
9:[64,128,0],
10:[192,128,0],
11:[64,0,128],
12:[192,0,128],
13:[64,128,128],
14:[192,128,128],
15:[0,64,0],
16:[128,64,0],
17:[0,192,0],
18:[128,192,0],
19:[0,64,128]}

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
    segclass_paths=[]
    segments=load_path("./VOC2007/SegmentationObject")
    for i in segments:
        image_paths.append('./VOC2007/JPEGImages'+'/'+i.replace('png','jpg'))
        segment_paths.append('./VOC2007/SegmentationObject'+'/'+i)
        segclass_paths.append('./VOC2007/SegmentationClass'+'/'+i)
        annotation_paths.append('./VOC2007/Annotations'+'/'+i.replace('png','xml'))
    #print(image_paths)
    #print(segment_paths)
    image=cv2.resize(cv2.imread(image_paths[num]),(256,256))
    segment=cv2.resize(cv2.imread(segment_paths[num]),(256,256))
    segclass=cv2.resize(cv2.imread(segclass_paths[num]),(256,256))
    annotation=annotation_paths[num]
    return image,segment,segclass,annotation

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
    #逻辑梳理：
    #1.生成zeromat
    #2.在segclass中查找box中类别所对应颜色的像素集合
    #3.在segobj中查找box中不同颜色的像素集合，挑选与segclass重合度最高的将zeromat对应位置置一
    #4.运行findcontours，查找出一系列轮廓（一个目标可能有多组轮廓）
    
    #期待输出：
    #[[obj1[obj1part1[y1,x1],[y2,x2]...][obj1part2[...]...]][obj2[...]...]...]
    objlist=load_xml_list(ann)
    objlinelist=[]
    #grayseg=cv2.cvtColor(seg,cv2.COLOR_RGB2GRAY)
    #binseg=cv2.threshold(grayseg,10,1,cv2.THRESH_BINARY)[1]
    zlis=[]
    objcolorlist=[]
    for n in range(len(objlist)):
        xmin=objlist[n][1][0]
        ymin=objlist[n][1][1]
        xmax=objlist[n][1][2]
        ymax=objlist[n][1][3]
        colordict={}
        zeroimg=np.zeros((len(seg),len(seg[0])))
        #查找segobj的box中不同颜色的集合
        for i in range(ymax-ymin):
            for j in range(xmax-xmin):
                color=[]
                for k in range(3):
                    color.append(seg[ymin+i][xmin+j][k])
                color=tuple(color)
                if color not in colordict.keys():
                    colordict[color]=np.zeros((256,256))
                    colordict[color][i][j]=1
                else:
                    colordict[color][i][j]=1
        #查找segclass中box中类别所对应的颜色区域
        anncolor=class_color[objlist[n][0]]
        anncolormat=np.zeros((256,256))
        for i in range(ymax-ymin):
            for j in range(xmax-xmin):
                color=[]
                for k in range(3):
                    color.append(seg[ymin+i][xmin+j][k])
                color=color
                if color==anncolor:
                    anncolormat[i][j]=1
        
        colordictlis=[]
        for i in range(len(colordict.keys())):
            colordictlis.append([list(colordict.keys())[i],colordict[list(colordict.keys())[i]]])
        colordictlis=sorted(colordictlis,key=lambda x: x[1],reverse=True)    
        plt.figure(n)
        plt.imshow(zeroimg)
        zlis.append(zeroimg)
    
    return zlis