# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:41:54 2019

@author: ADMIN
"""

import re
import os
import numpy as np
import cv2
import random
from xml.dom.minidom import parse
import xml.dom.minidom
import sys
sys.path.extend(['E:\\paper\\DeepFourier4','E:/paper/DeepFourier4'])
from runNMS import run_nms

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

def file_name(file_dir):
    root_=[]
    dirs_=[]
    files_=[]
    for root,dirs,files in os.walk(file_dir):
        root_.append(root)
        dirs_.append(dirs)
        files_.append(files)
    return root_,dirs_,files

def clsname2num(name):
    name_dict={'aeroplane':0,'bicycle':1,'bird':2,'boat':3,'bottle':4,'bus':5,'car':6,
               'cat':7,'chair':8,'cow':9,'diningtable':10,'dog':11,'horse':12,'motorbike':13,
               'person':14,'pottedplant':15,'sheep':16,'sofa':17,'train':18,'tvmonitor':19}
    return name_dict[name]

def get_xml(f_n):
    path='./VOC2007/Annotations/'
    DOMTree = xml.dom.minidom.parse(path+f_n)
    annotation = DOMTree.documentElement

    file_name = annotation.getElementsByTagName("filename")[0].firstChild.data
    size = annotation.getElementsByTagName("size")
    width=int(size[0].getElementsByTagName("width")[0].firstChild.data)
    height=int(size[0].getElementsByTagName("height")[0].firstChild.data)
    depth=int(size[0].getElementsByTagName("depth")[0].firstChild.data)
    
    #h_ratio=512/height
    #w_ratio=512/width
    
    objects=annotation.getElementsByTagName("object")
    ob_dict={}
    for i in range(len(objects)):
        tmp_dict={}
        tmp_dict['name']=clsname2num(objects[i].getElementsByTagName("name")[0].firstChild.data)
        bndbox=objects[i].getElementsByTagName("bndbox")[0]
        tmp_dict['xmin']=int(bndbox.getElementsByTagName("xmin")[0].firstChild.data)
        tmp_dict['xmax']=int(bndbox.getElementsByTagName("xmax")[0].firstChild.data)
        tmp_dict['ymin']=int(bndbox.getElementsByTagName("ymin")[0].firstChild.data)
        tmp_dict['ymax']=int(bndbox.getElementsByTagName("ymax")[0].firstChild.data)
        ob_dict[i]=tmp_dict
    xml_dict={}
    xml_dict['f_name']=file_name
    xml_dict['width']=width
    xml_dict['height']=height
    xml_dict['depth']=depth
    xml_dict['obj']=ob_dict
    return xml_dict

def load_image(imageurl):#加载vgg模型必须这样加载图像
    im = cv2.resize(cv2.imread(imageurl),(224,224)).astype(np.float32)
    return im

def get_locdata(xmldata):#没写完，等下接着写,这个函数要写成将xmldata输出为list的函数
    out=[]
    for i in range(len(xmldata)):
        objs=xmldata[i]['obj']
        width=xmldata[i]['width']
        height=xmldata[i]['height']
        w_ratio=224/width
        h_ratio=224/height
        onepic=[]
        for j in range(len(objs)):
            tmp=[]
            tmp.append(round(objs[j]['xmin']*w_ratio))
            tmp.append(round(objs[j]['ymin']*h_ratio))
            tmp.append(round(objs[j]['xmax']*w_ratio))
            tmp.append(round(objs[j]['ymax']*h_ratio))
            tmp.append(objs[j]['name'])
            onepic.append(tmp)
        out.append(onepic)
    return out

def get_picdata(f_n):
    pic_path='./VOC2007/JPEGImages/'
    pic_f=pic_path+f_n
    #pic=np.zeros((224,224,3))
    pic=load_image(pic_f)
    return pic

def get_mini_original_batch(size):
    path1='./VOC2007/Annotations/'
    path2='./VOC2007/JPEGImages/'
    r1,d1,f1=file_name(path1)
    r2,d2,f2=file_name(path2)
    floc=f1[0]
    fpic=f2[0]
    sjs=np.random.randint(0,len(floc),size)
    loc_dict={}
    pic_data=np.zeros((size,224,224,3))
    for i in range(len(sjs)):
        loc_dict[i]=get_xml(floc[sjs[i]])
        pic_data[i]=get_picdata(fpic[sjs[i]])
    return loc_dict,pic_data

def get_batch_by_num(num):
    path1='./VOC2007/Annotations/'
    path2='./VOC2007/JPEGImages/'
    r1,d1,f1=file_name(path1)
    r2,d2,f2=file_name(path2)
    floc=f1
    #print(floc)
    fpic=f2
    #sjs=np.random.randint(0,len(floc),size)
    sjs=[num]
    loc_dict={}
    pic_data=np.zeros((1,224,224,3))
    for i in range(len(sjs)):
        loc_dict[i]=get_xml(floc[sjs[i]])
        pic_data[i]=get_picdata(fpic[sjs[i]])
    return loc_dict,pic_data

def generate_box(hei,wid,cha):#height的范围在0~32
    #h_ratio=224/14
    #w_ratio=224/14
    scalelist=[40,79,158]
    scale=scalelist[int(cha/3)]
    #print(scale)
    ratiolist=[1,1/1.4,1.4]
    ratio=ratiolist[cha%3]
    #print(ratio)
    lefttop_h=(hei+0.5)*16-(scale*ratio)/2
    #print(lefttop_h)
    lefttop_w=(wid+0.5)*16-(scale/ratio)/2
    #print(lefttop_w)
    rightbottom_h=(hei+0.5)*16+(scale*ratio)/2
    #print(rightbottom_h)
    rightbottom_w=(wid+0.5)*16+(scale/ratio)/2
    #lefttop=[lefttop_h,lefttop_w]
    #rightbottom=[rightbottom_h,rightbottom_w]
    out=[int(lefttop_h),int(lefttop_w),int(rightbottom_h),int(rightbottom_w)]
    return out

def IoU(boxA,boxB):
        #boxA=[A的左上角x坐标left，A的左上角y坐标top，A的右下角x坐标right，A的右下角y坐标bottom]
    yA=max(boxA[0],boxB[0])
    xA=max(boxA[1],boxB[1])
    yB=min(boxA[2],boxB[2])
    xB=min(boxA[3],boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #print(float(boxAArea + boxBArea - interArea))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

'''def generate_rpn_train_data(loc_data):#loc_data[[[x1,y1,x2,y2,cls][obj2]]]
    outclssample=np.zeros((len(loc_data),14,14,9,2))
    outregsample=np.zeros((len(loc_data),14,14,9,4))
    ocs_no=np.full((len(loc_data),14,14,9),-1)
    reg_no=np.full((len(loc_data),14,14,9),-1)
    ioulist=[]
    iouclslist={}
    for i in range(len(loc_data[0])):
        #for j in range(len(loc_data[0][i])):
        iouclslist[i]=[]
        gt_lefttop_h=loc_data[0][i][1]
        gt_lefttop_w=loc_data[0][i][0]
        gt_rightbottom_h=loc_data[0][i][3]
        gt_rightbottom_w=loc_data[0][i][2]
        gt_cls=loc_data[0][i][4]
        gt_box=[gt_lefttop_h,gt_lefttop_w,gt_rightbottom_h,gt_rightbottom_w]
        for h in range(14):
            for w in range(14):
                for s in range(9):
                    tmp_box=generate_box(h,w,s)
                    if tmp_box[0]>0 and tmp_box[1]>0 and tmp_box[2]<224 and tmp_box[3]<224:
                        ioulist.append([IoU(tmp_box,gt_box),[h,w,s]])
                        iouclslist[i].append([IoU(tmp_box,gt_box),[h,w,s]])
        iouclslist[i]=sorted(iouclslist[i],reverse=True)
        
    right_sample_num=len(loc_data[0])*30#每个object30个框
    for c_num in range(len(loc_data[0])):
        k=0
        right_n=0
        while right_n<=round(right_sample_num/len(loc_data[i])):
            boxtmp=generate_box(iouclslist[c_num][k][1][0],iouclslist[c_num][k][1][1],iouclslist[c_num][k][1][2])
            if boxtmp[0]>0 and boxtmp[1]>0 and boxtmp[2]<224 and boxtmp[3]<224:
                outclssample[0][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][0]=1
                right_n=right_n+1
                boxtmp=generate_box(iouclslist[c_num][k][1][0],iouclslist[c_num][k][1][1],iouclslist[c_num][k][1][2])
                
                anchor_lefttop_h=boxtmp[0]
                anchor_lefttop_w=boxtmp[1]
                anchor_rightbottom_h=boxtmp[2]
                anchor_rightbottom_w=boxtmp[3]
                
                ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
                xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
                ha=anchor_rightbottom_h-anchor_lefttop_h
                wa=anchor_rightbottom_w-anchor_rightbottom_w
                
                gt_lefttop_h=loc_data[i][c_num][1]
                gt_lefttop_w=loc_data[i][c_num][0]
                gt_rightbottom_h=loc_data[i][c_num][3]
                gt_rightbottom_w=loc_data[i][c_num][2]
                
                gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
                gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
                gt_h=gt_rightbottom_h-gt_lefttop_h
                gt_w=gt_rightbottom_w-gt_lefttop_w
                ty=(gt_y-ya)/ha
                tx=(gt_x-xa)/wa
                tw=np.log(gt_w/wa)
                th=np.log(gt_h/ha)
                outregsample[i][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][0]=ty
                outregsample[i][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][1]=tx
                outregsample[i][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][2]=th
                outregsample[i][iouclslist[c_num][k][1][0]][iouclslist[c_num][k][1][1]][iouclslist[c_num][k][1][2]][3]=tw
            k=k+1
            neg_sample_num=0
            while neg_sample_num<right_sample_num:
                sjs=np.random.randint(0,len(ioulist)/len(loc_data[i]),1)[0]
                ifneg=0
                boxtmp=generate_box(ioulist[sjs][1][0],ioulist[sjs][1][1],ioulist[sjs][1][2])
                for ifn in range(len(loc_data[i])):
                    if ioulist[sjs+ifn*14*14*9][0]>=0.1:
                        ifneg=ifneg+1
                if ifneg==0 and boxtmp[0]>0 and boxtmp[1]>0 and boxtmp[2]<224 and boxtmp[3]<224:
                    outclssample[0][ioulist[sjs][1][0]][ioulist[sjs][1][1]][ioulist[sjs][1][2]][1]=1
                    neg_sample_num=neg_sample_num+1
            for he in range(14):
                for wi in range(14):
                    for st in range(9):
                        if outclssample[0][he][wi][st][0]==1 or outclssample[i][he][wi][st][1]==1:
                            ocs_no[0][he][wi][st]=0
                        if outclssample[0][he][wi][st][0]==1:
                            reg_no[0][he][wi][st]=0 
    return outclssample,outregsample,ocs_no,reg_no'''


def generate_rpn_train_data_v2(loc_data):#loc_data[[[x1,y1,x2,y2,cls][obj2]]]
      
    outclssample=np.zeros((len(loc_data),14,14,9,2))
    outregsample=np.zeros((len(loc_data),14,14,9,4))
    ocs_no=np.full((len(loc_data),14,14,9),-1)
    reg_no=np.full((len(loc_data),14,14,9),-1)
    
    loc_data=loc_data[0]
    for i in range(len(loc_data)):
        obj=loc_data[i]
        lefttop_h=obj[1]
        lefttop_w=obj[0]
        rightbottom_h=obj[3]
        rightbottom_w=obj[2]
        #clsobj=obj[4]
        
        gtbox=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
        ioulist=[]
        for h in range(14):
            for w in range(14):
                for s in range(9):
                    boxtmp=generate_box(h,w,s)
                    #print(gtbox,boxtmp)
                    #print(h,w,s)
                    ioulist.append([IoU(gtbox,boxtmp),[h,w,s]])
        sortlist=sorted(ioulist,reverse=True)
        
        pos_num=25
        reglist=[]
        
        j=0
        for j in range(len(sortlist)):
        #while j<pos_num: 
            #print(sortlist[j])
            hei=sortlist[j][1][0]
            wid=sortlist[j][1][1]
            cha=sortlist[j][1][2]
            
            boxtmp=generate_box(hei,wid,cha)
            anchor_lefttop_h=boxtmp[0]
            anchor_lefttop_w=boxtmp[1]
            anchor_rightbottom_h=boxtmp[2]
            anchor_rightbottom_w=boxtmp[3]
                
            ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
            xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
            ha=anchor_rightbottom_h-anchor_lefttop_h
            wa=anchor_rightbottom_w-anchor_lefttop_w
                
            gt_lefttop_h=lefttop_h
            gt_lefttop_w=lefttop_w
            gt_rightbottom_h=rightbottom_h
            gt_rightbottom_w=rightbottom_w
                
            gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
            gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
            gt_h=gt_rightbottom_h-gt_lefttop_h
            gt_w=gt_rightbottom_w-gt_lefttop_w
            ty=(gt_y-ya)/ha
            tx=(gt_x-xa)/wa
            #print(gt_w,wa)
            tw=np.log(gt_w/wa)
            th=np.log(gt_h/ha)
            #cor_box=correct_box(ty,tx,th,tw)
            
            reglist.append([ty,tx,th,tw])
        k=0
        #print(reglist)
        conf=1
        while k<pos_num or conf>0.4:
            hei=sortlist[k][1][0]
            wid=sortlist[k][1][1]
            cha=sortlist[k][1][2]
            
            ty=reglist[k][0]
            tx=reglist[k][1]
            th=reglist[k][2]
            tw=reglist[k][3]
            
            conf=sortlist[k][0]
            #print(generate_box(hei,wid,cha))
            #print([ty,tx,th,tw])
            #cor_box=correct_box(generate_box(hei,wid,cha),[ty,tx,th,tw])
            #if cor_box[0]>=0 and cor_box[0]<=223 and cor_box[1]>=0 and cor_box[1]<=223 and cor_box[2]>=0 and cor_box[2]<=223 and cor_box[3]>=0 and cor_box[3]<=223:
            outclssample[0][hei][wid][cha][0]=1
            outregsample[0][hei][wid][cha][0]=ty    
            outregsample[0][hei][wid][cha][1]=tx    
            outregsample[0][hei][wid][cha][2]=th    
            outregsample[0][hei][wid][cha][3]=tw    
            k=k+1
    neg_num=0
    gt_boxes=[]
    for i in range(len(loc_data)):
        obj=loc_data[i]
        lefttop_h=obj[1]
        lefttop_w=obj[0]
        rightbottom_h=obj[3]
        rightbottom_w=obj[2]
        gt_boxes.append([lefttop_h,lefttop_w,rightbottom_h,rightbottom_w])
    while neg_num<pos_num*len(gt_boxes):
        
        sjh=np.random.randint(0,14)
        sjw=np.random.randint(0,14)
        sjc=np.random.randint(0,9)
        
        negbox=generate_box(sjh,sjw,sjc)
        if_neg=0
        for i in range(len(gt_boxes)):
            if IoU(gt_boxes[i],negbox)<=0.05:
                if_neg=if_neg+1
                #print(if_neg,'if_neg')
                #print(len(gt_boxes),'gt')
        if if_neg==len(gt_boxes):
            outclssample[0][sjh][sjw][sjc][1]=1
            neg_num=neg_num+1
    for h in range(14):
        for w in range(14):
            for c in range(9):
                if outclssample[0][h][w][c][0]==1 and outclssample[0][h][w][c][1]==1:
                    outclssample[0][h][w][c][0]=0
    for h in range(14):
        for w in range(14):
            for c in range(9):
                if outclssample[0][h][w][c][0]==1 or outclssample[0][h][w][c][1]==1:
                    ocs_no[0][h][w][c]=0
                if outclssample[0][h][w][c][0]==1:
                    reg_no[0][h][w][c]=0
    return outclssample,outregsample,ocs_no,reg_no


def generate_fasthead_train_sample(boxes,loc_data):#boxes[N,4];    loc_data[[[x1,y1,x2,y2,cls][obj2]]]            
    out_clsv=np.zeros((len(boxes),21))
    out_regv=np.zeros((len(boxes),21,4))
    cls_no=np.full((len(boxes)),-1)
    reg_no=np.full((len(boxes),21,4),-1)
    
    all_pos_num=0
    all_neg_num=0
    loc_data=loc_data[0]
    
    poslist=[]
    neglist=[]
    for i in range(len(boxes)):
        boxtmp=boxes[i]
        ifneg=0
        for j in range(len(loc_data)):
            lefttop_h=loc_data[j][1]
            lefttop_w=loc_data[j][0]
            rightbottom_h=loc_data[j][3]
            rightbottom_w=loc_data[j][2]
            gt_cls=loc_data[j][4]
            #print(gt_cls)
            gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
            #neg_cls=0
            #print(boxtmp,gt_box)
            #################
            #控制正负样本选取#
            #################
            if IoU(boxtmp,gt_box)>=0.65:
                boxcls=int(gt_cls)
                #print(i,boxcls)
                out_clsv[i][boxcls]=1
                
                poslist.append(i)
                
                all_pos_num=all_pos_num+1
                #print(i,boxcls)
                
                anchor_lefttop_h=boxtmp[0]
                anchor_lefttop_w=boxtmp[1]
                anchor_rightbottom_h=boxtmp[2]
                anchor_rightbottom_w=boxtmp[3]
                    
                ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
                xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
                ha=anchor_rightbottom_h-anchor_lefttop_h
                wa=anchor_rightbottom_w-anchor_lefttop_w
                    
                if ha==0:
                    ha=0.01
                if wa==0:
                    wa=0.01
                
                gt_lefttop_h=lefttop_h
                gt_lefttop_w=lefttop_w
                gt_rightbottom_h=rightbottom_h
                gt_rightbottom_w=rightbottom_w
                    
                gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
                gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
                gt_h=gt_rightbottom_h-gt_lefttop_h
                gt_w=gt_rightbottom_w-gt_lefttop_w
                
                ty=(gt_y-ya)/ha
                tx=(gt_x-xa)/wa
                tw=np.log(gt_w/wa)
                th=np.log(gt_h/ha)
                out_regv[i][boxcls][0]=ty
                out_regv[i][boxcls][1]=tx
                out_regv[i][boxcls][2]=th
                out_regv[i][boxcls][3]=tw
            if IoU(boxtmp,gt_box)<=0.3:
                ifneg=ifneg+1
        if ifneg==len(loc_data):
            out_clsv[i][20]=1
            all_neg_num=all_neg_num+1
            neglist.append(i)
    #确保negnum是posnum的三倍
    #print(all_pos_num,all_neg_num)
    #all_neg_num=len(boxes)-all_pos_num
    
    if all_pos_num>=int(round(all_neg_num/3)):
        pos_num=int(round(all_neg_num/3))
        neg_num=int(pos_num*2)
    else:
        pos_num=int(all_pos_num)
        neg_num=int(round(pos_num*0.15))
    ############
    #设定负样本 #
    ############
    if neg_num==0:
        print('no pos')
        if np.random.uniform(0,1)<=0.5:
            neg_num=1
        else:
            neg_num=0
    #print(pos_num,neg_num)
    k=0
    #print(poslist)
    #print(neglist)
    while k<pos_num:
        sjs=np.random.randint(0,len(poslist))
        #print(out_clsv[sjs][20])
        #print(out_clsv[poslist[sjs]])
        if out_clsv[poslist[sjs]][20]!=1:
            cls_no[poslist[sjs]]=0
            objcls=vec2num(out_clsv[poslist[sjs]])
            reg_no[poslist[sjs]][objcls][0]=0
            reg_no[poslist[sjs]][objcls][1]=0
            reg_no[poslist[sjs]][objcls][2]=0
            reg_no[poslist[sjs]][objcls][3]=0
            k=k+1
    l=0
    #print(len(neglist))
    if len(neglist)>0:
        while l<neg_num:
            sjs=np.random.randint(0,len(neglist))
            if out_clsv[neglist[sjs]][20]==1:
                cls_no[neglist[sjs]]=0
                objcls=20
                l=l+1
    '''else:
        while l<1:#neg_num:
            print('fake_neg')
            sjs=np.random.randint(0,len(boxes))
            #if out_clsv[neglist[sjs]][20]==1:
            cls_no[sjs]=0
            objcls=20
            out_clsv[sjs][20]=1
            l=l+1  '''  
        
    out_regv=np.reshape(out_regv,[-1,84])
    reg_no=np.reshape(reg_no,[-1,84])
    if pos_num>0 and neg_num>0:
        flag=1
    else:
        flag=0
    return out_clsv,out_regv,cls_no,reg_no,flag
            #else:
             #   neg_cls=neg_cls+1
 
def generate_fasthead_train_sample_v2(boxes,loc_data):#boxes[N,4];    loc_data[[[x1,y1,x2,y2,cls][obj2]]]            
    out_clsv=np.zeros((len(boxes),2))
    out_regv=np.zeros((len(boxes),4))
    cls_no=np.full((len(boxes)),-1)
    reg_no=np.full((len(boxes),4),-1)
    
    all_pos_num=0
    all_neg_num=0
    loc_data=loc_data[0]
    
    poslist=[]
    neglist=[]
    for i in range(len(boxes)):
        boxtmp=boxes[i]
        ifneg=0
        for j in range(len(loc_data)):
            lefttop_h=loc_data[j][1]
            lefttop_w=loc_data[j][0]
            rightbottom_h=loc_data[j][3]
            rightbottom_w=loc_data[j][2]
            gt_cls=loc_data[j][4]
            #print(gt_cls)
            gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
            #neg_cls=0
            #print(boxtmp,gt_box)
            #################
            #控制正负样本选取#
            #################
            if IoU(boxtmp,gt_box)>=0.5:
                boxcls=int(gt_cls)
                #print(i,boxcls)
                out_clsv[i][0]=1
                cls_no[i]=0
                poslist.append(i)
                
                all_pos_num=all_pos_num+1
                #print(i,boxcls)
                
                anchor_lefttop_h=boxtmp[0]
                anchor_lefttop_w=boxtmp[1]
                anchor_rightbottom_h=boxtmp[2]
                anchor_rightbottom_w=boxtmp[3]
                    
                ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
                xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
                ha=anchor_rightbottom_h-anchor_lefttop_h
                wa=anchor_rightbottom_w-anchor_lefttop_w
                    
                if ha==0:
                    ha=0.01
                if wa==0:
                    wa=0.01
                
                gt_lefttop_h=lefttop_h
                gt_lefttop_w=lefttop_w
                gt_rightbottom_h=rightbottom_h
                gt_rightbottom_w=rightbottom_w
                    
                gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
                gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
                gt_h=gt_rightbottom_h-gt_lefttop_h
                gt_w=gt_rightbottom_w-gt_lefttop_w
                
                ty=(gt_y-ya)/ha
                tx=(gt_x-xa)/wa
                tw=np.log(gt_w/wa)
                th=np.log(gt_h/ha)
                out_regv[i][0]=ty
                out_regv[i][1]=tx
                out_regv[i][2]=th
                out_regv[i][3]=tw
                
                reg_no[i][0]=0
                reg_no[i][1]=0
                reg_no[i][2]=0
                reg_no[i][3]=0
            if IoU(boxtmp,gt_box)<=0.5:
                ifneg=ifneg+1
        if ifneg==len(loc_data):
            out_clsv[i][1]=1
            all_neg_num=all_neg_num+1
            neglist.append(i)
    #确保negnum是posnum的三倍
    #print(all_pos_num,all_neg_num)
    #all_neg_num=len(boxes)-all_pos_num
    
    '''if all_pos_num>=int(round(all_neg_num/3)):
        pos_num=int(round(all_neg_num/3))
        neg_num=int(pos_num*2)
    else:
        pos_num=int(all_pos_num)
        neg_num=int(round(pos_num*0.15))'''
    pos_num=all_pos_num
    neg_num=pos_num*3
    ############
    #设定负样本 #
    ############
    if neg_num==0:
        print('no pos')
        if np.random.uniform(0,1)<=0.5:
            neg_num=1
        else:
            neg_num=0
    #print(pos_num,neg_num)
    '''k=0
    #print(poslist)
    #print(neglist)
    while k<pos_num:
        sjs=np.random.randint(0,len(poslist))
        #print(out_clsv[sjs][20])
        #print(out_clsv[poslist[sjs]])
        if out_clsv[poslist[sjs]][1]!=1 and cls_no[poslist[sjs]]==-1:
            cls_no[poslist[sjs]]=0
            objcls=vec2num(out_clsv[poslist[sjs]])
            reg_no[poslist[sjs]][0]=0
            reg_no[poslist[sjs]][1]=0
            reg_no[poslist[sjs]][2]=0
            reg_no[poslist[sjs]][3]=0
            k=k+1'''
    l=0
    #print(len(neglist))
    if len(neglist)>0:
        while l<neg_num:
            sjs=np.random.randint(0,len(neglist))
            if out_clsv[neglist[sjs]][1]==1:
                cls_no[neglist[sjs]]=0
                objcls=20
                l=l+1
    '''else:
        while l<1:#neg_num:
            print('fake_neg')
            sjs=np.random.randint(0,len(boxes))
            #if out_clsv[neglist[sjs]][20]==1:
            cls_no[sjs]=0
            objcls=20
            out_clsv[sjs][20]=1
            l=l+1  '''  
        
    out_regv=np.reshape(out_regv,[-1,4])
    reg_no=np.reshape(reg_no,[-1,4])
    if pos_num>0 and neg_num>0:
        flag=1
    else:
        flag=0
    return out_clsv,out_regv,cls_no,reg_no,flag

def vec2num(vec):
    out=20
    for i in range(len(vec)):
         if vec[i]==1:
             out=i
    return out
def all_zero(vector):
    v_co=0
    for i in range(len(vector)):
        if vector[i]==0:
            v_co=v_co+1
    if v_co==len(vector):
        out='all_z'
    else:
        out='some'
    return out                
def correct_box(anchorbox,reg):
    out=[]
    anchor_lefttop_h=anchorbox[0]
    anchor_lefttop_w=anchorbox[1]
    anchor_rightbottom_h=anchorbox[2]
    anchor_rightbottom_w=anchorbox[3]
    ty=reg[0]
    tx=reg[1]
    th=reg[2]
    tw=reg[3]
    ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
    xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
    ha=anchor_rightbottom_h-anchor_lefttop_h
    wa=anchor_rightbottom_w-anchor_lefttop_w
    y=ty*ha+ya
    x=tx*wa+xa
    h=np.power(np.e,th)*ha
    w=np.power(np.e,tw)*wa
    lefttop_h=int(y-h/2)
    lefttop_w=int(x-w/2)
    rightbottom_h=int(y+h/2)
    rightbottom_w=int(x+w/2)
    out=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
    return out

def draw_rpn_boxes(pd,ocs,reg):
    im=pd[0]
    anchor_box=[]
    bbox=[]
    neg_box=[]
    for h in range(14):
        for w in range(14):
            for c in range(9):
                if ocs[0][h][w][c][0]==1:
                    anchor_box.append(generate_box(h,w,c))
                    bbox.append(correct_box(generate_box(h,w,c),reg[0][h][w][c]))
                    #print(generate_box(h,w,c),reg[0][h][w][c],correct_box(generate_box(h,w,c),reg[0][h][w][c]))
                    #print()
                if ocs[0][h][w][c][1]==1:
                    neg_box.append(generate_box(h,w,c))
    pos_box=bbox#anchor_box
    #print(anchor_box)
    #print()
    for i in range(len(anchor_box)):                
        cv2.rectangle(im,(pos_box[i][1],pos_box[i][0]),(pos_box[i][3],pos_box[i][2]),(0,255,0),1)#画出框
        #if pos_box[i][1]<0 or pos_box[i][1]>223:
            #print(pos_box[i][1])
        try:
            cv2.rectangle(im,(neg_box[i][1],neg_box[i][0]),(neg_box[i][3],neg_box[i][2]),(0,0,255),1)#画出框
        except:
            pass
    cv2.imshow("Image", im/255)    
    cv2.waitKey (0)
    cv2.destroyAllWindows()
    im=pd[0]
    return 0

def load_train_data(num):
    pdname='./train_data/pd_'+str(num)+'.npy'
    ocsname='./train_data/rpn_ocs_'+str(num)+'.npy'
    orsname='./train_data/rpn_ors_'+str(num)+'.npy'
    noname='./train_data/rpn_no_'+str(num)+'.npy'
    rnname='./train_data/rpn_rn_'+str(num)+'.npy'
    gldname='./train_data/gld_'+str(num)+'.npy'#[y1,x1,y2,x2,cls]
    ftname='./train_data/ft_'+str(num)+'.npy'#[[],[],[]]
    pd=np.load(pdname).astype(np.float32)
    ocs=np.load(ocsname).astype(np.float32)
    ors=np.load(orsname).astype(np.float32)
    no=np.load(noname).astype(np.float32)
    rn=np.load(rnname).astype(np.float32)
    gld=np.load(gldname).astype(np.float32)
    ft=np.load(ftname,allow_pickle=True)#.astype(np.float32)
    return pd,ocs,ors,no,rn,gld,ft

def split_box(boxes):
    out=[]
    for i in range(len(boxes)):
        lefttop_h=boxes[i][0]
        lefttop_w=boxes[i][1]
        rightbottom_h=boxes[i][2]
        rightbottom_w=boxes[i][3]
        if lefttop_h<0:
            lefttop_h=0
        if lefttop_h>223:
            lefttop_h=223
        if lefttop_w<0:
            lefttop_w=0
        if lefttop_w>223:
            lefttop_w=223
        if rightbottom_h<0:
            rightbottom_h=0
        if rightbottom_h>223:
            rightbottom_h=223
        if rightbottom_w<0:
            rightbottom_w=0
        if rightbottom_w>223:
            rightbottom_w=223
        out.append([lefttop_h,lefttop_w,rightbottom_h,rightbottom_w])
    return out
def generate_fasthead_train_data_v2(loc_data):
    pos_num=0
    loc_data=loc_data[0]
    #out_clsv=np.zeros((samplenum,21))
    #out_regv=np.zeros((samplenum,21,4))
    #cls_no=np.full((samplenum),-1)
    #reg_no=np.full((samplenum,21,4),-1)
    for i in range(len(loc_data)):
        ioulist=[]
        lefttop_h=loc_data[i][1]
        lefttop_w=loc_data[i][0]
        rightbottom_h=loc_data[i][3]
        rightbottom_w=loc_data[i][2]
        gt_cls=loc_data[i][4]
        gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
        #print(gt_box)
        for h in range(14):
            for w in range(14):
                for c in range(9):
                    boxtmp=generate_box(h,w,c)
                    #ioulist.append([IoU(gt_box,tmpbox),[h,w,c,gt_cls]])
                    iou=IoU(gt_box,boxtmp)
                    if iou>=0.5:
                        pos_num=pos_num+1
                    
    if pos_num>0:
        samplenum=pos_num*3
        
    else:
        print('no pos')
        samplenum=1
    #print(pos_num,samplenum)
    out_clsv=np.zeros((samplenum,2))
    out_regv=np.zeros((samplenum,4))
    cls_no=np.full((samplenum),-1)
    reg_no=np.full((samplenum,4),-1)
    
    boxlist=[]
    ser=0
    neglist=[]
    for h in range(14):
        for w in range(14):
            for c in range(9):
                boxtmp=generate_box(h,w,c)
                ifneg=0
                #ioulist.append([IoU(gt_box,tmpbox),[h,w,c,gt_cls]])
                for i in range(len(loc_data)):
                    ioulist=[]
                    lefttop_h=loc_data[i][1]
                    lefttop_w=loc_data[i][0]
                    rightbottom_h=loc_data[i][3]
                    rightbottom_w=loc_data[i][2]
                    gt_cls=loc_data[i][4]
                    gt_box=[lefttop_h,lefttop_w,rightbottom_h,rightbottom_w]
                    iou=IoU(gt_box,boxtmp)
                    if iou>=0.5:
                        anchor_lefttop_h=boxtmp[0]
                        anchor_lefttop_w=boxtmp[1]
                        anchor_rightbottom_h=boxtmp[2]
                        anchor_rightbottom_w=boxtmp[3]
                            
                        ya=(anchor_lefttop_h+anchor_rightbottom_h)/2
                        xa=(anchor_lefttop_w+anchor_rightbottom_w)/2
                        ha=anchor_rightbottom_h-anchor_lefttop_h
                        wa=anchor_rightbottom_w-anchor_lefttop_w
                            
                        if ha==0:
                            ha=0.01
                        if wa==0:
                            wa=0.01
                        
                        gt_lefttop_h=lefttop_h
                        gt_lefttop_w=lefttop_w
                        gt_rightbottom_h=rightbottom_h
                        gt_rightbottom_w=rightbottom_w
                            
                        gt_y=(gt_rightbottom_h+gt_lefttop_h)/2
                        gt_x=(gt_rightbottom_w+gt_lefttop_w)/2
                        gt_h=gt_rightbottom_h-gt_lefttop_h
                        gt_w=gt_rightbottom_w-gt_lefttop_w
                        ty=(gt_y-ya)/ha
                        tx=(gt_x-xa)/wa
                        tw=np.log(gt_w/wa)
                        th=np.log(gt_h/ha)
                        #print(ser,gt_cls)
                        gt_cls=int(gt_cls)
                        out_clsv[ser][0]=1
                        out_regv[ser][0]=ty
                        out_regv[ser][1]=tx
                        out_regv[ser][2]=th
                        out_regv[ser][3]=tw
                        cls_no[ser]=0
                        reg_no[ser][0]=0
                        reg_no[ser][1]=0
                        reg_no[ser][2]=0
                        reg_no[ser][3]=0
                        
                        boxlist.append(boxtmp)
                        ser=ser+1
                    if iou<=0.1:
                        ifneg=ifneg+1
                    if ifneg==len(loc_data):
                        neglist.append(boxtmp)
                        #out_clsv[ser][gt_cls]=1
                        #boxlist.append(boxtmp)
                        #cls_no[ser]=0
                        #reg_no[ser][gt_cls][0]=0
                        #reg_no[ser][gt_cls][1]=0
                        #reg_no[ser][gt_cls][2]=0
                        #reg_no[ser][gt_cls][3]=0
                        #ser=ser+1
    neg_num=0
    if samplenum>200:
        samplenum=200
    while neg_num<samplenum-pos_num:
        #print(ser)
        sjs=np.random.randint(0,len(neglist))
        out_clsv[ser][1]=1
        boxlist.append(neglist[sjs])
        cls_no[ser]=0
        #reg_no[ser][gt_cls][0]=0
        #reg_no[ser][gt_cls][1]=0
        #reg_no[ser][gt_cls][2]=0
        #reg_no[ser][gt_cls][3]=0
        ser=ser+1
        neg_num=neg_num+1
        out_regv=np.reshape(out_regv,[-1,4])
        reg_no=np.reshape(reg_no,[-1,4])
    boxlist=split_box(boxlist)
    return boxlist,out_clsv,out_regv,cls_no,reg_no
                
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
    image=cv2.resize(cv2.imread(image_paths[num]),(224,224))
    image=cv2.merge([image[:,:,2],image[:,:,1],image[:,:,0]])
    segment=cv2.resize(cv2.imread(segment_paths[num]),(224,224))
    segment=cv2.merge([segment[:,:,2],segment[:,:,1],segment[:,:,0]])
    segclass=cv2.resize(cv2.imread(segclass_paths[num]),(224,224))
    segclass=cv2.merge([segclass[:,:,2],segclass[:,:,1],segclass[:,:,0]])
    annotation=annotation_paths[num]
    return image,segment,segclass,annotation

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

def load_xml_list(annotation_path):
    DOMTree = xml.dom.minidom.parse(annotation_path)
    collection = DOMTree.documentElement
    size=collection.getElementsByTagName("size")[0]
    width=int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height=int(size.getElementsByTagName('height')[0].childNodes[0].data)
    widthratio=224/width
    heightratio=224/height
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
        objlist.append([objdict[name],[int(ymin),int(xmin),int(ymax),int(xmax)]])
    return objlist

def lxl2gld(lxl):
    res=[]
    for i in range(len(lxl)):
        restmp=[]
        for j in range(4):
            restmp.append(lxl[i][1][j])
            #print(restmp)
        restmp.append(lxl[i][0])
        res.append(restmp)
    return res

def trans_seg_to_list(seg,segclass,ann):
    #逻辑梳理：
    #1.生成zeromat
    #2.在segclass中查找box中类别所对应颜色的像素集合
    #3.在segobj中查找box中不同颜色的像素集合，挑选与segclass重合度最高的将zeromat对应位置置一
    #4.运行fourier_fit，查找出一系列cnlist（一个目标可能有多组轮廓）
    
    #期待输出：
    #[[objnum,[objpartn[coord]]]]
    #[[0,[[[y1,x1],[y2,x2]...],[y1,x1],[y2,x2]...],[1,[[...],...],...],...]]
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
                    colordict[color]=np.zeros((224,224))
                    colordict[color][i+ymin][j+xmin]=1
                else:
                    colordict[color][i+ymin][j+xmin]=1
        #查找segclass中box中类别所对应的颜色区域
        anncolor=class_color[objlist[n][0]]
        anncolormat=np.zeros((224,224))
        for i in range(ymax-ymin):
            for j in range(xmax-xmin):
                color=[]
                for k in range(3):
                    color.append(segclass[ymin+i][xmin+j][k])
                #print(color,anncolor)
                if color==anncolor:
                    anncolormat[i+ymin][j+xmin]=1
        #用anncolormat和colordict中每一个mat比较，挑选出重合度最高的mat
        ioucolorlist=[]
        
        for i in range(len(list(colordict.keys()))):
            iou=binImgIOU(anncolormat, colordict[list(colordict.keys())[i]])
            ioucolorlist.append([iou,list(colordict.keys())[i]])
        ioucolorlist=sorted(ioucolorlist,key=lambda x:x[0],reverse=True)
        choosedcolor=ioucolorlist[0][1]
        choosedmat=colordict[choosedcolor]
        
        #plt.figure(2*n)
        #plt.imshow(choosedmat)
        #plt.figure(2*n+1)
        #plt.imshow(anncolormat)
        #对重合度最高的mat运行fourier_fit,这里假定预测50个参数
        cnlists=fourier_fit(choosedmat,5)
        objlinelist.append([n,cnlists])
        
    return objlinelist

def binImgIOU(imga,imgb):
    counta=0
    countb=0
    countcoincide=0
    for i in range(len(imga)):
        for j in range(len(imga[0])):
            if imga[i][j]==1:
                counta+=1
            if imgb[i][j]==1:
                countb+=1
            if imga[i][j]==1 and imgb[i][j]==1:
                countcoincide+=1
    if countcoincide==0:
        countcoincide=1
    dived=counta+countb-countcoincide
    if dived==0:
        dived=0.1
    res=countcoincide/dived
    return res

def fourier_fit(pic,n,single=True):#目前只处理一个轮廓contours[0]
    #输入为二值图像
    pic=copy(pic)
    pic=pic.astype(np.uint8)
    res=[]
    nlis=[0]
    nlis+=[i+1 for i in range(n)]
    nlis+=[-i-1 for i in range(n)]
    nlis=sorted(nlis)
    if cv2.__version__!='4.4.0':
        binary,contours, hierarchy = cv2.findContours(pic,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #contours的是[y,x]坐标，而普通绘图用[x,y]坐标
    else:
        contours, hierarchy = cv2.findContours(pic,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #contours的是[y,x]坐标，而普通绘图用[x,y]坐标
    if single==True:
        cntlen=[]
        for i in range(len(contours)):
            cntlen.append(len(contours[i]))
        maxind=cntlen.index(max(cntlen))
        contours=[contours[maxind]]
    for c in range(len(contours)):
        contourtmp=contours[c]
        restmp=[]
        for k in range(len(nlis)):
            core_count=0+0j
            n_num=nlis[k]
            for i in range(len(contourtmp)):
                weight=np.e**((-1)*(n_num)*2*np.pi*(1/len(contourtmp))*i*1j)
                core_real=contourtmp[i][0][1]
                core_img=contourtmp[i][0][0]
                core_num=(core_real+core_img*1j)*weight
                core_count+=core_num
            complex_core=core_count/len(contourtmp)
            restmp.append(complex_core)
        res.append(restmp)
    
    return res

def copy(mat):
    res=np.zeros((len(mat),len(mat[0])))
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res[i][j]=mat[i][j]
    return res

def gen_train_ft_list(boxes,ocs,cls_no,ftlist,annotation_path):
    #思路整理：
    #1.输入为fasterhead网络输出的boxes和ftlist
    #首先需要将boxes和ftlist一一对应
    #具体需要将boxes中有目标的box与locdata匹配，再将locdata与ftlist匹配
    #*注：boxes为anchorbox
    xml=load_xml_list(annotation_path)
    ftout=[]
    for i in range(len(boxes)):
        ftout.append([])
    ft_no=np.full((len(boxes),11),-1)#该到这里了！！！！！！！！！！！！！！
    for i in range(len(cls_no)):
        if cls_no[i]!=-1:
            if ocs[i][0]==1:
                box=boxes[i]
                ioulist=[]
                for j in range(len(xml)):
                    gtbox=xml[j][1]
                    ioulist.append(IoU(box,gtbox))
                maxind=ioulist.index(max(ioulist))
                for j in range(len(ftlist)):
                    if ftlist[j][0]==maxind:
                        fttmp=ftlist[j][1]
                ftchange=copy_ft(fttmp)
                ftminus=box[0]+box[1]*j
                for j in range(len(ftchange)):
                    ftchange[j][5]-=ftminus
                ftout[i]=ftchange
                ft_no[i]=0
            else:
                ft_no[i]=0
    res=np.zeros((len(boxes),11,2))
    for i in range(len(ftout)):
        if ftout[i]!=[]:
            for j in range(len(ftout[i][0])):
                res[i][j][0]=ftout[i][0][j].real
                res[i][j][1]=ftout[i][0][j].imag
    return res,ft_no
    
def gen_train_ft_list_v2(ftlist,boxes,gld):
    #思路整理：
    #循环求出每个line和box的中心，一一匹配
    #
    gld=gld[0]
    ftout=[]
    ftline=[]
    #ftcenter=[]
    boxind=[]
    for i in range(len(ftlist)):
        fttmp=[]
        for j in range(len(ftlist[i][1][0])):
            ftr=ftlist[i][1][0][j].real
            ftj=ftlist[i][1][0][j].imag
            fttmp.append([ftr,ftj])
        ftline.append(fttmp)
    for i in range(len(boxes)):
        ioulis=[]
        box=boxes[i]
        for j in range(len(gld)):
            iou=IoU(box,gld[j])
            ioulis.append(iou)
        #iousor=sorted(ioulis,reverse=True)
        print(ioulis)
        ind=ioulis.index(max(ioulis))
        for j in range(len(ftlist)):
            if ftlist[j][0]==ind:
                fttmp=ftline[j]
                #print(fttmp[50],box)
                for k in range(len(fttmp)):
                    if k!=5:
                        fttmp[k][0]=fttmp[k][0]
                        fttmp[k][1]=fttmp[k][1]
                #fttmp[5][0]=(fttmp[5][0]-box[0])
                #fttmp[5][1]=(fttmp[5][1]-box[1]) 
                ftout.append(fttmp)
                boxind.append(ind)
    print(boxind)
    print(len(ftout))
    ftres=np.zeros((len(ftout),11,2))
    for i in range(len(ftout)):
        box=boxes[boxind[i]]
        #print(box)
        for j in range(len(ftout[i])):
            if j!=5:
                ftres[i][j][0]=ftout[i][j][0]
                ftres[i][j][1]=ftout[i][j][0]
        ftres[i][5][0]=ftout[i][5][0]-box[0]
        ftres[i][5][1]=ftout[i][5][1]-box[1]
        
            #print(ftout[i][j])
    return ftres

def gen_train_ft_list_v3(ftlist,boxes):
    #思路整理：
    #循环求出每个line和box的中心，一一匹配
    #
    ftout=[]
    ftline=[]
    #ftcenter=[]
    boxind=[]
    ftcenter=[]
    for i in range(len(ftlist)):
        fttmp=[]
        for j in range(len(ftlist[i][1][0])):
            ftr=ftlist[i][1][0][j].real
            ftj=ftlist[i][1][0][j].imag
            fttmp.append([ftr,ftj])
        ftline.append(fttmp)
        ftcenter.append(fttmp[5])
    for i in range(len(boxes)):
        box=boxes[i]
        boxcenter=[(box[0]+box[2])/2,(box[1]+box[3])/2]
        dist=[]
        for j in range(len(ftcenter)):
            dist.append((boxcenter[0]-ftcenter[j][0])**2+(boxcenter[1]-ftcenter[j][1])**2)
        ind=dist.index(min(dist))
        ft_select=ftline[ind]
        #ft_select[5][0]-=box[0]
        #ft_select[5][1]-=box[1]
        ft_select[5][0]=0
        ft_select[5][1]=0
        ftout.append(ft_select)
        
            #print(ftout[i][j])
    return ftout

def gen_train_ft_list_v4(ftlist,boxes):
    #思路整理：
    #循环求出每个line和box的中心，一一匹配
    #
    ftout=[]
    
    ftline=[]
    #ftcenter=[]
    ftcenter=[]
    for i in range(len(ftlist)):
        fttmp=[]
        for j in range(len(ftlist[i][1][0])):
            ftr=ftlist[i][1][0][j].real
            ftj=ftlist[i][1][0][j].imag
            fttmp.append([ftr,ftj])
        ftline.append(fttmp)
        ftcenter.append(fttmp[5])
    for i in range(len(boxes)):
        box=boxes[i]
        boxcenter=[(box[0]+box[2])/2,(box[1]+box[3])/2]
        dist=[]
        for j in range(len(ftcenter)):
            dist.append((boxcenter[0]-ftcenter[j][0])**2+(boxcenter[1]-ftcenter[j][1])**2)
        ind=dist.index(min(dist))
        ft_select=ftline[ind]
        #ft_select[5][0]-=box[0]
        #ft_select[5][1]-=box[1]
        ft_select[5][0]=0
        ft_select[5][1]=0
        ftout.append(ft_select)
    ftno=np.zeros((len(ftout),11))
    for i in range(len(ftout)):
        for j in range(11):
            if abs(ftout[i][j][0])>1 or abs(ftout[i][j][1])>1:
                ftno[i][j]=1
        sjs=np.random.randint(0,11)
        if sjs!=5:
            ftno[i][sjs]=1
            #print(ftout[i][j])
    
    return ftout,ftno

def copy_ft(lis):
    res=[]
    for i in range(len(lis)):
        restmp=[]
        for j in range(len(lis[i])):
            restmp.append(lis[i][j])
        res.append(restmp)
    return res

def load_train_data_v2(num):
    #####################
    #require            #
    #1.img              #
    #2.seg              #
    #3.segclass         #
    #4.annotation       #
    #5.ftlist           #
    #####################
    pdname='./train_data/pd_'+str(num)+'.npy'
    ocsname='./train_data/rpn_ocs_'+str(num)+'.npy'
    orsname='./train_data/rpn_ors_'+str(num)+'.npy'
    noname='./train_data/rpn_no_'+str(num)+'.npy'
    rnname='./train_data/rpn_rn_'+str(num)+'.npy'
    gldname='./train_data/gld_'+str(num)+'.npy'
    objlinelistname='./train_data/objlinelist_'+str(num)+'.npy'
    annoname='./train_data/annotation_'+str(num)+'.npy'
    
    
    pd=np.load(pdname)
    pd=np.reshape(pd,[1,224,224,3]).astype(np.float32)
    ocs=np.load(ocsname).astype(np.float32)
    ors=np.load(orsname).astype(np.float32)
    no=np.load(noname).astype(np.float32)
    rn=np.load(rnname).astype(np.float32)
    gld=np.load(gldname).astype(np.float32)
    obj_line_list=np.load(objlinelistname,allow_pickle=True)
    annotation_path=str(np.load(annoname))
    return pd,ocs,ors,no,rn,obj_line_list,annotation_path,gld

def draw_v_box_for_train(pd,s_box,clsv,regv):#还缺少一个nms
    im=pd[0]
    regv=np.reshape(regv,[-1,4])
    cls_lis={}
    cls_box={}
    for i in range(2):
        cls_lis[i]=[]
        cls_box[i]=[]
    for i in range(len(clsv)):
        if clsv[i][0]-clsv[i][1]>0:
            clsvn=np.argmax(clsv[i])
            tmpbox=s_box[i]
            regs=regv[i]
            cor_box=correct_box(tmpbox,regs)
            #lth=cor_box[0]
            #ltw=cor_box[1]
            #rbh=cor_box[2]
            #rbw=cor_box[3]
            cls_lis[clsvn].append(clsv[i][clsvn])
            cls_box[clsvn].append(cor_box)
    #print(cls_lis)
    cor_boxlis=[]
    for i in range(1):
        if len(cls_lis[i])>0:
            boxlistmp=cls_box[i]
            conflist=cls_lis[i]
            #print(boxlistmp)
            #print(conflist)
            maxoutput=10
            iouthreshold=0.2
            cor_boxlis=run_nms(boxlistmp,conflist,maxoutput,iouthreshold)
    cor_boxlis=split_box(cor_boxlis)       
    return cor_boxlis  

def draw_fourier(cnlis,flat,size,offset=100,iscomplex=True):
    
    cnliscom=[]
    if iscomplex==False:
        for i in range(len(cnlis)):
            cnliscomtmp=[]
            for j in range(len(cnlis[i])):
                if j==5:
                    cnliscomtmp.append(cnlis[i][j][0]+offset+(cnlis[i][j][1]+offset)*1j)
                else:
                    cnliscomtmp.append(cnlis[i][j][0]+cnlis[i][j][1]*1j)
            cnliscom.append(cnliscomtmp)
        cnlis=cnliscom
    high=size[0]
    width=size[1]
    pic=np.zeros((high,width))
    nlis=[0]
    nlis+=[i+1 for i in range(int(len(cnlis[0])/2))]
    nlis+=[-i-1 for i in range(int(len(cnlis[0])/2))]
    nlis=sorted(nlis)
    for n in range(len(cnlis)):
        cnlistmp=cnlis[n]
        for t in range(flat):
            nowpoint=[0,0]
            for i in range(len(cnlistmp)):
                point_i=(cnlistmp[i]*np.e**(nlis[i]*2*np.pi*t*(1/flat)*1j)).real+nowpoint[0]
                point_j=(cnlistmp[i]*np.e**(nlis[i]*2*np.pi*t*(1/flat)*1j)).imag+nowpoint[1]
                nowpoint=[point_i,point_j]
            picpoint_i=int(nowpoint[0])
            picpoint_j=int(nowpoint[1])
            if picpoint_i>size[0]-1:
                picpoint_i=size[0]-1
            if picpoint_i<0:
                picpoint_i=0
            if picpoint_j>size[1]-1:
                picpoint_j=size[1]-1
            if picpoint_j<0:
                picpoint_j=0
            pic[picpoint_i,picpoint_j]=1
    return pic
