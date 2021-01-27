# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 16:07:18 2021

@author: asus
"""

import numpy as np
#import cv2

from utils.datautil import load_raw_data_by_num,file_name,generate_rpn_train_data,get_locdata,trans_seg_to_list,load_xml_list,lxl2gld

#####################
#require            #
#1.img              #
#2.seg              #
#3.segclass         #
#4.annotation       #
#5.ftlist           #
#####################

r,d,f=file_name('./VOC2007/SegmentationObject/')

for i in range(len(f)):
    image,segment,segclass,annotation=load_raw_data_by_num(i)
    lxl=load_xml_list(annotation)
    gld=lxl2gld(lxl)
    ocs,ors,no,rn=generate_rpn_train_data(annotation)
    objlinelist=trans_seg_to_list(segment,segclass,annotation)
    
    
    np.save('./train_data/pd_'+str(i)+'.npy',image)
    np.save('./train_data/rpn_ocs_'+str(i)+'.npy',ocs)
    np.save('./train_data/rpn_ors_'+str(i)+'.npy',ors)
    np.save('./train_data/rpn_no_'+str(i)+'.npy',no)
    np.save('./train_data/rpn_rn_'+str(i)+'.npy',rn)
    np.save('./train_data/gld_'+str(i)+'.npy',gld)
    np.save('./train_data/objlinelist_'+str(i)+'.npy',objlinelist)
    np.save('./train_data/annotation_'+str(i)+'.npy',annotation)
    
    print(i)