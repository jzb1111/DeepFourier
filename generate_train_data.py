# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:05:27 2019

@author: ADMIN
"""

import numpy as np
#import cv2
from support.read_data import file_name,get_batch_by_num,get_locdata,generate_rpn_train_data_v2,load_raw_data_by_num,load_xml_list,lxl2gld,trans_seg_to_list

#r,d,f=file_name('./VOC2007/Annotations/')
r,d,f=file_name('./VOC2007/SegmentationObject/')

for i in range(len(f)):
    #xml,pd=get_batch_by_num(i)
    #gld=get_locdata(xml)
    pd,segment,segclass,annotation=load_raw_data_by_num(i)
    pd=np.array([pd])
    lxl=load_xml_list(annotation)
    gld=lxl2gld(lxl)
    gld=[gld]
    ocs,ors,no,rn=generate_rpn_train_data_v2(gld)
    objlinelist=trans_seg_to_list(segment,segclass,annotation)
    
    np.save('./train_data/pd_'+str(i)+'.npy',pd)
    np.save('./train_data/rpn_ocs_'+str(i)+'.npy',ocs)
    np.save('./train_data/rpn_ors_'+str(i)+'.npy',ors)
    np.save('./train_data/rpn_no_'+str(i)+'.npy',no)
    np.save('./train_data/rpn_rn_'+str(i)+'.npy',rn)
    np.save('./train_data/gld_'+str(i)+'.npy',gld)
    np.save('./train_data/ft_'+str(i)+'.npy',objlinelist)
    print(i)