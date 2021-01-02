# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 07:57:32 2020

@author: asus
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time

inputcircle=[256,256]
circlecenter=[128,128]
circlelen=20

high=256
width=256
a=[128,118]
b=[128,138]
l=40
def gencircle(inputcircle,circlecenter,circlelen):
    circlepic=np.zeros((inputcircle[0],inputcircle[1]))
    for i in range(inputcircle[0]):
        for j in range(inputcircle[1]):
            dist=abs(((i-circlecenter[0])**2+(j-circlecenter[1])**2)**0.5-20)
            if dist<1:
                circlepic[i][j]=1
    return circlepic


def genellipse(a,b,l,high,width):
    pic=np.zeros((high,width))
    for i in range(high):
        for j in range(width):
            if abs(((i-a[0])**2+(j-a[1])**2)**0.5+((i-b[0])**2+(j-b[1])**2)**0.5-l)<1:
                pic[i][j]=1
    return pic

testcircle=gencircle(inputcircle,circlecenter,circlelen)
plt.figure(0)
plt.imshow(testcircle)

testellipse=genellipse(a,b,l,high,width)
plt.figure(1)
plt.imshow(testellipse)

def fourrier_fit(pic,n):
    e=np.e
    loop=n-1
    #core_real=0
    #core_img=0
    
    
    res=[]
    for k in range(n):
        core_count=0+0j
        count=0
        weight=e**((-1)*(k+1)*2*np.pi*1j)
        for i in range(len(pic)):
            for j in range(len(pic[i])):
                if pic[i][j]==1:
                    core_real=i
                    core_img=j
                    count+=1
                    core_num=(core_real+core_img*1j)*weight
                    core_count+=core_num
        complex_core=core_count/count
        res.append(complex_core)
    
    return res

def dist(p1,p2):
    #dist=0
    dist=((p1[0]-p2[0])**2+(p1[1]-p2[2])**2)**0.5
    return dist

def fourier_fit2(pic,n):#目前只处理一个轮廓contours[0]
    pic=pic.astype(np.uint8)
    res=[]
    #findcontour
    binary,contours, hierarchy = cv2.findContours(pic,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #contours的是yx坐标，而普通绘图用xy坐标
    #plt.figure(4)
    #处理0级
    core_count_zero=0+0j
    for i in range(len(contours[0])):
        core_real=contours[0][i][0][0]
        core_img=contours[0][i][0][1]
        core_num=core_real+core_img*1j
        core_count_zero+=core_num
    complex_core=core_count_zero/len(contours[0])
    res.append(complex_core)    
    for k in range(n):
        core_pos_count=0+0j
        core_neg_count=0+0j
        #count=0
        #for i in range(flat):
        for i in range(len(contours[0])):
            #if pic[i][j]==1:
            weight_pos=np.e**((-1)*(k+1)*2*np.pi*(1/len(contours[0]))*i*1j)
            weight_neg=np.e**((k+1)*2*np.pi*(1/len(contours[0]))*i*1j)
            
            #print(weight)
            core_real=contours[0][i][0][1]
            core_img=contours[0][i][0][0]
            #count+=1
            core_pos_num=(core_real+core_img*1j)*weight_pos
            core_neg_num=(core_real+core_img*1j)*weight_neg
            #print(core_real,core_img,core_num)
            core_pos_count+=core_pos_num
            core_neg_count+=core_neg_num
        complex_pos_core=core_pos_count/len(contours[0])
        complex_neg_core=core_neg_count/len(contours[0])
        
        res.append([complex_neg_core,complex_pos_core])
        #res.append(complex_pos_core)
        
    return res,contours

#cnt_=fourier_fit2(testcircle,5)

def draw_fourier(cnlis,flat,high,width):
    pic=np.zeros((high,width))
    
    for t in range(flat):
        nowpoint=[0,0]
        for i in range(len(cnlis)):
            point_i=(cnlis[i]*np.e**(i*2*np.pi*t*(1/flat)*1j)).real+nowpoint[0]
            point_j=(cnlis[i]*np.e**(i*2*np.pi*t*(1/flat)*1j)).imag+nowpoint[1]
            nowpoint=[int(point_i),int(point_j)]
            #print(point_i,point_j,nowpoint)
            pic[int(point_i)][int(point_j)]=1
    return pic

def draw_fourier2(cnlis,flat,size):
    high=size[0]
    width=size[1]
    pic=np.zeros((high,width))
    
    for t in range(flat):
        nowpoint=[0,0]
        for i in range(len(cnlis)):
            if i == 0 :    
                point_i=(cnlis[i]*np.e**(i*2*np.pi*t*(1/flat)*1j)).real+nowpoint[0]
                point_j=(cnlis[i]*np.e**(i*2*np.pi*t*(1/flat)*1j)).imag+nowpoint[1]
            else:
                point_neg_i=(cnlis[i][0]*np.e**(i*2*np.pi*t*(1/flat)*1j)).real+nowpoint[0]
                point_pos_i=(cnlis[i][1]*np.e**((-1)*i*2*np.pi*t*(1/flat)*1j)).real+point_neg_i
                point_neg_j=(cnlis[i][0]*np.e**(i*2*np.pi*t*(1/flat)*1j)).imag+nowpoint[1]
                point_pos_j=(cnlis[i][1]*np.e**((-1)*i*2*np.pi*t*(1/flat)*1j)).imag+point_neg_j
                point_i=point_pos_i
                point_j=point_pos_j
            nowpoint=[int(point_i),int(point_j)]
            print(point_i,point_j,nowpoint)
            pic[int(point_i)][int(point_j)]=1
    return pic

#testcnlis=[120+120j,20+20j]
#cnlis=fourier_fit2(testcircle,5)
cnlis,cnts=fourier_fit2(testellipse,10)

fpic=draw_fourier2(cnlis,15,[256,256])
plt.figure(2)
plt.imshow(fpic)