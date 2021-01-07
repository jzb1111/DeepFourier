# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 09:48:42 2020

@author: -
"""

import numpy as np
import cv2


def loadpic(url,binary=False,binaryReverse=False,canny=False):
    img=cv2.imread(url)
    img=cv2.resize(img,(256,256))
    if binary:
        img=cv2.threshold(img,np.max(img)/2,1,cv2.THRESH_BINARY)[1]
    if binaryReverse:
        img=cv2.threshold(img,np.max(img)/2,1,cv2.THRESH_BINARY_INV)[1]
    if canny:
        img= cv2.Canny(img,0.2,0.8)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return img

def copy(mat):
    res=np.zeros((len(mat),len(mat[0])))
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res[i][j]=mat[i][j]
    return res

def gencircle(circlecenter,circlelen,size):
    inputcircle=size
    circlepic=np.zeros((inputcircle[0],inputcircle[1]))
    for i in range(inputcircle[0]):
        for j in range(inputcircle[1]):
            dist=abs(((i-circlecenter[0])**2+(j-circlecenter[1])**2)**0.5-20)
            if dist<1:
                circlepic[i][j]=1
    return circlepic


def genellipse(a,b,l,size):
    high=size[0]
    width=size[1]
    pic=np.zeros((high,width))
    for i in range(high):
        for j in range(width):
            if abs(((i-a[0])**2+(j-a[1])**2)**0.5+((i-b[0])**2+(j-b[1])**2)**0.5-l)<1:
                pic[i][j]=1
    return pic

def fourier_fit(pic,n):#目前只处理一个轮廓contours[0]
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

    for k in range(len(nlis)):
        core_count=0+0j
        n_num=nlis[k]
        for i in range(len(contours[0])):
            weight=np.e**((-1)*(n_num)*2*np.pi*(1/len(contours[0]))*i*1j)
            core_real=contours[0][i][0][1]
            core_img=contours[0][i][0][0]
            core_num=(core_real+core_img*1j)*weight
            core_count+=core_num
        complex_core=core_count/len(contours[0])
        res.append(complex_core)
    return res

def draw_fourier(cnlis,flat,size):
    high=size[0]
    width=size[1]
    pic=np.zeros((high,width))
    nlis=[0]
    nlis+=[i+1 for i in range(int(len(cnlis)/2))]
    nlis+=[-i-1 for i in range(int(len(cnlis)/2))]
    nlis=sorted(nlis)
    
    for t in range(flat):
        nowpoint=[0,0]
        for i in range(len(cnlis)):
            point_i=(cnlis[i]*np.e**(nlis[i]*2*np.pi*t*(1/flat)*1j)).real+nowpoint[0]
            point_j=(cnlis[i]*np.e**(nlis[i]*2*np.pi*t*(1/flat)*1j)).imag+nowpoint[1]
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