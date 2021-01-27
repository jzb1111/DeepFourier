# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 08:10:49 2020

@author: -
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.transutils import gencircle,genellipse,fourier_fit,draw_fourier,loadpic

circle=gencircle([128,128],20,[224,224])
ellipes=genellipse([128,108],[128,148],50,[224,224])
#circlecnlis=fourier_fit(circle,5)
cnlis=fourier_fit(ellipes,15)

df=draw_fourier(cnlis,100,[224,224])

plt.figure(0)
plt.imshow(ellipes)

plt.figure(1)
plt.imshow(df)

planepic=loadpic('img/changan.jpg',binaryReverse=True,binaryThresholdFactor=0.8,canny=False)
plt.figure(2)
plt.imshow(planepic)
#fourier_fit(img,级数的个数)
pcnlis=fourier_fit(planepic,50)
#draw_fourier(级数序列,先挑精细度（多少个点组成线条）,[画布大小])
pdf=draw_fourier(pcnlis,1000,[224,224])
plt.figure(3)
plt.imshow(pdf)