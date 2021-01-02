# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 08:10:49 2020

@author: -
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.transutils import gencircle,genellipse,fourier_fit,draw_fourier,loadpic

circle=gencircle([128,128],20,[256,256])
ellipes=genellipse([128,108],[128,148],50,[256,256])
#circlecnlis=fourier_fit(circle,5)
cnlis=fourier_fit(ellipes,15)

df=draw_fourier(cnlis,100,[256,256])

plt.figure(0)
plt.imshow(ellipes)

plt.figure(1)
plt.imshow(df)

planepic=loadpic('img/plane.jpg',binaryReverse=True,canny=False)
plt.figure(2)
plt.imshow(planepic)
pcnlis=fourier_fit(planepic,50)
pdf=draw_fourier(pcnlis,1000,[256,256])
plt.figure(3)
plt.imshow(pdf)