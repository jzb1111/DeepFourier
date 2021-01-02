# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 08:21:41 2020

@author: asus
"""

from manimlib.imports import *
import numpy as np
#import matplotlib.pyplot as plt
import sys
#import importlib
#importlib.reload(sys)
print('ok')
sys.path.extend(['E:\\paper\\fourierDeepLearning','E:/paper/fourierDeepLearning'])
from utils.transutils import gencircle,genellipse,fourier_fit,draw_fourier,loadpic
#sys.path.append('E:\paper\fourierDeepLearning')
planepic=loadpic('img/plane.jpg',binaryReverse=True,canny=False)
#plt.figure(0)
#plt.imshow(planepic)
cnlist=fourier_fit(planepic,5)
for i in range(len(cnlist)):
    cnlist[i]=cnlist[i]/50

class manimAnimation(Scene):
    def construct(self):
        animflat=100
        #sys.path.extend(['E:/paper/fourierDeepLearning/'])
        self.solvelist(cnlist)
        #group=VGroup()
        for i in range(len(self.aniCirclelist)):
            #group=VGroup(group,self.aniCirclelist[i])
        #self.add(group)
            self.play(FadeIn(self.aniCirclelist[i]))
        #self.wait(5)
        rotateanimlist=[]
        for i in range(animflat):
            nowpoint=[0,0]
            for j in range(len(self.aniCirclelist)):
                if j%2==0:
                    rotateanimlist.append(Rotate(self.aniCirclelist[j],2*PI/animflat*i*np.ceil(j/2)))
                else:
                    rotateanimlist.append(Rotate(self.aniCirclelist[j],(-1)*2*PI/animflat*i*np.ceil(j/2)))
                

        self.play(*[rotateanim for rotateanim in rotateanimlist])
    
    def solvelist(self,cnlist):
        #zflist:zerofirstlist[0,-1,1,-2,2...]
        self.zflist=[]
        l=int(len(cnlist)/2)
        zero=cnlist[l]
        self.zflist.append(zero)
        for i in range(l):
            self.zflist.append(cnlist[l-i-1])
            self.zflist.append(cnlist[l+i+1])
        self.circlelist=[]
        #self.circlelist.append(self.zflist[0])
        self.circlelist.append([self.zflist[0],self.zflist[1]])
        for i in range(len(self.zflist)-2):
            self.circlelist.append([self.circlelist[i][0]+self.zflist[i+1],self.zflist[i+2]])
            #print(self.circlelist[i])
        self.aniCirclelist=[]
        veclist=[]
        for i in range(len(self.circlelist)):
            #print(self.circlelist[i][1])
            r=(self.circlelist[i][1].real**2+self.circlelist[i][1].imag**2)**0.5
            #if i==0:
            #center=DOWN*self.circlelist[i][0].real+RIGHT*self.circlelist[i][0].imag
            #circle_i=Circle(arc_center=center,radius=r,color=BLUE)
            #circle_i.shift(DOWN*self.circlelist[i][0].real+RIGHT*self.circlelist[i][0].imag)
            circle_i=Circle(radius=r,color=BLUE)
            
                
            arr_start=circle_i.get_center()
            arr_end=circle_i.get_center()+self.circlelist[i][1].real*DOWN+self.circlelist[i][1].imag*RIGHT
            
            arr=Arrow(start=arr_start,end=arr_end)
            veclist.append(arr)
            circle_arr_group=VGroup(circle_i,arr)
            circle_arr_group.save_state()
            
            def upd(obj):
                obj.restore()
                #obj.rotate(veclist[-1].get_angle())
                obj.shift(veclist[-1].get_vector())
                #print(veclist[-1].get_vector())
            
            if i!=0:
                circle_arr_group.add_updater(upd)
            self.aniCirclelist.append(circle_arr_group)
            last_arr=arr
            
            
class DtFourierScene(Scene):
    def construct(self):
        self.zflist=[]
        l=int(len(cnlist)/2)
        zero=cnlist[l]
        self.zflist.append(zero)
        for i in range(l):
            self.zflist.append(cnlist[l-i-1])
            self.zflist.append(cnlist[l+i+1])
        self.circlelist=[]
        self.circlelist.append([self.zflist[0],self.zflist[1]])
        for i in range(len(self.zflist)-2):
            self.circlelist.append([self.circlelist[i][0]+self.zflist[i+1],self.zflist[i+2]])
        self.aniCirclelist=[]
        veclist=[]
        axes = Axes()
        def anim1(obj,j):
            if j%2==0:
                obj.rotate(2*PI*np.ceil(j/2))#, about_point=ORIGIN)
            else:
                obj.rotate((-1)*2*PI*np.ceil(j/2))

        def anim2(obj,vec):
            obj.restore()
            #obj.rotate(-vec1.get_angle())
            obj.shift(vec.get_vector())
        for i in range(len(self.circlelist)):
            r=(self.circlelist[i][1].real**2+self.circlelist[i][1].imag**2)**0.5
            circle_i=Circle(radius=r,color=BLUE)
            arr_start=circle_i.get_center()
            arr_end=circle_i.get_center()+self.circlelist[i][1].real*DOWN+self.circlelist[i][1].imag*RIGHT
            arr=Arrow(start=arr_start,end=arr_end)
            veclist.append(arr)
            gup1 = VGroup(circle_i, arr)
            if i!=0:
                gup1.save_state()
                self.aniCirclelist[-1].add_updater(anim1,i)
                gup1.add_updater(anim2,veclist[-1])
            self.aniCirclelist.append(gup1)
            
            

        

        #gup1.add_updater(anim1)
        #gup2.add_updater(anim2)

        #path = TracedPath(vec2.get_end, stroke_width=6, stroke_color=ORANGE)
        #path.add_updater(lambda a, dt: a.shift(DOWN * dt))
        self.add(axes, gup1, gup2)#, path)
        self.wait(6)