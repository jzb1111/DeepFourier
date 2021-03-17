# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 21:58:39 2021

@author: asus
"""

import tensorflow as tf
import tensorflow.contrib as tc

class fast_ft():
    def __init__(self,rois):
        #self.xs=xs
        self.rois=rois
        #self.boxes=boxes
        
    def _build_model_(self):
        self.rois=tf.reshape(self.rois,[-1,300])
        #self.rois=tf.reshape(self.rois,[1,-1])
        bfc=self.base_fc(self.rois)
        clsv=self.cls_vector(bfc)
        #regv=self.reg_vector(bfc)
        return clsv#,regv
    
    def base_fc(self,rois):
        out=tc.layers.fully_connected(rois,512)
        out=tc.layers.fully_connected(out,512)
        return out
    
    def cls_vector(self,vector):
        out=tc.layers.fully_connected(vector,22,activation_fn=None)
        out=tf.reshape(out,[-1,11,2])
        #out=tf.nn.softmax(out)
        #output=tf.reshape(tf.nn.softmax(tf.reshape(output,(-1,2))),(-1,28,28,2))
        return out
    