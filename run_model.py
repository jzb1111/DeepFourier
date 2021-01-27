# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:32:29 2019

@author: ADMIN
"""

import tensorflow as tf
import os

def run_rpn(pd):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    config = tf.ConfigProto()#对session进行参数配置
    config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction=0.7#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        output_graph_def=tf.GraphDef()
        
        with open('./model/rpn300000.pb',"rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ =tf.import_graph_def(output_graph_def,name='')
            
        with tf.Session(config=config) as sess:
            sess.graph.as_default()
            init=tf.global_variables_initializer()
            sess.run(init)
            
            xs=sess.graph.get_tensor_by_name("input_xs:0")
            
            rpncls=sess.graph.get_tensor_by_name("clsmap:0")
            rpnreg=sess.graph.get_tensor_by_name("regmap:0")
            
            
            clsout=sess.run(rpncls,feed_dict={xs:pd})   
            regout=sess.run(rpnreg,feed_dict={xs:pd})
    return clsout,regout

def run_fasthead(pd,eval_boxes):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    config = tf.ConfigProto()#对session进行参数配置
    config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction=0.7#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        output_graph_def=tf.GraphDef()
        
        with open('./model/fasthead300000.pb',"rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ =tf.import_graph_def(output_graph_def,name='')
            
        with tf.Session(config=config) as sess:
            sess.graph.as_default()
            init=tf.global_variables_initializer()
            sess.run(init)
            
            xs=sess.graph.get_tensor_by_name("input_xs:0")
            eva_box=sess.graph.get_tensor_by_name("boxes:0")
            
            clsv=sess.graph.get_tensor_by_name("clsv:0")
            boxout=sess.graph.get_tensor_by_name("regv:0")
            
            
            clsv=sess.run(clsv,feed_dict={xs:pd,eva_box:eval_boxes})   
            regv=sess.run(boxout,feed_dict={xs:pd,eva_box:eval_boxes})
    return clsv,regv

def run_fasthead_nms(e_clsv,e_regv,e_boxes):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    config = tf.ConfigProto()#对session进行参数配置
    config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction=0.7#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        output_graph_def=tf.GraphDef()
        
        with open('./fasthead_nms.pb',"rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ =tf.import_graph_def(output_graph_def,name='')
            
        with tf.Session(config=config) as sess:
            sess.graph.as_default()
            init=tf.global_variables_initializer()
            sess.run(init)
            
            clsv=sess.graph.get_tensor_by_name("clsv:0")
            regv=sess.graph.get_tensor_by_name("regv:0")
            boxes=sess.graph.get_tensor_by_name("boxes:0")
            
            clsout=sess.graph.get_tensor_by_name("clsout:0")
            boxout=sess.graph.get_tensor_by_name("boxout:0")
            
            cls_out=sess.run(clsout,feed_dict={clsv:e_clsv,regv:e_regv,boxes:e_boxes})   
            box_out=sess.run(boxout,feed_dict={clsv:e_clsv,regv:e_regv,boxes:e_boxes})
    return cls_out,box_out

def run_fast_ft(pd,eval_anchorbox):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    config = tf.ConfigProto()#对session进行参数配置
    config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction=0.7#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        output_graph_def=tf.GraphDef()
        
        with open('./model/fastft300000.pb',"rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ =tf.import_graph_def(output_graph_def,name='')
            
        with tf.Session(config=config) as sess:
            sess.graph.as_default()
            init=tf.global_variables_initializer()
            sess.run(init)
            
            xs=sess.graph.get_tensor_by_name("input_xs:0")
            eva_box=sess.graph.get_tensor_by_name("anchorbox:0")
            
            #clsv=sess.graph.get_tensor_by_name("clsv:0")
            ftout=sess.graph.get_tensor_by_name("ftout:0")
            
            
            #clsv=sess.run(clsv,feed_dict={xs:pd,eva_box:eval_boxes})   
            ftv=sess.run(ftout,feed_dict={xs:pd,eva_box:eval_anchorbox})
    return ftv