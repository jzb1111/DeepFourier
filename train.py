# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 08:40:12 2021

@author: asus
"""

import tensorflow as tf
#from VGG import vgg16
from resnet import ResNet
from RPN import R_P_N
from ROI_pool import ROIs,ROIs_v2,roi_box
from fast_head import fasthead
from FastFt import fast_ft
from nms_for_train import nms
from utils.get_loss import rpn_cls_loss,rpn_reg_loss,fasthead_cls_loss,fasthead_reg_loss,fast_ft_loss
from utils.datautil import gen_train_ft_list,generate_fasthead_train_sample_v2,file_name,load_train_data
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

config = tf.ConfigProto()#对session进行参数配置
config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction=0.7#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True

lr=0.001

xs=tf.placeholder(tf.float32,[1,224,224,3],name='input_xs')

eval_boxes=tf.placeholder(tf.int32,[None,4],name='boxes')
eval_boxes=tf.reshape(eval_boxes,[-1,4])

eval_anchorbox=tf.placeholder(tf.int32,[None,4],name='anchorbox')
eval_anchorbox=tf.reshape(eval_anchorbox,[-1,4])

gt_clsv=tf.placeholder(tf.float32,[None,2])
gt_regv=tf.placeholder(tf.float32,[None,4])
gt_clsvn=tf.placeholder(tf.float32,[None])
gt_regvn=tf.placeholder(tf.float32,[None])

gt_ft=tf.placeholder(tf.float32,[None,101,2])
gt_ftno=tf.placeholder(tf.float32,[None,101])

gt_rpncls=tf.placeholder(tf.float32,[1,14,14,9,2])
gt_rpnreg=tf.placeholder(tf.float32,[1,14,14,9,4])
gt_clsno=tf.placeholder(tf.float32,[1,14,14,9])
gt_regno=tf.placeholder(tf.float32,[1,14,14,9])

#vgg_net=vgg16(xs)
#vggout=vgg_net._build_model_()

vgg_net=ResNet(xs)
vggout=vgg_net._build_model()

rpn_net=R_P_N(vggout)
clsmap,regmap=rpn_net._build_model_()

clsmap=tf.reshape(clsmap,[1,14,14,9,2],name='clsmap')
regmap=tf.reshape(regmap,[1,14,14,9,4],name='regmap')

#for train:
#eval_boxes:the boxes which closest with gt_box

#roipool=ROIs(vggout,eval_boxes)#tf,tf#for train,we input some gt_box,and for use we input the real box
#rois=roipool._build_model()

roipool=ROIs_v2(vggout,eval_boxes)#tf,tf#for train,we input some gt_box,and for use we input the real box
rois=roipool._build_model()

fast_h=fasthead(rois)
clsv,regv=fast_h._build_model_()

clsv_=tf.reshape(clsv,[-1,2],name='clsv')
regv_=tf.reshape(regv,[-1,4],name='regv')

nms_f_t=nms(clsv,regv,eval_boxes)
clsout,boxout=nms_f_t._build_model_()

clsout=tf.reshape(clsout,[-1,2],name='clsout')
boxout=tf.reshape(boxout,[-1,4],name='boxout')

#新增fastFT
roipool_ft=ROIs_v2(vggout,eval_anchorbox)
rois_ft=roipool_ft._build_model()

fastft_h=fast_ft(rois_ft)
ftv=fastft_h._build_model_()
ftv_=tf.reshape(ftv,[-1,101,2],name='ftout')

rpnclsloss=rpn_cls_loss(clsmap,gt_rpncls,gt_clsno)
rpnregloss=rpn_reg_loss(regmap,gt_rpnreg,gt_regno)

rpnloss=rpnclsloss+rpnregloss

fastheadclsloss=fasthead_cls_loss(clsv,gt_clsv,gt_clsvn)
fastheadregloss=fasthead_reg_loss(regv,gt_regv,gt_regvn)
fastftloss=fast_ft_loss(ftv_,gt_ft,gt_ftno)

fastheadloss=fastheadclsloss+fastheadregloss

totalloss=rpnloss+fastheadloss+fastftloss
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(totalloss)
    train_rpn=tf.train.GradientDescentOptimizer(lr).minimize(rpnloss)

r,d,files=file_name('./VOC2007/SegmentationObject/')

#saver=tf.train.Saver()

init=tf.initialize_all_variables()

clsloss=[]
regloss=[]

#sboxlis=[]
#gtclsvlis=[]
with tf.Session(config=config) as sess:
    sess.run(init)
    for i in range(500001):
        sjs=np.random.randint(0,len(files))
        #pd,ocs,ors,no,rn,gld=load_train_data(sjs)
        #obj_line_list是傅里叶级数
        pd,ocs,ors,no,rn,obj_line_list,annotation_path=load_train_data(sjs)
        #trans_seg_to_list(seg,segclass,annotation_path)
        eva_clsmap=sess.run(clsmap,feed_dict={xs:pd})
        eva_regmap=sess.run(regmap,feed_dict={xs:pd})
        
        ###########################################
        #these four are for test
        roibox=roi_box(eva_clsmap,eva_regmap)
        s_anchor,s_boxes=roibox._build_model()
        s_boxes=np.array(s_boxes)
        s_boxes=s_boxes.astype(np.int32)
        s_anchor=np.array(s_anchor)
        s_anchor=s_anchor.astype(np.int32)
        #print('s_boxes',s_boxes)
        #sboxlis.append(s_boxes)
        gtclsv,gtregv,clsvno,regvno,flag=generate_fasthead_train_sample_v2(s_boxes,annotation_path)
        gtft,ft_no=gen_train_ft_list(s_anchor,gtclsv,clsvno,obj_line_list,annotation_path)
        #gtclsvlis.append(gtclsv)
        if flag==1:
            gtclsv=gtclsv.astype(np.float32)
            gtregv=gtregv.astype(np.float32)
            clsvno=clsvno.astype(np.float32)
            regvno=regvno.astype(np.float32)
            
            ###########################################
            
            ###########################################
            #these four are for train
            #s_boxes,gtclsv,gtregv,clsvno,regvno=generate_fasthead_train_data_v2(gld)
            #s_boxes=np.array(s_boxes).astype(np.int32)
            #gtclsv=gtclsv.astype(np.float32)
            #gtregv=gtregv.astype(np.float32)
            #clsvno=clsvno.astype(np.float32)
            #regvno=regvno.astype(np.float32)
            
            sess.run(train_step,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn,eval_boxes:s_boxes,gt_clsv:gtclsv,gt_regv:gtregv,gt_clsvn:clsvno,gt_regvn:regvno,eval_anchorbox:s_anchor,gt_ft:gtft,gt_ftno:ft_no})
            onerpnclsloss=sess.run(rpnclsloss,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn})
            onerpnregloss=sess.run(rpnregloss,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn})
            twofhclsloss=sess.run(fastheadclsloss,feed_dict={xs:pd,eval_boxes:s_boxes,gt_clsv:gtclsv,gt_regv:gtregv,gt_clsvn:clsvno,gt_regvn:regvno})
            twofhregloss=sess.run(fastheadregloss,feed_dict={xs:pd,eval_boxes:s_boxes,gt_clsv:gtclsv,gt_regv:gtregv,gt_clsvn:clsvno,gt_regvn:regvno})
            twofastftloss=sess.run(fastftloss,feed_dict={xs:pd,eval_anchorbox:s_anchor,gt_ft:gtft,gt_ftno:ft_no})
            threetotalloss=sess.run(totalloss,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn,eval_boxes:s_boxes,gt_clsv:gtclsv,gt_regv:gtregv,gt_clsvn:clsvno,gt_regvn:regvno,eval_anchorbox:s_anchor,gt_ft:gtft,gt_ftno:ft_no})
            print(i)
            print(threetotalloss,onerpnclsloss,onerpnregloss,twofhclsloss,twofhregloss)
        else:
            print('no pos')
            sess.run(train_rpn,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn})
            onerpnclsloss=sess.run(rpnclsloss,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn})
            onerpnregloss=sess.run(rpnregloss,feed_dict={xs:pd,gt_rpncls:ocs,gt_rpnreg:ors,gt_clsno:no,gt_regno:rn})
            print(onerpnclsloss,onerpnregloss)
        if i%10000==0:
            output_graph_def1=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['clsmap','regmap'])#sess.graph_def
            output_graph_def2=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['clsv','regv'])#sess.graph_def
            output_graph_def3=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['ftout'])
            #tflite_model = tf.lite.toco_convert(output_graph_def, [xs], [gpoutput])   #这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
            #open("./model/for_lite/model_mobile"+str(i)+".pb", "wb").write(output_graph_def)
            with tf.gfile.FastGFile('./model/rpn'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def1.SerializeToString())
            with tf.gfile.FastGFile('./model/fasthead'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def2.SerializeToString())
            with tf.gfile.FastGFile('./model/fastft'+str(i)+'.pb', mode = 'wb') as f:
                f.write(output_graph_def3.SerializeToString())
            