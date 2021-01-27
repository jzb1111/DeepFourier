# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 18:16:02 2021

@author: asus
"""

import tensorflow as tf

anchorlist=tf.placeholder(tf.float32,[None,4],name='anchorlist')
boxlist=tf.placeholder(tf.float32,[None,4],name='boxlist')
conflist=tf.placeholder(tf.float32,[None],name='conflist')
maxoutput=tf.placeholder(tf.int32,name='maxoutput')
iouthreshold=tf.placeholder(tf.float32,name='iouthreshold')



nms=tf.image.non_max_suppression(boxlist,conflist,max_output_size=maxoutput,iou_threshold=iouthreshold,name='nonms')

nmsbox=tf.gather(boxlist,nms,name='nms')
nmsanchor=tf.gather(anchorlist,nms,name='nms_anchor')
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #nnm=sess.run(nms,feed_dict={box:a,conf:b})
    output_graph_def=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['nms','nms_anchor'])
    #print(sess.graph)
    with tf.gfile.FastGFile('./nms.pb', mode = 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    #builder = tf.saved_model.builder.SavedModelBuilder('./model/nmsmodel/')
    #builder.add_meta_graph_and_variables(sess, ['tag_string'])
    #builder.save()