#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@author: Bing
"""
import tensorflow as tf
from numpy import array
from numpy import float32

tf.compat.v1.disable_eager_execution() #2.0与1.0版本不兼容
"""或使用以下这两句"""
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
sess = tf.compat.v1.InteractiveSession() #创建交互式会话，tensorflow2.0版本中的确没有Session这个属性
input1 = tf.compat.v1.placeholder(tf.float32) #创建占位符
input2 = tf.compat.v1.placeholder(tf.float32)
res = tf.multiply(input1,input2) #创建乘法
print(res.eval(feed_dict = {input1:[7.],input2:[7.]}))
#求值。对feed_dict参数赋值
print(array([14.],dtype = float32))
