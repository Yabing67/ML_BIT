#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@author: Bing
"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution() #2.0与1.0版本不兼容
sess = tf.compat.v1.InteractiveSession() #创建交互式会话，tensorflow2.0版本中的确没有Session这个属性
a = tf.Variable([[1.0,2.0]]) #创建变量数组
b = tf.constant([[3.0,4.0]]) #创建常量数组
sess.run(tf.compat.v1.global_variables_initializer()) #变量初始化
res = tf.add(a,b) #创建加法操作
print(res.eval())