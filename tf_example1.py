#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
@author: Bing
"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution() #2.0与1.0版本不兼容
mat1 = tf.constant([[3.,3.]]) #创建矩阵
mat2 = tf.constant([[2.],[2.]])
product = tf.matmul(mat1,mat2) #创建op执行矩阵乘法
sess = tf.compat.v1.Session() #启动默认图，tensorflow2.0版本中的确没有Session这个属性
res = sess.run(product) #在默认图中进行op操作
print(res)
sess.close()