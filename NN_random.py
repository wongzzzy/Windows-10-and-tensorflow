# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
def add_layer(inputs,in_size,out_size,n_layer,activate_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer1'):
        with tf.name_scope('Weight'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name+'/Weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
            
    if activate_function == None:
        outputs = Wx_plus_b 
    else:
        outputs = activate_function(Wx_plus_b)
    tf.summary.histogram(layer_name+'outputs',outputs)
    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_inputs')
    ys = tf.placeholder(tf.float32,[None,1],name='y_inputs')

l1 = add_layer(xs, 1, 10, n_layer=1,activate_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1,n_layer=2, activate_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)



init = tf.initialize_all_variables()
merged = tf.summary.merge_all()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("D:/deep_learning/logs",tf.get_default_graph())
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%20==0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(result,i)