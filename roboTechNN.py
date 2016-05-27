#######
####### Code necessary for putting together the TF neural network underlying RoboTech
#######
####### SCT 03/01/2016

import pandas as pd
import numpy as np
import tensorflow as tf

def weight_variable(shape,std=.01):
  initial = tf.truncated_normal(shape, stddev=std)
  return tf.Variable(initial)

def bias_variable(shape,bias=-0.0):
  initial = tf.constant(bias, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, strides=[1,1,1,1],padding='SAME'):
  return tf.nn.conv2d(x, W, strides=strides,padding=padding)

def max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'):
  return tf.nn.max_pool(x,ksize=ksize,strides=strides,padding=padding)

def inference(x_input, height=5, width=5, num_features1=32, num_features2=64, features_fc=100,keep_prob=1.0,num_classes = 2):
  """ Build the net so we can use it for inference, then use logistic regression to classify the output """ 
  with tf.name_scope('layer1conv') as scope:
    x = tf.reshape(x_input, [-1, 32,32,1])
    weights = weight_variable([height, width, 1, num_features1])
    biases = bias_variable([num_features1])
    layer1 = tf.nn.relu(conv2d(x,weights)+biases)
  with tf.name_scope('layer1pool') as scope:
    pool1 = max_pool(layer1) #2x2 max pooling
  with tf.name_scope('layer2conv') as scope:
    weights = weight_variable([height,width, num_features1 ,num_features2])
    biases = bias_variable([num_features2])
    layer2 = tf.nn.relu(conv2d(pool1,weights)+biases)
  with tf.name_scope('layer2pool') as scope:
    pool2 = max_pool(layer2)
  with tf.name_scope('fullyconnected') as scope:
    weights = weight_variable([num_features2*8*8,features_fc])
    bias = bias_variable([features_fc])
    flat2 = tf.reshape(pool2, [-1,num_features2*8*8])
    fully_connected = tf.nn.relu(tf.matmul(flat2,weights)+bias)
    l2_drop = tf.nn.dropout(fully_connected,keep_prob) #implement dropout
  with tf.name_scope('softmax') as scope:
    weights = weight_variable([features_fc,2])
    biases = bias_variable([num_classes])
    logits = tf.nn.softmax(tf.matmul(fully_connected,weights)+biases)
  return logits

def loss(logits, labels,num_classes):
	""" Provides the loss function, using crossentropy loss """
  	# Convert from sparse integer labels in the range [0, NUM_CLASSSES)
  	# to 1-hot dense float vectors (that is we will have batch_size vectors,
  	# each with NUM_CLASSES values, all of which are 0.0 except there will
  	# be a 1.0 in the entry corresponding to the label).
  	batch_size = tf.size(labels)
  	labels = tf.expand_dims(labels, 1)
  	indices = tf.expand_dims(tf.range(0, batch_size), 1)
  	concated = tf.concat(1, [indices, labels])
  	onehot_labels = tf.sparse_to_dense(
      	concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)
  	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
    	                                                    onehot_labels,
  	                                                        name='xentropy')
  	loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  	return loss

def train(loss,learning_rate=1e-3):
	""" creates a training op using the loss function loss"""
	# Add a scalar summary for the snapshot loss.
 	tf.scalar_summary(loss.op.name, loss)
 	# Create the gradient descent optimizer with the given learning rate.
 	optimizer = tf.train.AdamOptimizer(learning_rate)
 	# Create a variable to track the global step.
 	global_step = tf.Variable(0, name='global_step', trainable=False)
 	# Use the optimizer to apply the gradients that minimize the loss
  	# (and also increment the global step counter) as a single training step.
  	train_op = optimizer.minimize(loss, global_step=global_step)
  	return train_op

def score(logits, labels):
	""" returns the success of the network on the examples in logits with corresponding classes labels """
	# For a classifier model, we can use the in_top_k Op.
	# It returns a bool tensor with shape [batch_size] that is true for
  	# the examples where the label's is was in the top k (here k=1)
  	# of all logits for that example.
  	correct = tf.nn.in_top_k(logits, labels, 1)
  	# Return the number of true entries.
  	return tf.reduce_sum(tf.cast(correct, tf.int32))