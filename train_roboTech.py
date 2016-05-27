import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.misc import imsave, imrotate
import cPickle
import random
import time
import tensorflow as tf
import roboTechNN
import os
from scipy import ndimage

### Neural net parameters
height = 5 # size of convolutional filter height
width = 5 # size of convolutional filter width
learning_rate = 1e-4 # learning rate for optimizer
keep = .5 # keep probability for dropout during training
batch_size = 200 # size of training batches
RUNTHROUGHS = 50 #number of times to pass through the full data set
num_features1 = 100
num_features2 = 100
num_classes =2 
generate_new_data = False # for if we have a new batch of images for which I've never generated rotated examples
training = True

def get_file_names(ims_directory):
	""" Get the image file names in a directory"""
	list_of_files = [file for file in os.listdir(ims_directory) if (file.endswith('.jpg'))]
	return list_of_files

def generate_synthetic_data(ims_directory):
	"""Generate rotations and crops of data to increase the amount of training data provided"""
	list_of_files = get_file_names(ims_directory)
	imlist =[ndimage.imread(ims_directory+'/'+file, flatten=True) for file in list_of_files]
	for im,name in zip(imlist,list_of_files): # go through the image, rotate it 3 times, each by 90 degrees
		name = os.path.splitext(name)[0]
		for k in range(0,7):
			rot = imrotate(im,45*(k+1),interp='bilinear') # rotate 90 degrees
			imsave(ims_directory+'/'+name+'_'+str(k)+'.jpg',rot) # save the image

def pad_image(im):
	height = np.shape(im)[0]
	width = np.shape(im)[1]
	if height < 32 or width <32: # if a dimension is too small, pad it with 0's
		im = np.pad(im,((int(max(0,np.floor(float(32-height)/2))),
			int(max(0,np.ceil(float(32-height)/2)))),(int(max(0,np.floor(float(32-width)/2))),
			int(max(0,np.ceil(float(32-width)/2))))),
			'edge')
	height = np.shape(im)[0]
	width = np.shape(im)[1]
	if height > 32 or width > 32:
		im = im[max(0,(height-32)/2):(32+(height-32)/2),max(0,(width-32)/2):(32+(width-32)/2)]
	im = np.ndarray.flatten(im)
	return im

def curate_data(ims_directory):
	""" convert the training data into 32x32 images using 0-padding """
	list_of_files = get_file_names(ims_directory) # get the files
	imlist =[ndimage.imread(ims_directory+'/'+file, flatten=True) for file in list_of_files] # read image, turn it greyscale, add to list
	padded_ims = []
	# make a list of images padded or cropped to 32x32
	for im in imlist:
		padded_im = pad_image(im)
		padded_ims.append(padded_im)
	return padded_ims

def main():

	##### Train the neural network on the malware data
	print "running session..."
	sess = tf.Session()
	keep_prob = tf.placeholder(tf.float32)

	print "reading in data..."
	if generate_new_data:
		generate_synthetic_data('ims/Mating')
		generate_synthetic_data('ims/Negative')
	#test_ims = np.array(curate_data('ims')) # get the positive data
	#test_names = get_file_names('ims')
	#print test_names
	#num_test = np.shape(test_ims)[0]
	#test_ims = test_ims.astype(np.float)/255
	pos_ims = np.array(curate_data('ims/Mating')) # get the positive data
	num_pos = np.shape(pos_ims)[0]
	neg_ims = np.array(curate_data('ims/Negative')) # get the negative data
	num_neg = np.shape(neg_ims)[0]
	padded_ims = np.vstack((pos_ims,neg_ims)).astype(np.float)/255
	num_samples,num_features = np.shape(padded_ims)[0], np.shape(padded_ims)[1]
	labels = np.squeeze(np.vstack((np.ones((num_pos,1)),np.zeros((num_neg,1))))) # create the training labels

	print "data read... number of training samples: "+str(num_samples)
	x = tf.placeholder(tf.float32, shape=[None,num_features]) # the x input placeholder
	y_ = tf.placeholder(tf.int32, shape=[None]) # the label value placeholder

	print "instantiating neural net..."
	net = roboTechNN.inference(x,height=height,width=width,num_features1=num_features1,num_features2=num_features2,keep_prob=keep_prob) # create the tensorflow NN op with output labels_hat
	lossFn = roboTechNN.loss(net, y_, num_classes) # create the loss function op
	num_correct = roboTechNN.score(net,y_)
	train_NN = roboTechNN.train(lossFn,learning_rate=learning_rate) # train the NN
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()

	running_score = []

	if training:
		print "performing batch training..."
		for k in range(0,RUNTHROUGHS):
			print "RUNTHROUGH NUMBER: "+str(k+1)
			## Shuffle the order of the data
			order = range(0,num_samples)
			random.shuffle(order)
			x_shuff = padded_ims[order,:]
			labels_shuff = labels[order]
			print "shuffled"
			for i in range(0,num_samples/batch_size):
				x_batch = x_shuff[batch_size*i:batch_size*(i+1),:]
				y_batch = labels_shuff[batch_size*i:(batch_size*(i+1))]
				if (batch_size*i)%100 == 0 and i > 0 :
					x_eval = x_shuff[(batch_size*i)-100:(batch_size*i),:]
					y_eval = labels_shuff[(batch_size*i)-100:batch_size*(i)]
					score = num_correct.eval(session=sess,feed_dict={x:x_eval, y_:y_eval, keep_prob: 0.5})  # score the output on this batch, if unmodified
					print("step %d, fraction correct %g"%(i, float(score)/100))
				train_NN.run(session=sess,feed_dict={x: x_batch, y_: y_batch, keep_prob: keep}) # now actually train
			current_score = float(num_correct.eval(session=sess,feed_dict={x:padded_ims,y_:labels,keep_prob: 1.0}))/num_samples
			running_score.append(current_score)
			print "\n Runthrough score: %g"%(current_score)
		print("\nFinal fraction correct: %g"%(float(num_correct.eval(session=sess,feed_dict={x:padded_ims,y_:labels,keep_prob: 1.0}))/num_samples))
		guess = np.argmax(sess.run(net,feed_dict={x:padded_ims,y_:labels,keep_prob:1.0}),axis=1)
		j = 0 
		with open('./train_predictions.csv', 'w') as fout:
			fout.write('ID,Prediction,TrueClass\n')
			#for currPred in clazz:
			#	fout.write(str(j)+','+str(currPred)+'\n')
			#	j = j+1
			for logs, clazz in zip(guess,labels):
				fout.write(str(logs)+','+str(clazz)+'\n')
				j = j+1
		save_path = saver.save(sess, "model.ckpt")
		print("Model saved in file: %s" % save_path)

	else:
		saver.restore(sess, 'model.ckpt')
		print "model restored.."
		print "classifying samples..."
		#num_samples,num_features = np.shape(test_ims)[0], np.shape(test_ims)[1]
		print("Final fraction correct: %g"%(float(num_correct.eval(session=sess,feed_dict={x:padded_ims,y_:labels,keep_prob: 1.0}))/num_samples))
		#sess.run()

		logits = sess.run(net, feed_dict = {x:padded_ims, keep_prob: 1.0})
		clazz = np.argmax(logits,axis=1)
		j = 0
		with open('./ims_predictions.csv', 'w') as fout:
			fout.write('ID,Prediction\n')
			#for currPred in clazz:
			#	fout.write(str(j)+','+str(currPred)+'\n')
			#	j = j+1
			for logs, file in zip(clazz,test_names):
				fout.write(str(logs)+','+str(file)+'\n')
				j = j+1



"""	
	with open('running_score.csv', 'w') as fout:
		fout.write('Runthrough,Prediction\n')
		k = 1
		for score in running_score:
			fout.write(str(k)+','+str(score)+'\n')
			k = k + 1

	train_out = np.argmax(sess.run(net, feed_dict={x:x_train_full, keep_prob: 1.0}),axis=1) # classify
	with open ('nn_confusion.csv', 'w') as fout:
		fout.write('True class, Prediction\n')
		for ID,predict in zip(labels_train_full,train_out):
			fout.write(str(ID)+','+str(predict)+'\n')

	########## Use the RF classifier to determine if malware. If malware, pass it to the NN #########

	print "NN predicting on supposed malware"
	# return the output of the NN on the test set
	nn_softmax_test = sess.run(net, feed_dict={x: x_train_full, keep_prob: 1})
	nn_class = np.argmax(nn_softmax_test,axis=1)
	np.savetxt('trainoutnn.csv',nn_class,delimiter=',')

	print "Combining NN and RF predictions"

	Y_test_pred = ~Y_test_pred_RF*nn_class + 8*Y_test_pred_RF
	#Y_test_pred = nn_class
	with open('./train_predictions_nn.csv', 'w') as fout:
		fout.write('ID,Prediction\n')
		for currID, currPred in zip(ID_test_all, Y_test_pred):
			fout.write(str(currID)+','+str(currPred)+'\n')
"""

if __name__ == "__main__":
    main()