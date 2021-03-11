'''Train kernel methods on the MNIST dataset.
Should have tensorflow (>=1.2.1) and GPU device.
Run command:
	python run_expr.py
'''

from __future__ import print_function

import argparse
import collections
import keras
import numpy as np
import time

from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K

import kernels
import mnist
import utils

from backend_extra import hasGPU
from layers import KernelEmbedding, RFF, RELU
from optimizers import PSGD, SGD

from numpy import linalg as LA
import numpy as np
## extra starts here

import sys
sys.path.insert(0, '../picado2')

import sparkle

import picado.core.context as ctx
ctx.set_shared_var('max.cv.num', 1)

## extra ends here

##parameters

num_classes = 10	    # number of classes
label_ch_prob_all = np.array([ 0 ])
num_training_data = 10001

#regularizations = np.array([ 0.0001, 0.00005, 0.00001 ])
regularizations = np.array([ 0.0 ])
##


assert keras.backend.backend() == u'tensorflow', \
       "Requires Tensorflow (>=1.2.1)."
# assert hasGPU(), "Requires GPU."


for regularization in regularizations:
	## new add code start here
	from random import randint
	tr_score_all_CE = np.array([])
	te_score_all_CE = np.array([])

	tr_score_all_L2 = np.array([])
	te_score_all_L2 = np.array([])

	rkhs_norm_all = np.array([])

	for label_ch_prob in label_ch_prob_all:

		print ("label_ch_prob"+str(label_ch_prob))
		(x_train, y_train), (x_cv, y_cv), (x_test, y_test) = sparkle.load_mnist(num_training_data)

		index_one_train = np.nonzero( y_train )
		y_train = index_one_train[1]

		index_one_test = np.nonzero( y_test)
		y_test = index_one_test[1]

		##end add code

		ch_label_train_y_n = np.random.uniform(0,1,y_train.shape[0])
		ch_label_test_y_n = np.random.uniform(0,1,y_test.shape[0])
		def gen_random_label():
			return randint(0, num_classes-1 )

		for each_label_index in range(y_train.shape[0]):
			if ch_label_train_y_n[each_label_index] <= label_ch_prob :
				y_train[each_label_index] = gen_random_label()

		for each_label_index in range(y_test.shape[0]):
			if ch_label_test_y_n[each_label_index] <= label_ch_prob :
				y_test[each_label_index] = gen_random_label()
		# print "data label successfully corrupted"
				

		## new add code ends here
		
		n, D = x_train.shape    # (n_sample, n_feature)
		#d = np.int32(n / 2) * 2 # number of random features
		d_all = np.array([  10000, 11000, 12000, \
							13000, 14000, 20000, 40000, 60000])
		#d_all = np.array([ 8000, 8200, 8400, 8600, 8800, 9000, 9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600,\
		#					11800, 12000]) ## zoomed #RFF
		#d_all = np.array([ 300 , 500 ])
		# convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)
		  

		## start of orginal kernel matrix

		input_shape = (D,) # n_feature, (sample) index
		x1 = Input(shape=input_shape, dtype='float32', name='indexed-feat') # makes a input layer of (?,D)


		## end of original kernel matrix


		for d in d_all:

			print ("rff num : "+str(d))
			## get the kernel matrix corresponding to RFF

			relu_sample = np.float32( np.random.randn(D+1, d) ) ## this generates N(0,1) sample from all (orig_dim + 1)
			relu_weights = 1.0*relu_sample/(np.sqrt(  np.sum(  np.square( relu_sample  ) , axis=0 ) ) ) ## normalized vector 

			
			input_shape = (D,)
			x = Input(shape=input_shape, dtype='float32', name='feat') # shape of (?,D)
			rf_f = RELU(relu_weights, input_shape=input_shape)(x) # x can have any number of rows/data . 
			represe_rff = Model(x, rf_f) # given x, you will get feature matrix in RELU domain

			x_train_rff_represent =  represe_rff.predict(x_train)  ### the feature matrix in RELU domain; this is ( #data_points * #RFF )
			#print  ("RFF train matrix is : " + str(x_train_rff_represent.shape) )
			krff_train = np.matmul(x_train_rff_represent, x_train_rff_represent.transpose())  ## the kernel matrix in RFF domain; L2 space
			krff_train = krff_train + regularization*np.identity(n)
			print  ("RFF train kernel matrix is : " + str(krff_train.shape) )

			x_test_rff_represent =  represe_rff.predict(x_test)  ### the feature matrix in RFF domain; this is ( #data_points * #RFF )
			#print  ("RFF test matrix is : " + str(x_test_rff_represent.shape) )
			krff_test = np.matmul(x_test_rff_represent, x_train_rff_represent.transpose())  ## the kernel matrix in RFF domain; L2 space
			print  ("RFF test kernel matrix is : " + str(krff_test.shape) )
			### end of RFF kernel matrix

			

			alpha = np.linalg.solve(krff_train, y_train)  
			#alpha = np.linalg.pinv( krff_train  ).dot( y_train )
			#alpha = np.linalg.lstsq(krff_train, y_train)[0]
			print  ("weight shape : " + str(alpha.shape) )

			y_test_estimated = (krff_test).dot(alpha)
			y_train_estimated = (krff_train - regularization*np.identity(n)).dot(alpha)

			rkhs_norm = np.sqrt( np.trace( ((alpha.transpose()).dot(krff_train)).dot(alpha) ) )

			mse_test = np.sum(np.square(y_test_estimated - y_test))/y_test.shape[0]
			mse_train =  np.sum(np.square(y_train_estimated - y_train))/y_train.shape[0]
			ce_test = 100.0 - np.mean(np.equal(np.argmax(y_test, axis=-1), np.argmax(y_test_estimated, axis=-1)))*100
			ce_train = 100.0 - np.mean(np.equal(np.argmax(y_train, axis=-1), np.argmax(y_train_estimated, axis=-1)))*100

			print  ("mse_test : " + str(mse_test) )
			print  ("mse_train : " + str(mse_train) )
			print  ("ce_test : " + str(ce_test) )
			print  ("ce_train : " + str(ce_train) )
			print  ("rkhs_norm : "  + str(rkhs_norm) )

			tr_score_all_CE = np.append(tr_score_all_CE, ce_train )
			te_score_all_CE = np.append(te_score_all_CE, ce_test )
			tr_score_all_L2 = np.append(tr_score_all_L2, mse_train )
			te_score_all_L2 = np.append(te_score_all_L2, mse_test )
			rkhs_norm_all   = np.append(rkhs_norm_all  , rkhs_norm )
			
	

	import pickle
	def savepkl(fpath, data):
		with open(fpath, 'w') as f: 
			pickle.dump(data, f)


	dict1 = { 'rkhs_norm': rkhs_norm_all }
	dict_ce = { 'num_rff': d_all, 'train': tr_score_all_CE, 'test': te_score_all_CE }
	dict_l2 = { 'train': tr_score_all_L2, 'test': te_score_all_L2 }
	dict_comb = { 'ce': dict_ce, 'l2_loss': dict_l2  , 'dict1': dict1  }

	pickle_name = 'RELU'  + '_mnist_' + str(num_training_data) +'_reg_'+str(regularization) + '_dual'  + '.p'
	#pickle_name = str(args_dict['kernel']) + '_rff'  + '_mnist_directsolve_' + str(num_training_data) +'_reg_'+str(regularization) + '_zoomed' + '.p'
	dict_final = { 'RELU'  : dict_comb }
	savepkl( '../pickle/mnist/relu/dual/'+ pickle_name , dict_final ) ## i am here
	#savepkl( '../pickle/mnist/fourier_feature/sin_cos/zoomed/'+ pickle_name , dict_final ) ## i am here

	
	
	
