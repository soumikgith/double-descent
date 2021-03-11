
from __future__ import print_function

import argparse
import collections
import keras
import numpy as np
import time
import math
import gc

from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K

import kernels
import mnist
import utils

from backend_extra import hasGPU

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler , EarlyStopping , Callback
from keras.initializers import glorot_uniform 
from keras import regularizers
from sklearn.decomposition import PCA

## extra starts here

import sys
sys.path.insert(0, '../picado2')

import sparkle

import picado.core.context as ctx
ctx.set_shared_var('max.cv.num', 1)

## extra ends here
num_classes = 10 
#label_ch_prob_all = np.array([ 0 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1 ])
label_ch_prob_all = np.array([ 0.0 ])
all_epochs =   6000 
num_training_data = 400*10

hidden_units = np.array([  2, 3, 4, 5, 8, 12, 18, 24, 28, 30, 32, 35, 38, 40, 45, 48, 50, 52, 55, 60 , 70, 100 , 200 , 300 , 1000 ]) ## 50 is traget 800*50 
#hidden_units = np.array([33,  35, 38, 40, 50, 60 , 100 , 200 , 300 , 1000 ])
#hidden_units = np.array([   5 ])
#hidden_units = np.array([  5 , 18 , 24 ])

pca_status = False
# if pca_status:
# 	pca_comp = 100
batch_size = 1000     ## because we need gradient descent
num_iterations = 5
## normal
#initial_lrates = np.array([   0.01    ])
## regul
initial_lrates = np.array([  0.0001 , 0.01 , 0.01 , 0.01  ,  0.01 , 0.01 , 0.01 , 0.02 , 0.01,  0.01 , 0.01 , 0.01, 0.02 , 0.02 , 0.02 , 0.03 ,  0.03 , 0.03 , 0.04 , 0.02 , 0.03 , 0.02 , 0.02 , 0.02 , 0.02  ])
#initial_lrates = np.array([ 0.03 ,  0.03 , 0.03 , 0.04 , 0.02 , 0.03 , 0.02 , 0.02 , 0.02 , 0.02  ])
weights_ini_seeds = np.array([ 12 , 5, 8, 13, 22 , 1 ])

regul = 0.0

# for initialization of layers 
mu_ini = 0
sigma_ini = 0.1


assert keras.backend.backend() == u'tensorflow', \
       "Requires Tensorflow (>=1.2.1)."
# assert hasGPU(), "Requires GPU."
## new add code start here
from random import randint


for label_ch_prob in label_ch_prob_all:

	print ("label_ch_prob: "+str(label_ch_prob))
	## we are interested in only 2 classes as of now
	(x_train, y_train), (x_cv, y_cv), (x_test, y_test) = sparkle.load_mnist(num_training_data)

	index_one_train = np.nonzero( y_train )
	y_train = index_one_train[1]

	index_one_test = np.nonzero( y_test)
	y_test = index_one_test[1]
	
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


	## get the filtered version of each image and features are (7*7) now .
	# filter_size = 4
	# x_train_inter = []
	# for each_image in range(x_train.shape[0]):
		# temp_image = np.zeros( ( 28/filter_size , 28/filter_size  ) )
		# orig_image = np.reshape( x_train[each_image] , (28 , 28) )
                # #print ("orig image shape: " + str(np.mean(orig_image[0:4,0:4]) )  )
		# for row in np.arange(0,28,filter_size):
			# for column in np.arange(0,28,filter_size):
				# temp_image[ row/filter_size , column/filter_size ] = np.mean(orig_image[ row : (row+filter_size) , column:(column+ filter_size) ])
		# x_train_inter.append( temp_image.flatten() )
	# x_train = np.asarray(x_train_inter)

	# x_test_inter = []
	# for each_image in range(x_test.shape[0]):
		# temp_image = np.zeros( ( 28/filter_size , 28/filter_size  ) )
		# orig_image = np.reshape( x_test[each_image] , (28 , 28) )
		# for row in np.arange(0,28,filter_size):
			# for column in np.arange(0,28,filter_size):
				# temp_image[ row/filter_size , column/filter_size ] = np.mean(orig_image[ row : (row+filter_size) , column:(column+ filter_size) ])
		# x_test_inter.append( temp_image.flatten() )
	# x_test = np.asarray(x_test_inter)

	n, D = x_train.shape    # (n_sample, n_feature)



	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	print ("x_train: " + str(x_train.shape) )                                                                                  
	print ("y_train: " + str(y_train.shape) )                                                                                  
	print ("x_test: " + str(x_test.shape) )                                                                                    
	print ("y_test: " + str(y_test.shape) )
	
	### enter nn code here ; for all NN models the input data should be same . 
	for num_iteration in range(num_iterations):
		print ("num of iteration : " + str(num_iteration) )

		tr_score_all_CE = np.array([])
		te_score_all_CE = np.array([])

		tr_score_all_L2 = np.array([])
		te_score_all_L2 = np.array([])

		seed_ini = weights_ini_seeds[num_iteration]		
		inilrate_hidden_index = 0
		for index_hu in np.arange(hidden_units.shape[0]):

			print ("hidden units are: " +  str(hidden_units[index_hu]) )

			######################################################################################################################
			## model.get_weights  model.set_weights  model.layers[i].set_weights(listOfNumpyArrays)
			#load previous weights to be part of next run
			if ( (index_hu!=0) and (index_hu<=17) ):
				import h5py
				#previous_weight_file = 'fcnn_model_weights/my_model_weights_hu_' + str(hidden_units[index_hu-1]) + '.h5'
				previous_weight_file = 'fcnn_model_weights_10/my_model_weights_hu_' + str(hidden_units[index_hu-1]) + '_' + str(seed_ini) + '.h5'
				previous_weights = h5py.File(previous_weight_file, 'r')
				
				#list(previous_weights)
				ker1 = previous_weights[u'dense_1']['dense_1']['kernel:0'][:] ## this is for 1st layer FCNN
				bia1 = previous_weights[u'dense_1']['dense_1']['bias:0'][:] ## this is for 1st layer FCNN
				ker2 = previous_weights[u'dense_2']['dense_2']['kernel:0'][:] ## this is for 2nd layer FCNN
				bia2 = previous_weights[u'dense_2']['dense_2']['bias:0'][:] ## this is for 2nd layer FCNN
			
			#######################
			model = Sequential()
			
			
			
			def my_init_kernel_1(shape, shape2):
				print ( "kernel 1 shape is : " + str(shape) )
				x =  ker1 
				#y =  np.zeros(shape2)
				y = np.random.normal( mu_ini, sigma_ini, shape2 )
				return np.concatenate( ( x , y  ), axis=1  )
				
			def my_init_bias_1(shape , shape3 ):
				print ("bias 1 shape is : " + str(shape))
				return np.concatenate( ( bia1 , np.random.normal( mu_ini, sigma_ini, shape3 )  ) )
				
			def my_init_kernel_2(shape, shape4):
				print ( "kernel 2 shape is : " + str(shape) )
				x =  ker2
				#y =  np.zeros(shape4)
				y = np.random.normal( mu_ini, sigma_ini, shape4 )
				return np.concatenate( ( x , y  ), axis=0  )
				
			def my_init_bias_2(shape ):
				print ("bias 2 shape is : " + str(shape))
				return  bia2 
			
			
			L1_all = np.random.normal( mu_ini, sigma_ini, ( x_train.shape[1]+1 , hidden_units[index_hu]  )  )/np.sqrt(hidden_units[index_hu]) 
			L2_all = np.random.normal( mu_ini, sigma_ini, ( hidden_units[index_hu] +1 , num_classes  )  )/np.sqrt(hidden_units[index_hu]) 
			
			def my_init_kernel_op_1( shape ):
				y = L1_all[ np.arange(x_train.shape[1]) , : ]
				return y 
				
			def my_init_bias_op_1( shape ):
				y = L1_all[ -1 , : ]
				return  y
				
			def my_init_kernel_op_2( shape ):
				
				y = L2_all[ np.arange( hidden_units[index_hu]) , : ]
				return y 
				
			def my_init_bias_op_2( shape ):
				y = L2_all[ -1 , : ]
				return  y 
			
			
			if ( (index_hu!=0) and (index_hu<=17) ):
				diff_hidden_unit = hidden_units[index_hu] - hidden_units[index_hu-1]
				model.add(Dense( hidden_units[index_hu] , kernel_regularizer=regularizers.l2(regul), \
								kernel_initializer = lambda shape: my_init_kernel_1(shape, shape2=( shape[0] , diff_hidden_unit ) ) , \
								bias_initializer = lambda shape: my_init_bias_1( shape, shape3 = diff_hidden_unit )  ,  \
								activation='relu', input_shape=(x_train.shape[1],)  ) )
				
				model.add(Dense(  num_classes , kernel_initializer = lambda shape: my_init_kernel_2(shape, shape4=( diff_hidden_unit , shape[1]  ) ) , \
								bias_initializer = lambda shape: my_init_bias_2( shape )   ) )  
								
			elif( index_hu==0 ):
				model.add(Dense( hidden_units[index_hu] , kernel_regularizer=regularizers.l2(regul), kernel_initializer = glorot_uniform(seed=seed_ini), \
								activation='relu', input_shape=(x_train.shape[1],)  ) )
				
				model.add(Dense(  num_classes , kernel_initializer = glorot_uniform(seed=seed_ini)  ) )

			# else:
				# model.add(Dense( hidden_units[index_hu] , kernel_regularizer=regularizers.l2(regul), kernel_initializer = glorot_uniform(seed=seed_ini), \
								# activation='relu', input_shape=(x_train.shape[1],)  ) )
				
				# model.add(Dense(  num_classes , kernel_initializer = glorot_uniform(seed=seed_ini)  ) )
				
			else:
				model.add(Dense( hidden_units[index_hu] , kernel_regularizer=regularizers.l2(regul), kernel_initializer = lambda shape: my_init_kernel_op_1(shape ) , \
								bias_initializer = lambda shape: my_init_bias_op_1( shape )  ,
								activation='relu', input_shape=(x_train.shape[1],)  ) )
				
				model.add(Dense(  num_classes , kernel_initializer = lambda shape: my_init_kernel_op_2(shape ) , \
								bias_initializer = lambda shape: my_init_bias_op_2( shape )  ) )

			model.summary()

			######################################################################################################################
			##terminate_criteria = EarlyStopping(monitor='val_acc', min_delta=0.0002, patience=10)

			### callback function for train/valid accuracy ##############################################################
		
			class EarlyStoppingByLossVal(Callback):
				def __init__(self, monitor='val_acc', value=1.0, verbose=1):
					super(Callback, self).__init__()
					self.monitor = monitor
					self.value = value
					self.verbose = verbose

				def on_epoch_end(self, epoch, logs={}):
					current = logs.get(self.monitor)
					if current is None:
						warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

					if current >= self.value:
						if self.verbose > 0:
							print("Epoch %05d: early stopping THR" % epoch)
						self.model.stop_training = True

			class NBatchLogger(Callback):
			    ##A Logger that log average performance per `display` steps.
			    def __init__(self, display):
			        self.step = 0
			        self.display = display
			        self.metric_cache = {}

			    def on_epoch_end(self, epoch, logs={}):
			        self.step += 1
			        for k in self.params['metrics']:
			            if k in logs:
			                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
			        if self.step % self.display == 0:
			            metrics_log = ''
			            for (k, v) in self.metric_cache.items():
			                val = v / self.display
			                if abs(val) > 1e-3:
			                    metrics_log += ' - %s: %.4f' % (k, val)
			                else:
			                    metrics_log += ' - %s: %.4e' % (k, val)
			            print('step: {}/{} ... {}'.format(self.step,
			                                          self.params['steps'],
			                                          metrics_log))
			            self.metric_cache.clear()

			def step_decay(epoch , initial_lrate , index_hu ):
			   
			    if ( (index_hu>=0) and (index_hu<=17) ):
				    drop = 0.9
			    else:
				    drop = 1.0
				
			    epochs_drop = 500.0
			    lrate = initial_lrate * math.pow(drop,  
					   math.floor((1+epoch)/epochs_drop))
			    if lrate<=initial_lrate*0.25:
				    lrate = initial_lrate*0.25
			    return lrate

			initial_lrate = initial_lrates[inilrate_hidden_index]
			step_decay_out = lambda epoch : step_decay(epoch , initial_lrate , index_hu )
			lrate_fun = LearningRateScheduler(step_decay_out)

			EarlyStop = EarlyStoppingByLossVal(monitor='val_acc', value=1.0, verbose=1)
			Display_after = NBatchLogger(display = 1000)
			
			if ( (index_hu>=0) and (index_hu<=17) ):
				callback_list = [ EarlyStop , Display_after , lrate_fun ]
                        else:
				callback_list = [ Display_after , lrate_fun ]
                        
			## end of callbacks #################################################################

			model.compile(loss= 'mean_squared_error' ,
			              optimizer= SGD( momentum=0.95  ) ,
			              metrics=['accuracy'])


			history = model.fit(x_train, y_train,
			                    batch_size=batch_size,
			                    epochs= all_epochs ,  verbose = 0 ,
			                    validation_data=(x_train, y_train), callbacks= callback_list  )
			

			model.save_weights( 'fcnn_model_weights_10/my_model_weights_hu_' + str(hidden_units[index_hu]) + '_' + str(seed_ini) + '.h5' )
            #for layer in model.layers:
    		#	weights = layer.get_weights() # list of numpy arrays
    		#	print ("weight length is : "+str(len(weights))+" weight : " + str(weights[0].shape)+" bias : " + str(weights[1].shape) )

			### end of nn code 

			# score_test = model.evaluate(x_test, y_test, verbose=0)
			# score_train = model.evaluate(x_train, y_train, verbose=0)
			# mse_test = score_test[0]
			# mse_train =  score_train[0]
			# ce_test = 100.0 - score_test[1]*100.0
			# ce_train = 100.0 - score_train[1]*100.0
			
			y_test_estimated = model.predict(x_test, verbose=0)
			y_train_estimated = model.predict(x_train, verbose=0)

			mse_test = np.sum(np.square(y_test_estimated - y_test))/y_test.shape[0]
			mse_train =  np.sum(np.square(y_train_estimated - y_train))/y_train.shape[0]
			ce_test = 100.0 - np.mean(np.equal(np.argmax(y_test, axis=-1), np.argmax(y_test_estimated, axis=-1)))*100
			ce_train = 100.0 - np.mean(np.equal(np.argmax(y_train, axis=-1), np.argmax(y_train_estimated, axis=-1)))*100
			
			
			print  ("mse_test : " + str(mse_test) )
			print  ("mse_train : " + str(mse_train) )
			print  ("ce_test : " + str(ce_test) )
			print  ("ce_train : " + str(ce_train) )

			tr_score_all_CE = np.append(tr_score_all_CE, ce_train )
			te_score_all_CE = np.append(te_score_all_CE, ce_test )
			tr_score_all_L2 = np.append(tr_score_all_L2, mse_train )
			te_score_all_L2 = np.append(te_score_all_L2, mse_test )

			del model
	                K.clear_session()
	                reload (K)
			gc.collect()

			inilrate_hidden_index = inilrate_hidden_index + 1

	## start modifying here

		import pickle
		def savepkl(fpath, data):
			with open(fpath, 'w') as f: 
				pickle.dump(data, f)



		dict_ce = { 'hidden_units': hidden_units, 'train': tr_score_all_CE, 'test': te_score_all_CE }
		dict_l2 = { 'train': tr_score_all_L2, 'test': te_score_all_L2 }
		dict_comb = { 'ce': dict_ce, 'l2_loss': dict_l2  }

		pickle_name = 'FCNN_1hiddenlayer' + '_overfit'   + '_mnist_10class-' + str(num_training_data) \
						 + '_sgdmoment-on_reuse-weight_seed-'+ str(seed_ini) + '_it-' + str(num_iteration) + '_final' + '.p'
		#pickle_name = 'FCNN_1hiddenlayer' + '_overfit'   + '_mnist_2class-' + str(num_training_data) \
		#				 + '_sgdmoment-on_regul-0.01' + 'it-' + str(num_iteration) + '.p'
		dict_final = { 'FCNN_1hiddenlayer'  : dict_comb }
		#savepkl( '../pickle/mnist/neural_network/FCnn/reuse_weight/10class/'+ pickle_name , dict_final ) ## i am here
		savepkl( '../pickle/mnist/neural_network/FCnn/reuse_weight_normalizednorm/'+ pickle_name , dict_final ) ## i am here

print (" successful !!! ")






