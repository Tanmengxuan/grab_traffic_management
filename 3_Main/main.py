from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

import numpy as np
import pandas as pd
import os
import time
import argparse
import pdb
import h5py
import time
import json
from utils import plot_test

parser = argparse.ArgumentParser() 

#Commands
parser.add_argument('--train', action = 'store_true', help = "train the model")
parser.add_argument('--test',  action = 'store_true', help = "test the model")
parser.add_argument('--test_file', action = 'store_true',  help = 'test the entire file')
parser.add_argument('--plot_sample',  action = 'store_true', help = "plot predicted sample")

#Training and test parameters
parser.add_argument('--epoch', default = 50, type = int, help = 'number of epcohs to train')
parser.add_argument('--batch_size', default = 64, type = int, help = 'mini batch size')
parser.add_argument('--rnn', default = 150, type = int, help = 'number rnn units')
parser.add_argument('--dense',default = 150, type = int, help = 'dim of dense layer')
parser.add_argument('--len', default = 193 ,type = int, help = 'length of sequence')
parser.add_argument('--drop', default = 0.1 ,type = float, help = 'dropout rate')

#Input and output
parser.add_argument('--model_name', default ='ar_toy',type = str, help = 'name of model to save')
parser.add_argument('--save_path', default ='checkpoints/',type = str, help = "path of save model")
parser.add_argument('--file_path' , default = '../processed_data/train_processed_8c.h5', type = str, help = 'file path')
parser.add_argument('--json_path' , default = '../processed_data/geohash_8c.json', type = str, help = 'path of feature idx')

#Task parameters
parser.add_argument('--test_steps', default =5 ,type = int, help = 'number of future steps to predict')
parser.add_argument('--test_sample', default = '0:6' , type = str, help = 'a sequence of T test sample specify by the start_index:end_index')

args = parser.parse_args()

if args.test:
	tf.enable_eager_execution(config = config)


class Data_input(object):
	
	'''
	A class used to create objects that store information on the indexes of features in different location clusters 

	'''
	def __init__(self, start_feature_idx, end_feature_idx ):
		
		self.start_feature_idx = start_feature_idx
		self.end_feature_idx = end_feature_idx
		
	def start_idx(self):
		
		return self.start_feature_idx
	
	def end_idx(self):
		
		return self.end_feature_idx
	
	def num_features(self):
		
		return self.end_feature_idx  - self.start_feature_idx 


class Pipeline(object):

	'''
	A class used to set up the data pipelines for different datasets during model training
	'''

	def __init__(self, data, seq_len, batch_size, data_stru):

		self.data = data
		self.seq_len = seq_len
		self.batch_size = batch_size
		self.data_stru = data_stru
		self.buffer_size = 8000

	def split_input_target(self, chunk):
		
		input_data = []
		
		for data in self.data_stru:
			
			start_feature = data.start_idx()
			end_feature = data.end_idx()
			
			input_data.append( chunk[:, start_feature: end_feature ][:-1] )
		
		target_text = chunk[1:]
		
		return tuple(input_data), target_text

	def data_pipe(self):

		examples_per_epoch = len(self.data)//self.seq_len
		print ('examples_per_epoch', examples_per_epoch)
		# Create training examples / targets
		tf_dataset = tf.data.Dataset.from_tensor_slices(self.data)
		dataset_slide = tf_dataset.window(self.seq_len, shift=1, drop_remainder=True)
		sequences_slide = dataset_slide.flat_map(lambda window: window.batch(self.seq_len))
		examples_per_epoch = len(self.data) - self.seq_len + 1

		dataset = sequences_slide.map( self.split_input_target, num_parallel_calls = 4)

		steps_per_epoch = examples_per_epoch//self.batch_size

		dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)
		dataset = dataset.repeat()
		dataset = dataset.prefetch(1)

		return dataset, steps_per_epoch

class Predict(object):

	def __init__(self, model_test, test_data, input_lens, num_samples, data_stru):
		
		self.model_test = model_test
		self.test_data = test_data
		self.input_lens = input_lens
		self.num_samples = num_samples
		self.data_stru = data_stru

	def get_input(self, test_data, start_idx, end_idx):
		
		input_list = []
			
		for data in self.data_stru:
			start_feature = data.start_idx()
			end_feature = data.end_idx()
			feature_size = data.num_features()

			data_input = test_data[ start_idx : end_idx, start_feature:end_feature ].reshape(1, -1, feature_size).astype('float32')

			input_list.append(data_input)
		
		return input_list

	def split(self, pred):
		
		predicted_list = []
		
		for data in self.data_stru:
			start_feature = data.start_idx()
			end_feature = data.end_idx()
			predicted_list.append(   pred[:,:, start_feature:end_feature]	)
		
		return predicted_list

	def prediction(self):

		all_predictions = []
		all_targets = []

		for idx in range(len(self.input_lens)):

			input_len = self.input_lens[idx]
			num_sample = self.num_samples[idx]
			start_idx = 0 if args.test_file else int(args.test_sample.split(':')[0])
			end_idx = start_idx + input_len 

			for sample_idx in range(num_sample):
				self.model_test.reset_states()
				
				input_test = self.get_input(self.test_data, start_idx, end_idx)
				test_target = self.test_data[end_idx: end_idx + args.test_steps ].reshape(1, -1, self.test_data.shape[-1]).astype('float32') 

				start_idx = end_idx + args.test_steps
				end_idx = start_idx + input_len

				input_seq = input_test
				all_targets.append(test_target)

				for step in range(args.test_steps):
					# pick last prediction in sequence
					prediction = self.model_test(input_test).numpy()[:, -1:, :] 
					all_predictions.append(prediction[0])
					
					input_test = self.split(prediction)

		all_predictions = np.array(all_predictions).reshape(1,-1, self.test_data.shape[-1])
		all_targets = np.array(all_targets).reshape(1, -1, self.test_data.shape[-1])

		test_loss = rmse(all_targets, all_predictions)

		return test_loss, input_seq, all_targets, all_predictions

class RNN(object):
	
	def __init__(self, units, dense_size, drop_rate, feature_size, batch_size, state):
		
		self.units = units
		self.dense = dense_size
		self.drop = drop_rate
		self.feature_size = feature_size
		self.batch_size = batch_size
		self.state = state
		
	def rnn_model(self):

		model_input = tf.keras.layers.Input(batch_shape = (self.batch_size, None, self.feature_size))

		dense0 = tf.keras.layers.Dense(self.dense, activation = None)(model_input)
		dense0 = tf.keras.layers.Dropout(self.drop)(dense0)
		
		rnn =  tf.keras.layers.CuDNNGRU( self.units,
			return_sequences=True,
			recurrent_initializer='glorot_uniform',
			stateful= self.state)(dense0)

		rnn = tf.keras.layers.Dropout(self.drop)(rnn)

		dense1 = tf.keras.layers.Dense(self.dense, activation = None)(rnn)
		
		model_output =  tf.keras.layers.Dense(self.feature_size, activation = 'sigmoid')(dense1)

		return model_input, model_output


def rmse(y_true, y_pred):
	
	'''
	root mean squared loss function used to train model
	'''

	return tf.math.sqrt(tf.losses.mean_squared_error(y_true, y_pred))


def combine_rnn(rnn_units, dense_size, drop_rate, batch_size, state = False):
	
	'''
	a function used to initialize the multiple rnns and combines them at the end
	'''
	input_list = []
	output_list = []
	
	for data in data_stru:
		
		feature_size = data.num_features()
		model_input, model_output = RNN(rnn_units, dense_size, drop_rate, feature_size, batch_size, state).rnn_model()

		input_list.append(model_input)
		output_list.append(model_output)
		
	final_out = tf.keras.layers.concatenate(output_list, axis = -1)
	model = tf.keras.models.Model(inputs = input_list , outputs = final_out)
	
	return model


def main(train, val, data_stru):

	if args.train:
		train_data, steps_per_epoch_train = Pipeline(train, args.len, args.batch_size, data_stru).data_pipe()
		val_data, steps_per_epoch_val = Pipeline(val, args.len, args.batch_size, data_stru).data_pipe() 

		model = combine_rnn(args.rnn, args.dense, args.drop, args.batch_size)

		model.compile(
		optimizer = tf.train.AdamOptimizer(),
		loss = rmse)
		
		#tf.keras.utils.plot_model(model)
		print (model.summary())

		tf.keras.backend.set_session(tf.Session(config=config))

		checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
		filepath= args.save_path + args.model_name,
		save_weights_only=True,
		monitor = 'val_loss',
		save_best_only = True 
		)

		tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(args.model_name))

		history = model.fit(train_data, epochs= args.epoch, steps_per_epoch=steps_per_epoch_train,
		validation_data = val_data, validation_steps = steps_per_epoch_val, 
		callbacks=[checkpoint_callback, tensorboard])

		min_train_loss = min(history.history['loss'])
		min_train_val_loss = min(history.history['val_loss'])

		print ('\n train: {}, validation: {}'.format(min_train_loss, min_train_val_loss)) 

	if args.test:

		start_time = time.time()
		model_test = combine_rnn( args.rnn, args.dense, args.drop, batch_size = 1, state = True)
		model_test.load_weights(args.save_path + args.model_name)
		
		test_data = h5py.File('../processed_data/test_processed.h5', 'r').get('data')[:]
		test_data = test_data.astype('float32')

		if args.test_file:
			input_lens = [24, 48, 96, 164, 192]
			num_samples = []

			for input_len in input_lens:
				num_samples.append(len(test_data) // ( input_len + args.test_steps))

		elif args.test_sample:
			start, end = args.test_sample.split(':')
			start_sample_idx = int(start)
			end_sample_idx = int(end) + 1

			input_lens = [ end_sample_idx - start_sample_idx ] 
			num_samples = [1]	


		test_loss, input_seq, all_targets, all_predictions = Predict(model_test, test_data, input_lens, num_samples, data_stru).prediction()
		print ('\n test_loss: {}'.format(test_loss)) 
		print ('inference time :{}'.format(time.time() - start_time))

		if args.plot_sample:
			plot_test(input_seq, all_targets, all_predictions, args.model_name)


if __name__ == '__main__':

	hf = h5py.File( args.file_path, 'r')
	traffic = hf.get('data')[:]
	train = traffic[:int (len(traffic) * 0.8)].round(3).astype('float32')
	val = traffic[int (len(traffic) * 0.8):].round(3).astype('float32')


	#Load the list of features indexes that contain the start and end index of features from each location cluster 
	#these indexes corresponse to the column indexes of the processed data arrays
	features_index = json.loads((open(args.json_path,'r')).read())['feature_idx']
	data_stru = []
	for feature in features_index:
		data_stru.append(Data_input(feature[0], feature[1]))

	main(train, val, data_stru)

