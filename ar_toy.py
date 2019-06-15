from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
##tf.enable_eager_execution(config = config)

import numpy as np
import pandas as pd
import os
import time
import argparse
import matplotlib.pyplot as plt
import pdb
import h5py
#from tf.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import json
from tensorflow.keras.regularizers import l1, l2


parser = argparse.ArgumentParser() 

parser.add_argument('--train', action = 'store_true', help = "train the model")
parser.add_argument('--test',  action = 'store_true', help = "test the model")
parser.add_argument('--save_path', default ='training_checkpoints/',type = str, help = "path of save model")
parser.add_argument('--epoch', default = 2000 ,type = int, help = 'number of epcohs to train')
parser.add_argument('--rnn', default = 10 ,type = int, help = 'number rnn units')
parser.add_argument('--len', default = 193 ,type = int, help = 'length of sequence')
parser.add_argument('--drop', default = 0.1 ,type = float, help = 'dropout rate')
parser.add_argument('--dense',type = int, help = 'dim of dense layer')
#parser.add_argument('--dense', default = 100 ,type = int, help = 'dim of dense layer')


parser.add_argument('--model_name', default ='ar_toy',type = str, help = 'name of model')
parser.add_argument('--test_steps', default =5 ,type = int, help = 'number of future steps to predict')
parser.add_argument('--test_sample', default = '0:6' , type = str, help = 'a sequence of T test sample specify by the start_index:end_index')
#parser.add_argument('--test_file', type = int, help = 'T to test the entire file')
parser.add_argument('--test_file', action = 'store_true',  help = 'test the entire file')

parser.add_argument('--plot_sample',  action = 'store_true', help = "plot test sample")

args = parser.parse_args()

if args.test:
	tf.enable_eager_execution(config = config)

def sin_wave(amp = 1, w = 1, y = 0):
	
	x = np.arange(0, 100*np.pi, 0.31)

	return amp* np.sin( w* x) + y

def norm(data):

	max_data = np.max(data, axis = 0)
	min_data = np.min(data, axis = 0)
	range_data = max_data - min_data

	normed_data = (data - min_data)/ (range_data + 1e-09)

	return normed_data, min_data, range_data

hf = h5py.File('train_processed.h5', 'r')
traffic = hf.get('data')[:]
#a =  traffic[:, 1299].reshape(-1, 1)
#b =  traffic[:, 1200].reshape(-1, 1)
#c =  traffic[:, 1309].reshape(-1, 1)

time_list = np.arange(0, 1, step = 1/96.).tolist()
last = np.array(time_list*61).reshape(-1, 1)
#pdb.set_trace()
#toy = np.concatenate([a,b,c], axis = -1) 
#toy = np.concatenate([a, b,c,last], axis = -1) 
#toy = traffic
#toy = traffic[:, :100]
#toy = traffic[:, :150]
#toy = traffic[:, 200:300]
#toy = traffic[:, 400:500]
#toy = traffic[:, 300:500]
#toy = traffic[:, 500:600]
#toy = traffic[:, 600:700]
#toy = traffic[:, 700:800]
#toy = traffic[:, 800:900]
#toy = traffic[:, 900:1000]
#toy = traffic[:, 1000:1100]
#toy = traffic[:, 1100:1200]
toy = traffic[:, 1200:]
#toy = traffic[:, 443: 443*2]
#toy = traffic[:, :443]
#toy = np.concatenate([toy, last], axis = -1)
#sin wave
#a= sin_wave(amp = 0.5, w=1, y = 0.5 )
#b= sin_wave(amp = 0.25, w=1, y = 0.5 )
#c= sin_wave(amp = 0.4, w=2, y = 0.5 )
#wave_1 = a.reshape(-1, 1)
#toy = wave_1
#wave_2 = b.reshape(-1, 1)
#wave_3 = c.reshape(-1, 1)
#toy = np.concatenate([wave_1, wave_2, wave_3], axis = -1)

#toy_train_raw = toy[:int (len(toy) * 0.8)].round(3).astype('float32')
#toy_val_raw = toy[int (len(toy) * 0.8):].round(3).astype('float32')


#toy_train, toy_train_min, toy_train_range = norm(toy_train_raw)
#toy_val, toy_val_min, toy_val_range = norm(toy_val_raw)

#toy_val = (toy_val_raw - toy_train_min )/(toy_train_range + 1e-09)

#rolling_size = 10 
#toy_smooth = pd.DataFrame(toy).rolling(window = rolling_size).mean()
#toy_smooth = np.array(toy_smooth.iloc[rolling_size-1:])
#toy = np.concatenate([toy[:rolling_size-1], toy_smooth], axis = 0)

toy_train = toy[:int (len(toy) * 0.8)].round(3).astype('float32')
toy_val = toy[int (len(toy) * 0.8):].round(3).astype('float32')
#toy_val_un = toy[int (len(toy) * 0.8):].round(3).astype('float32')

#toy_train, toy_train_min, toy_train_range = norm(toy_train)
#toy_val, toy_val_min, toy_val_range = norm(toy_val)

#toy_val = toy_val_raw

#toy = np.arange(0.1, 0.99, 0.001)
#toy = np.array([toy]).T # shape (890, 1)
#toy_train = toy[:800].round(3).astype('float32')
#toy_val = toy[800:].round(3).astype('float32')



# The maximum length sentence we want for a single input in characters
seq_length_toy = args.len
examples_per_epoch_toy = len(toy_train)//seq_length_toy
examples_per_epoch_toy_val = len(toy_val)//seq_length_toy

seq_length_toy_val = args.len 

# Create training examples / targets
char_dataset_toy = tf.data.Dataset.from_tensor_slices(toy_train)
char_dataset_toy = char_dataset_toy.window(seq_length_toy, shift=1, drop_remainder=True)
sequences_toy = char_dataset_toy.flat_map(lambda window: window.batch(seq_length_toy))
examples_per_epoch_toy = len(toy_train) - seq_length_toy + 1

char_dataset_toy_val = tf.data.Dataset.from_tensor_slices(toy_val)
char_dataset_toy_val = char_dataset_toy_val.window(seq_length_toy_val, shift=1, drop_remainder=True)
sequences_toy_val = char_dataset_toy_val.flat_map(lambda window: window.batch(seq_length_toy_val))
examples_per_epoch_toy_val = len(toy_val) - seq_length_toy_val + 1


#sequences_toy = char_dataset_toy.batch(seq_length_toy+1, drop_remainder=True)
#sequences_toy_val = char_dataset_toy_val.batch(seq_length_toy_val+1, drop_remainder=True)

def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text, target_text


dataset_toy = sequences_toy.map(split_input_target, num_parallel_calls = 4)

dataset_toy_val = sequences_toy_val.map(split_input_target, num_parallel_calls = 4)

BATCH_SIZE_toy = 64 
steps_per_epoch_toy = examples_per_epoch_toy//BATCH_SIZE_toy
BATCH_SIZE_toy_val = 64 
steps_per_epoch_toy_val = examples_per_epoch_toy_val//BATCH_SIZE_toy_val




# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE_toy = 6000

dataset_toy = dataset_toy.shuffle(BUFFER_SIZE_toy).batch(BATCH_SIZE_toy, drop_remainder=True)
dataset_toy = dataset_toy.repeat()
dataset_toy = dataset_toy.prefetch(1)


dataset_toy_val = dataset_toy_val.shuffle(BUFFER_SIZE_toy).batch(BATCH_SIZE_toy_val, drop_remainder = True)
dataset_toy_val = dataset_toy_val.repeat()
dataset_toy_val = dataset_toy_val.prefetch(1)

# Length of the vocabulary in chars
#vocab_size_toy = 1
feature_size = toy_train.shape[-1]

# The embedding dimension
embedding_dim_toy = 10

# Number of RNN units
rnn_units_toy = args.rnn 

rnn1 = tf.keras.layers.CuDNNGRU
#rnn2 = tf.keras.layers.CuDNNLSTM
#rnn = tf.keras.layers.RNN

def build_model_toy(feature_size, embedding_dim, rnn_units, batch_size, state= False):
	model = tf.keras.Sequential([
	#	 tf.keras.layers.Embedding(vocab_size, embedding_dim,
	#							   batch_input_shape=[batch_size, None]),
	
	#tf.keras.layers.Dense(args.dense, activation = None),
	#tf.keras.layers.Dropout(args.drop),

	rnn1(rnn_units,
	return_sequences=True,
	recurrent_initializer='glorot_uniform',
	#batch_input_shape = [batch_size, None, feature_size],
	stateful= state ),
	
	tf.keras.layers.Dropout(args.drop),

	tf.keras.layers.Dense(250, activation = None),
	#tf.keras.layers.Dropout(args.drop),

	#tf.keras.layers.Dense(args.dense, activation = None),
	#tf.keras.layers.Dropout(args.drop),

	tf.keras.layers.Dense(feature_size, activation = 'sigmoid')
	])

	#inputs = tf.keras.layers.CuDNNGRU(
	#rnn_units,
	#return_sequences=True,
	#recurrent_initializer='glorot_uniform',
	#stateful= state, batch_input_shape = [batch_size, None, feature_size])

	#outputs = tf.keras.layers.Dense(feature_size, activation = 'sigmoid')(inputs)

	#model = tf.keras.Model(inputs, outputs)

	return model

model_toy = build_model_toy(
feature_size,
embedding_dim=embedding_dim_toy,
rnn_units=rnn_units_toy,
batch_size=BATCH_SIZE_toy)

#pdb.set_trace()

def loss_toy(y_true, y_pred):
	#return tf.math.sqrt(tf.keras.losses.mse(y_true, y_pred))
	return tf.math.sqrt(tf.losses.mean_squared_error(y_true, y_pred))

model_toy.compile(
optimizer = tf.train.AdamOptimizer(),
loss = loss_toy)

def plot_test(input_test, target, prediction):

	#labels = [ "target", "prediction"]
	#
	#plot_data = dict()
	#plot_data["train_loss"] = train_loss
	#plot_data["val_loss"] = val_loss
	#pdb.set_trace()
	
	features = input_test.shape[-1]

	target = np.concatenate( [input_test, target], axis = 1 )
	steps = range(target.shape[1])
	start_pred = input_test.shape[1]

	#prediction = prediction.ravel()
	
	fig, ax = plt.subplots(figsize = (20,8))
	
	#for feature_idx in range(features):
	features = [1329]
	for feature_idx in features:

		plt.plot(steps[:], target[:,:,feature_idx].ravel(), label = 'target_{}'.format(feature_idx), color = '#1f77b4')
		plt.plot(steps[start_pred:], prediction[:,:,feature_idx].ravel(), label = 'prediciton_{}'.format(feature_idx), linestyle='dashed', color = '#ff7f0e')

	ax.legend()
	#ax.set_ylim(0.0, 5)
	plt.title(args.model_name)
	#plt.xlabel('no. of epoch')
	fig.savefig('plot_test_sample/' + args.model_name + '.png')

if args.train:

	# Directory where the checkpoints will be saved
	#checkpoint_dir = './training_checkpoints'
	# Name of the checkpoint files
	#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

	tf.keras.backend.set_session(tf.Session(config=config))

	checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
	filepath= args.save_path + args.model_name,
	save_weights_only=True,
	monitor = 'val_loss',
	save_best_only = True 
	)

	tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(args.model_name))

	
	history_toy = model_toy.fit(dataset_toy, epochs= args.epoch, steps_per_epoch=steps_per_epoch_toy, validation_data = dataset_toy_val, validation_steps = steps_per_epoch_toy_val,  callbacks=[checkpoint_callback, tensorboard])
	#history_toy = model_toy.fit(dataset_toy, epochs=EPOCHS, steps_per_epoch=steps_per_epoch_toy, callbacks=[checkpoint_callback, tensorboard])

	#pdb.set_trace()
	min_train_loss = min(history_toy.history['loss'])
	min_train_val_loss = min(history_toy.history['val_loss'])

	print ('\n train: {}, validation: {}'.format(min_train_loss, min_train_val_loss)) 
	#print ('\n train: {}'.format(min_train_loss)) 
	
	fp = json.dumps({'min_train_loss': min_train_loss, 'min_train_val_loss': min_train_val_loss})
	open('json_out/' + args.model_name + '.json','w').write(fp)


if args.test:

	start_time = time.time()

	model_toy = build_model_toy(feature_size, embedding_dim_toy, rnn_units_toy, batch_size=1, state = True)
	model_toy.load_weights(args.save_path + args.model_name)
	#pdb.set_trace()
	model_toy.build(tf.TensorShape([1, None, feature_size]))
	
	#test_sample = np.array([[
	#[0.220], [0.221], [0.222], [0.223], [0.224],[0.225],[0.226],[0.227] ]]).astype('float32')
	#pdb.set_trace()

	#test_sample = np.array([[
	#[0.121], [0.122], [0.123], [0.124], [0.125],[0.126],[0.127],[0.128],[0.129] ]]).astype('float32')

	#test_file = toy_train
	test_file = toy_val
	#test_file = toy_val_un
	#test_file_raw = toy_val_raw

	pred_ahead = args.test_steps

	if args.test_file:

		#input_len = args.test_file
		input_lens = [24, 48, 96, 164, 192 ]
		#input_lens = [96, 192]
		num_samples = []

		for input_len in input_lens:
			num_samples.append(len(test_file) // ( input_len + pred_ahead))

		#start_idx = 0
		#end_idx = start_idx + input_len

	elif args.test_sample:
		start, end = args.test_sample.split(':')
		start_sample_idx = int(start)
		end_sample_idx = int(end) + 1

		input_lens = [ end_sample_idx - start_sample_idx ] 
		num_samples = [1]	

	
	#input_test_tensor = tf.data.Dataset.from_tensor_slices(toy_va)

	#input_test = np.array([[
	#[0.921], [0.922], [0.923], [0.924], [0.925],[0.926],[0.927],[0.928] ]]).astype('float32')


	
	def prediction(input_lens, num_samples): 

		all_predictions = []
		all_targets = []

		for idx in range(len(input_lens)):
			
			input_len = input_lens[idx]
			num_sample = num_samples[idx]
			start_idx = 0 if args.test_file else start_sample_idx
			end_idx = start_idx + input_len 

			for sample_idx in range(num_sample):
				model_toy.reset_states()

				input_test = test_file[ start_idx : end_idx ].reshape(1, -1, feature_size).astype('float32') 
				test_target = test_file[end_idx: end_idx + pred_ahead ].reshape(1, -1, feature_size).astype('float32') 
				#print ('\n input_test: {}'.format(input_test))
				#print ('\n test_target: {}'.format(test_target))
				#pdb.set_trace()
				start_idx = end_idx + pred_ahead
				end_idx = start_idx + input_len

				input_seq = input_test
				#input_seq = input_seq[:,:,:-1]
				all_targets.append(test_target)

				for step in range(pred_ahead):	
					prediction = model_toy(input_test).numpy()[:, -1:, :] # pick last prediction in sequence
					input_test = prediction

					all_predictions.append(prediction[0])
		#pdb.set_trace()		
		all_predictions = np.array(all_predictions).reshape(1,-1, feature_size)

		#all_predictions = np.array(all_predictions).reshape(-1, feature_size)
		#all_predictions = all_predictions * (toy_val_range + 1e-09) + toy_val_min 
		#all_predictions = all_predictions.reshape(1, -1, feature_size)

		all_targets = np.array(all_targets).reshape(1, -1, feature_size)
		#print ('\n prediction: {}'.format( all_predictions))
		#print ('all_targets: ', all_targets)
		#pdb.set_trace()
		#all_predictions = all_predictions[:, :, :-1]
		#all_targets = all_targets[:, :, :-1]
		test_loss = loss_toy(all_targets, all_predictions)

		return test_loss, input_seq, all_targets, all_predictions
	
	test_loss, input_seq, all_targets, all_predictions = prediction(input_lens, num_samples)

	print ('\n test_loss: {}'.format(test_loss)) 
	
	print ('inference time :{}'.format(time.time() - start_time))

	if args.plot_sample:
		plot_test(input_seq, all_targets, all_predictions)
		print ('\n fig saved')
