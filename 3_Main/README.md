# 3 - Running the code 

## Software packages requirements 

The scripts are written using Python 3.5.2 and tested in the Ubuntu 16.04 environment.
Please make sure Tensorflow 1.13 with gpu support is being installed.

Alternative, you can use the docker image that I have created for 
this project, it contains all the necessary supporting packages.
You only need to install CUDA version 10. 

To use the docker image, run
```
$ docker pull tmxxuan/tf1.13-gpu-py3:traffic
```

Next, run the image using nvidia docker:
```
$ nvidia-docker run -it --rm -p 8888:8888 tmxxuan/tf1.13-gpu-py3:traffic bash 
```

If you do not have docker and nvidia-docker installed, you can refer to this [guide](https://www.tensorflow.org/install/docker) for instructions

## Steps to run the code

There are only two steps involved in evaluating the trained model on the test set.

## Step 1

### Preprocess the raw test dataset

Run
```
$ python process.py --input_raw <file path of test set>
```
This command will preprocess the raw test set and will be saved as
`test_processed.h5` in the `../processed_data/` folder.


## Step 2


## Evaluating the trained model 

### Test on the entire preprocessed test dataset

After `test_processed.h5` has been saved, we can evaluate the model (named as 'best_model') that
has been trained and saved in the `checkpoints/` folder.

Run
```
$ python main.py --test --model_name best_model --test_file
```

The `--test_file` flag will test the entire preprocessed test dataset by taking in inputs of various lengths T
and outputting predictions that are 5 time intervals ahead. 
The predicted values will be stored and evaluated with the true values using the the root mean squared error (RMSE) function,
printing the RMSE for all test samples at the end.

The length T of input sequences given to the model can be easily modified by changing the values in the `input_lens` variable in the `main.py` script. 

```python3

def main(train, val, data_stru):

	'''

	'''
	if args.test_file:
		input_lens = [24, 48, 96, 164, 192]
```

Each integer value in `input_lens` represents a value of T. The length of `input_lens` will determine how many times the model will iterate through the entire preprocessed test data.

Doing so will create sufficient test samples of different lengths T to evaluate the trained model.

The python psuedo code on what `--test_file` is doing:

```python3

# the number of future steps to predict
test_steps = 5

# num_samples will contain the numbers of test examples the preprocess test_data can create given each T 
num_samples = []
for T in input_lens:
	num_samples.append( len(test_data) // ( input_len + test_steps))	

# start of evaluation
For T in input_lens:
	start_index = 0
	end_index = start_index + end_index

	For samples in range(num_samples[idx]): 
		model.reset_states() # so that the model will treat each test example as the first time it is seeing it 
		input_sequence = test_data[start_index: end_index]
		
		start_index = end_index + test_steps
		end_idx = start_index + T

		For step in range(test_steps):
			predcition = model.predict(input_sequence)			
			input_sequence = predcition

		End For when model has output T+5 predcitions

	End For when all possible test examples are evaluated

End For when all test cases of T have been evaluated

```

The actual code is located in `def prediction(self)` under the `Predict` class in `main.py`

Below is a simplied explanation of the psuedo code:

```

	For instance, in the case of the first T = 24 in the list, 
	the model will do the following 3 steps: 

	1. Take in the preprocessed samples from row index 0 to 23 and predict the values in row index 24 to 28.
	2. The model will reset its cell states. 
	3. The model takes in the next input sequence of length T = 24 (row index 29 to 53) and predict values from index 54 to 58.

	These 3 steps repeats untils it has iterated through the entire preprocessed test set.

The above process will then be applied the rest of the values (T) in the `input_lens` list variable.

```
#### Test on a specific sequence 

To test on a specific sequence in the preprocessed file and outputting a prediction sequence of your choice,
you can use the `--test_sample` and `--test_steps` flags.

For instance, running the command
```
$ python main.py --test --model_name best_model --test_sample 0:300 --test_steps 100
```
will give the model preprocessed samples from row index 0 to 300 as an input sequence and the it will predict the 
demand values from all locations that are 100 timesteps ahead. 

#### Visualize the predicted values 

`--test_sample` is usually used with `--plot_sample` to visualize the predicted samples against the true samples.
Hence running
```
$ python main.py --test --model_name best_model --test_sample 0:300 --test_steps 100 --plot_sample
```
will plot the predicted demands at locations 'qp09eq' and 'qp09ep' that corresponses to column index 200 and 199 respectively.
You can change the location by changing the integers in the `features_idx` variable in the `utils.py` script. 


```python3

def plot_test(input_test, target, prediction, model_name):

	'''

	'''

	features = [ 200, 199]

```

## Training the model

If you want to re-train the model, run

```
$ python main.py --train --model_name <name of your model> 
```

This trains a model with default parameters:

- Training epochs: `--epoch 50`
- Mini batch size: `--batch_size 64`
- GRU units: `--rnn 150`
- Dense layers dim: `--dense 150`
- Length of sequence for training: `--len 193`
- Dropout rate: `--drop 0.1`

