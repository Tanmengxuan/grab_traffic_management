import matplotlib.pyplot as plt
import numpy as np

def plot_test(input_test, target, prediction, model_name):

	feature_list = []
	
	for f in input_test:
		feature_list.append(f)

	input_test = np.concatenate(feature_list, axis = 2)
	
	features = input_test.shape[-1]

	target = np.concatenate( [input_test, target], axis = 1 )
	steps = range(target.shape[1])
	start_pred = input_test.shape[1]

	fig, ax = plt.subplots(figsize = (20,8))

	#for feature_idx in range(features):
	features = [199, 200]
	for feature_idx in features:

		plt.plot(steps[:], target[:,:,feature_idx].ravel(), label = 'target_{}'.format(feature_idx), color = '#1f77b4')
		plt.plot(steps[start_pred:], prediction[:,:,feature_idx].ravel(), label = 'prediciton_{}'.format(feature_idx), linestyle='dashed', color = '#ff7f0e')
		
	ax.legend()
	plt.title(model_name)

	fig.savefig('plot_test_sample/' + model_name + '.png')

