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
	#features = [199, 200]
	features = [ 200, 199]
	target_labels = [ 'target_qp09eq','target_qp09ep']
	prediction_labels = ['predcition_qp09eq', 'predcition_qp09ep']
	for feature_idx in range(len(features)):

		#plt.plot(steps[:], target[:,:,feature_idx].ravel(), label = 'target_qp09eq'.format(feature_idx), color = '#1f77b4')
		#plt.plot(steps[start_pred:], prediction[:,:,feature_idx].ravel(), label = 'prediciton_qp09eq'.format(feature_idx), linestyle='dashed', color = '#ff7f0e')
		
		plt.plot(steps[:], target[:,:,features[feature_idx]].ravel(), label = target_labels[feature_idx])
		plt.plot(steps[start_pred:], prediction[:,:,features[feature_idx]].ravel(), label = prediction_labels[feature_idx], linestyle='dashed')

	ax.legend()
	plt.xlabel('Time intervals index', fontsize = 15)
	plt.ylabel('Demand', fontsize = 15)
	#plt.title(model_name)
	
	#plt.show()
	save_path = 'plot_test_sample/' + model_name + '.png'
	fig.savefig(save_path)
	print ('\n fig saved in {}'.format(save_path))

