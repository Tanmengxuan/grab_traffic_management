import numpy as np
import pandas as pd
import json
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--input_raw' ,type = str, help = 'input path of raw test file')
args = parser.parse_args()

class Process(object):

	def __init__(self, raw_data, geohash_dict):

		self.raw_data = raw_data
		self.geohash_dict = geohash_dict
		self.daytime_dict = self.get_daytime_dict()

		#store all raw data values in array
		self.geohash_array = raw_data['geohash6'].values
		self.daytime_array = raw_data['day'].astype(str).values + ',' + raw_data['timestamp'].values
		self.demand_array = raw_data['demand'].values


	#create index dict of unique time stamps
	def get_daytime_dict(self):

		daytime_dict = {}
		count = 0
		for day in range( self.raw_data['day'].min(), self.raw_data['day'].max() + 1 ):
			for hour in range(0, 24):
				for mins in [0, 15, 30, 45]:
					daytime = '{},{}:{}'.format(day, hour, mins)
					daytime_dict[daytime] = count
					
					count += 1

		return daytime_dict

	def get_processed_data(self):

		# initialize nested list to create processed data array of (timesteps, num of unique geohash6 )
		row = len(self.daytime_dict)
		col = len(self.geohash_dict)

		assert col == 1329, 'number of unique geohashs in test set must be the same as that of the training set'

		data_nested_list =[]
		for i in range(row):
			data_nested_list.append([0 for j in range(col)])

		# append values to nested list
		for idx in range(len(self.raw_data)):
			row_idx = self.daytime_dict[ self.daytime_array[idx] ]

			if self.geohash_array[idx] not in self.geohash_dict:
				print ('location {} was not present in training data'.format(self.geohash_array[idx]))
				continue
			else:
				col_idx = self.geohash_dict[ self.geohash_array[idx] ]
			
			data_nested_list[row_idx][col_idx] = self.demand_array[idx]

		processed_data_array = np.array(data_nested_list)

		return processed_data_array

if __name__ == '__main__':

	raw_data = pd.read_csv(args.input_raw, sep = ',')
	fp = json.loads((open('geohash_8c.json','r')).read())
	geohash_dict = fp['coord_dict']

	processed_data_array = Process(raw_data, geohash_dict).get_processed_data()

	# save data array to file
	hf = h5py.File( 'processed_data/test_processed.h5', 'w')
	hf.create_dataset('data', data =  processed_data_array)
	hf.close()

	print ('\n Processed file saved')

