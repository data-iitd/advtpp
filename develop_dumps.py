import argparse
import pickle
import numpy as np
import sys, pdb, os

DATA_PATH = os.environ.get("DATA_PATH", "./data")


def mk_dict(len_cats, attr):
	# attr is 'train', 'dev'/'val' or 'test'
	temp = {}
	temp[attr] = []
	temp['dim_process'] = len_cats
	temp['args'] = []
	return temp


def serialize_data(time_data, event_data, store_dict, store_key):
	for i in range(len(time_data)):
		prev = 0
		temp_list = []

		for j in range(len(time_data[i])):
			temp_dict = {}
			val = float(time_data[i][j])		
			temp_dict['time_since_start'] = val
			diff = val - prev
			if diff <= 0.0:
				diff = 0.00001
			temp_dict['time_since_last_event'] = diff
			temp_dict['type_event'] = event_data[i][j] - 1
			prev = val
			temp_list.append(temp_dict)
		store_dict[store_key].append(temp_list)

	return store_dict


def standardize_data(time_data, event_data, scale):
	scale = args.scale
	max_t = 0.0
	min_t = 10000.0

	cat_list = []

	# Getting max and min for times
	for i in range(len(time_data)):
		for j in range(len(time_data[i])):
			cat_list.append(event_data[i][j])
			val = float(time_data[i][j])
			if(val > max_t):
				max_t = val

			if(val < min_t):
				min_t = val

	unique = np.unique(np.asarray(cat_list))
	len_cats = len(unique)

	if 0 in unique:
		incr = 1
	else:
		incr = 0

	for i in range(len(time_data)):
		for j in range(len(time_data[i])):
			val = float(time_data[i][j])
			new_val = float(scale * ((val - min_t)/(max_t - min_t)))
			time_data[i][j] = new_val
			event_data[i][j] += incr

	return time_data, event_data, len_cats


parser = argparse.ArgumentParser()
parser.add_argument('--folder', help="Folder where the dataset text files are available")
parser.add_argument('--fold', type=int, default=1, help="Which fold of the data to use")
parser.add_argument('--scale', type=float, default=10.0, help="Scale of standardization")


if __name__ == "__main__":

	args = parser.parse_args()
	folder = args.folder
	data_fold = args.fold

	# XXX: Currently filenames are hardcoded, may change in future

	file_path = os.path.join(os.path.join(DATA_PATH, folder), f'fold{str(data_fold)}')

	with open(os.path.join(file_path, 'train_ev.txt'), 'r') as in_file:
		eventTrain = [[int(y) for y in x.strip().split()] for x in in_file]

	with open(os.path.join(file_path, 'test_ev.txt'), 'r') as in_file:
		eventTest = [[int(y) for y in x.strip().split()] for x in in_file]

	with open(os.path.join(file_path, 'train_ti.txt'), 'r') as in_file:
		timeTrain = [[float(y) for y in x.strip().split()] for x in in_file]

	with open(os.path.join(file_path, 'test_ti.txt'), 'r') as in_file:
		timeTest = [[float(y) for y in x.strip().split()] for x in in_file]

	# Updating Train and Test
	timeTrain, eventTrain, train_len_cats = standardize_data(timeTrain, eventTrain, args.scale)
	timeTest, eventTest, test_len_cats = standardize_data(timeTest, eventTest, args.scale)

	train_dict = mk_dict(train_len_cats, 'train')
	test_dict = mk_dict(test_len_cats, 'test')
	dev_dict = mk_dict(train_len_cats, 'dev')

	time_len = len(timeTrain)
	event_len = len(eventTrain)
	timeTrain, timeVal = timeTrain[:int(time_len * 0.85)], timeTrain[int(time_len * 0.85):]
	eventTrain, eventVal = eventTrain[:int(event_len * 0.85)], eventTrain[int(event_len * 0.85):]

	serialize_data(timeTrain, eventTrain, train_dict, 'train')
	serialize_data(timeVal, eventVal, dev_dict, 'dev')
	serialize_data(timeTest, eventTest, test_dict, 'test')

	pickle.dump(train_dict, open(os.path.join(file_path, 'train.pkl'), 'wb'))
	pickle.dump(dev_dict, open(os.path.join(file_path, 'val.pkl'), 'wb'))
	pickle.dump(test_dict, open(os.path.join(file_path, 'test.pkl'), 'wb'))

	print("Finished Creating Dumps for "+folder)