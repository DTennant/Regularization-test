def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def prepare_pixel(im):
	import numpy as np
	assert len(im) == 3072
	r = im[0:1024]; r = np.reshape(r, [32, 32, 1])
	g = im[1024:2048]; g = np.reshape(g, [32, 32, 1])
	b = im[2048:3072]; b = np.reshape(b, [32, 32, 1])
	return np.concatenate([b, g, r], -1)

def prepare_data_batch(data_batch):
	import numpy as np
	#assert data_batch.shape[1:] == (3072,)
	p_data_batch = np.zeros((data_batch.shape[0], 32, 32, 3))
	for i in range(data_batch.shape[0]):
		p_data_batch[i] = prepare_pixel(data_batch[i])
	return p_data_batch

def show_im(im):
	import matplotlib.pyplot as plt
	plt.imshow(im)
	plt.show()

def create_one_hot(values):
	import numpy as np
	return np.eye(values.max() + 1)[values]

# return (X_train, y_train), (X_test, y_test) when val = False
# return (X_train, y_train), (X_val, y_val), (X_test, y_test)
def cifar10_data_split(data_dir, val=False, one_hot=True):
	import numpy as np
	print('Started Loading data')
	train_data_dict = []
	for i in range(5):
		train_data_dict.append(unpickle(data_dir + 'data_batch_%d' % (i + 1)))
	test_data_dict = unpickle(data_dir + 'test_batch')
	X_train, y_train = train_data_dict[0].get(b'data'), train_data_dict[0].get(b'labels')
	for i in range(1, 5):
		X_train = np.concatenate((X_train, train_data_dict[i].get(b'data')))
		y_train = np.concatenate((y_train, train_data_dict[i].get(b'labels')))
	assert X_train.shape[1:] == (3072,) and y_train.shape == (50000,)
	print('Finishied loading data')
	if val:
		print('Prepare train, validation and test data (val = True)')
	else:
		print('Prepare train and test data (val = False)')
	#X_train = prepare_data_batch(X_train)
	if one_hot: y_train = create_one_hot(y_train)
	#assert X_train.shape[1:] == (32, 32, 3,)
	X_test, y_test = test_data_dict.get(b'data'), test_data_dict.get(b'labels')
	#X_test = prepare_data_batch(X_test)
	if one_hot: y_test = create_one_hot(np.array(y_test))
	X_val, y_val = [], []
	if val:
		X_val = X_test[:round(X_test.shape[0] / 2), :]
		y_val = y_test[:round(y_test.shape[0] / 2), :]
		X_test = X_test[round(X_test.shape[0] / 2):, :]
		y_test = y_test[round(y_test.shape[0] / 2):, :]
	print('Done!')
	if val:
		return (X_train, y_train), (X_val, y_val), (X_test, y_test)
	else:
		return (X_train, y_train), (X_test, y_test)


# download the dataset, not tested
cifar10_expected_bytes = 170498071
cifar10_url = 'https://www.cs.toronto.edu/~kriz/'
def maybe_download(filename, expected_bytes=cifar10_expected_bytes, force=False, url=cifar10_url):
	from six.moves.urllib.request import urlretrieve
	import os
	if force or not os.path.exists(filename):
		filename, _ = urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', filename)
	else:
		raise Exception('Failed to verify ' + filename + '.')
	return filename

def maybe_extract(filename, force=False):
	import os, sys, tarfile, gzip
	root = os.path.splitext(os.path.splitext(filename)[0])[0]
	if os.path.isdir(root) and not force:
		print('%s already present - Skipping extraction of %s.' % (root, filename))
	else:
		print('Extracting data for %s. Please wait.' % root)
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
	data_folder = 'cifar-10-batches-py'
	print(data_folder)
	return data_folder