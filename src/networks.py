import os
import math
import numpy as np
from load_data import load_dataset, get_1D_image_features, paths_to_tensor, get_tensors_for_vggface, get_tensors
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from keras_vggface.vggface import VGGFace
from tqdm import tqdm

train_files, train_targets = load_dataset('../faces_aligned/train')
test_files, test_targets = load_dataset('../faces_aligned/test')
valid_files, valid_targets = load_dataset('../faces_aligned/valid')

TRAIN_INSTANCES = len(train_files)
TEST_INSTANCES = len(test_files)
VALID_INSTANCES = len(valid_files)

EPOCHS = 50
BATCH_SIZE = 60
IMAGE_SIZE = 224

#################################################################################################################
##                                                                                                             ##
##                                                RANDOM MODEL                                                 ##
##                                                                                                             ##
#################################################################################################################

def train_random_model():
	X_train = get_1D_image_features(train_files, IMAGE_SIZE)
	y_train = train_targets

	model = DummyClassifier(strategy='stratified',random_state=0)
	model.fit(X_train, y_train)

	return model


def test_random_model(model, img_path, img_label):
	img_features = get_1D_image_features([img_path], IMAGE_SIZE)
	prediction = np.argmax(model.predict(img_features))

	print('Random Model Predicted: %d (True label: %d)' % (prediction+1, img_label))


def get_random_model_accuracy(model):
	X_test = get_1D_image_features(valid_files, IMAGE_SIZE)
	y_test = valid_targets

	predictions = np.argmax(model.predict(X_test), axis=1)
	true_labels = np.argmax(y_test, axis=1)

	return accuracy_score(true_labels, predictions)


#################################################################################################################
##                                                                                                             ##
##                                                BASIC NET                                                    ##
##                                                                                                             ##
#################################################################################################################

def train_basic_net():
	train_tensors = get_tensors(train_files, IMAGE_SIZE)
	test_tensors = get_tensors(test_files, IMAGE_SIZE)

	model = Sequential()
	model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(GlobalAveragePooling2D())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(80, activation='softmax'))

	# model.summary()

	if not os.path.isfile('../saved_models/weights.best.from_scratch.hdf5'):
		checkpointer = ModelCheckpoint(filepath='../saved_models/weights.best.from_scratch.hdf5', 
	                               verbose=1, save_best_only=True)

		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(train_tensors, train_targets,
		        epochs=EPOCHS,
		        validation_data=(test_tensors, test_targets),
		        batch_size=BATCH_SIZE,
		        callbacks=[checkpointer])

	model.load_weights('../saved_models/weights.best.from_scratch.hdf5')

	return model


def test_basic_net(model, img_path, img_label):
	test_tensor = get_tensors([img_path], IMAGE_SIZE)
	prediction = np.argmax(model.predict(test_tensor))

	print('BasicNet Predicted: %d (True label: %d)' % (prediction+1, img_label))


def get_basic_net_accuracy(model):
	valid_tensors = get_tensors(valid_files, IMAGE_SIZE)

	return get_keras_accuracy(model, valid_tensors)

#################################################################################################################
##                                                                                                             ##
##                                                VGGFACE                                                      ##
##                                                                                                             ##
#################################################################################################################

def save_vggface_bottleneck_features():
	print("Getting bottleneck features for VGGFace")

    # build the VGG16 network
	model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')

	train_tensors = get_tensors_for_vggface(train_files, IMAGE_SIZE)
	bottleneck_features_train = [model.predict(np.expand_dims(tensor, axis=0)) for tensor in tqdm(train_tensors)]
	len(bottleneck_features_train)
	np.save(open('../bottleneck_features/vggface_train_2.npy', 'w'), bottleneck_features_train)

	test_tensors = get_tensors_for_vggface(test_files, IMAGE_SIZE)
	bottleneck_features_test = [model.predict(np.expand_dims(tensor, axis=0)) for tensor in tqdm(test_tensors)]
	np.save(open('../bottleneck_features/vggface_test_2.npy', 'w'), bottleneck_features_test)

	valid_tensors = get_tensors_for_vggface(valid_files, IMAGE_SIZE)
	bottleneck_features_validation = [model.predict(np.expand_dims(tensor, axis=0)) for tensor in tqdm(valid_tensors)]
	np.save(open('../bottleneck_features/vggface_valid.npy', 'w'), bottleneck_features_validation)


def train_vggface_net():
	if not (os.path.isfile('../bottleneck_features/vggface_train.npy') and 
		os.path.isfile('../bottleneck_features/vggface_test.npy') and
		os.path.isfile('../bottleneck_features/vggface_valid.npy')):
		save_vggface_bottleneck_features()

	train_data = np.load(open('../bottleneck_features/vggface_train.npy'))
	test_data = np.load(open('../bottleneck_features/vggface_test.npy'))

	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(80, activation='softmax'))

	# TRAIN ONLY IF NEEDED
	if not os.path.isfile('../saved_models/weights.best.from_vggface.hdf5'):
		checkpointer = ModelCheckpoint(filepath='../saved_models/weights.best.from_vggface.hdf5', 
		                           verbose=1, save_best_only=True)

		model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(train_data, train_targets,
		        epochs=EPOCHS,
		        batch_size=BATCH_SIZE,
		        validation_data=(test_data, test_targets),
		        callbacks=[checkpointer])

	model.load_weights('../saved_models/weights.best.from_vggface.hdf5')

	return model


def test_vggface_net(classifier, img_path, img_label):
	model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
	test_tensor = get_tensors_for_vggface([img_path], IMAGE_SIZE)
	img_features = model.predict(test_tensor)
	prediction = np.argmax(classifier.predict(np.expand_dims(img_features, axis=0)))

	print('VGGFace Predicted: %d (True label: %d)' % (prediction+1, img_label))


def get_vggface_net_accuracy(model):
	valid_data = np.load(open('../bottleneck_features/vggface_valid.npy'))
	
	return get_keras_accuracy(model, valid_data)


#################################################################################################################
##                                                                                                             ##
##                                                UTILS                                                        ##
##                                                                                                             ##
#################################################################################################################

def print_dataset_statistics():
	# print statistics about the dataset
	print('There are %s total face images.\n' % len(np.hstack([train_files, valid_files, test_files])))
	print('There are %d training face images.' % len(train_files))
	print('There are %d validation face images.' % len(valid_files))
	print('There are %d test face images.\n'% len(test_files))


def get_keras_accuracy(model, valid_tensors):
	# Get test data
	predictions = np.array([np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in valid_tensors])
	true_labels = np.argmax(valid_targets, axis=1)
	test_accuracy = accuracy_score(true_labels, predictions)
	return test_accuracy
