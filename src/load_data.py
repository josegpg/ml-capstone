import numpy as np
from keras.utils import np_utils
from keras.preprocessing import image
from sklearn.datasets import load_files

#################################################################################################################
##                                                                                                             ##
##                                          DATASET FOR RANDOM MODEL                                           ##
##                                                                                                             ##
#################################################################################################################

def get_image_features(img_path, img_size):
	img = image.load_img(img_path, target_size=(img_size, img_size))
	x = image.img_to_array(img)
	return np.reshape(x, -1)

def get_1D_image_features(img_paths, img_size):
	list_of_instances = [get_image_features(img_path, img_size) for img_path in img_paths]
	return list_of_instances


#################################################################################################################
##                                                                                                             ##
##                                          DATASET FOR BASIC NETWORK                                          ##
##                                                                                                             ##
#################################################################################################################

def load_dataset(path):
    data = load_files(path)
    face_files = np.array(data['filenames'])
    face_targets = np_utils.to_categorical(np.array(data['target']), 80)
    return face_files, face_targets

def path_to_tensor(img_path, img_size):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, img_size):
    list_of_tensors = [path_to_tensor(img_path, img_size) for img_path in img_paths]
    return np.vstack(list_of_tensors)

def get_tensors(img_paths, img_size):
	tensors = paths_to_tensor(img_paths, img_size).astype('float32')/255.0
	return tensors


#################################################################################################################
##                                                                                                             ##
##                                          DATASET FOR VGGFACE MODEL                                          ##
##                                                                                                             ##
#################################################################################################################

def preprocess_for_vggface(tensor):
	x = tensor[:, :, :, ::-1]
	x[:, :, :, 0] -= 93.5940
	x[:, :, :, 1] -= 104.7624
	x[:, :, :, 2] -= 129.1863
	return x

def paths_to_tensor_for_vggface(img_paths, img_size):
    list_of_tensors = [preprocess_for_vggface(path_to_tensor(img_path, img_size)) for img_path in img_paths]
    return np.vstack(list_of_tensors)

def get_tensors_for_vggface(img_paths, img_size):
	tensors = paths_to_tensor_for_vggface(img_paths, img_size).astype('float32')
	return tensors


