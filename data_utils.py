import os
import numpy as np
import shutil
import sys
from PIL import Image
import scipy.misc
from glob import glob

FOLDERS = ['/Users/yao/large_data_file_no_sync/imaterials/train_chunk0', 
			'/Users/yao/large_data_file_no_sync/imaterials/train_chunk1',
			'/Users/yao/large_data_file_no_sync/imaterials/train_chunk2',
			'/Users/yao/large_data_file_no_sync/imaterials/train_chunk3',
			'/Users/yao/large_data_file_no_sync/imaterials/train_chunk4',
			'/Users/yao/large_data_file_no_sync/imaterials/train_chunk5',
			'/Users/yao/large_data_file_no_sync/imaterials/train_chunk6',
			'/Users/yao/large_data_file_no_sync/imaterials/valid']

PATH = '/Users/yao/large_data_file_no_sync/imaterials'

def get_folder_size(path):
	return sum([len(files) for r, d, files in os.walk(path)]) - 1

def get_class_size(i):
	counter = 0
	for folder in FOLDERS:
		subfolder = os.path.join(folder, str(i))
		counter += len(os.listdir(subfolder))
	return counter

def largest_n_class(n):
	counter = np.zeros(128)
	for i in range(1, 129):
		counter[i-1]=get_class_size(i)
	index = np.argsort(counter)
	return index[-n:][::-1]+1

def generate_class_folder(i):
	os.mkdir(os.path.join(PATH, str(i)))
	for folder in FOLDERS:
		files = os.listdir(os.path.join(folder, str(i)))
		for file in files:
			file_path = os.path.join(folder, str(i)+'/'+file)
			if os.path.isfile(file_path):
				shutil.copy(file_path, os.path.join(PATH, str(i)))

def resize_images_in_folder(folder, folder_resized, resize_h=64):
	imgs_name = [i for i in os.listdir(folder) if i.endswith('jpeg')]

	if not os.path.exists(folder_resized):
		os.makedirs(folder_resized)

	counter = 0
	for img_name in imgs_name:
		img = scipy.misc.imread(os.path.join(folder, img_name)).astype(np.float)
		resize_w = resize_h
		resized_image = scipy.misc.imresize(img,[resize_h, resize_w])
		scipy.misc.imsave(os.path.join(folder_resized, img_name), resized_image)

		counter += 1
		if counter % 100 == 0:
			print(counter)


if __name__=='__main__':
	# class_i = largest_n_class(5)
	# for i in class_i:
	# 	generate_class_folder(i)
	# 	print(get_class_size(i))
	for i in [12, 42, 92, 125]:
		resize_images_in_folder('/Users/yao/Google Drive/projects_ml/GAN_Theories/Datas/imaterials_original/'+str(i),\
			'/Users/yao/Google Drive/projects_ml/GAN_Theories/Datas/imaterials/'+str(i))



	