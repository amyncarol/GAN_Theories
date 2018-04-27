import os,sys
from PIL import Image
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

prefix = './Datas/'

def get_img(img_path, is_crop=False, crop_h=256, resize_h=64):
	img=scipy.misc.imread(img_path).astype(np.float)
	resize_w = resize_h
	if is_crop:
		crop_w = crop_h
		h, w = img.shape[:2]
		j = int(round((h - crop_h)/2.))
		i = int(round((w - crop_w)/2.))
		cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])
	else:
		cropped_image = scipy.misc.imresize(img,[resize_h, resize_w])
	return np.array(cropped_image)/255.0

class Data(object):
	def __init__(self):
		self.z_dim = 100
		self.size = 64
		
	def __call__(self, batch_size, is_crop=False, crop_h=256):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-2:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, is_crop, crop_h, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
	
		return batch_imgs

	def data2fig(self, samples, reshape=False, cmap=None):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			if reshape==False and cmap==None:
				plt.imshow(sample)
			elif reshape==True and cmap!=None:
				plt.imshow(sample.reshape(self.size, self.size), cmap=cmap)
		return fig


class celebA(Data):
	def __init__(self, folder):
		super().__init__()
		self.channel = 3
		self.batch_count = 0
		datapath = os.path.join(prefix, folder)
		self.data = glob(os.path.join(datapath, '*.jpg'))

	def __call__(self, batch_size):
		return super().__call__(batch_size, is_crop=True, crop_h=128)

class imaterials(Data):
	def __init__(self, folder):
		super().__init__()
		self.channel = 3
		self.batch_count = 0
		datapath = os.path.join(prefix, folder)
		self.data = glob(os.path.join(datapath, '*.jpeg'))

class cifar(Data):
	def __init__(self):
		super().__init__()
		self.channel = 3
		self.batch_count = 0
		datapath = prefix + 'cifar10'
		self.data = glob(os.path.join(datapath, '*'))
		
class mnist(Data):
	def __init__(self):
		super().__init__()
		datapath = prefix + 'mnist'
		self.channel = 1
		self.data = input_data.read_data_sets(datapath, one_hot=True)

	def __call__(self,batch_size):
		batch_imgs = np.zeros([batch_size, self.size, self.size, self.channel])

		batch_x,y = self.data.train.next_batch(batch_size)
		batch_x = np.reshape(batch_x, (batch_size, 28, 28, self.channel))
		for i in range(batch_size):
			img = batch_x[i,:,:,0]
			batch_imgs[i,:,:,0] = scipy.misc.imresize(img, [self.size, self.size])
		batch_imgs /= 255.
		return batch_imgs, y

	def data2fig(self, samples):
		return super().data2fig(samples, reshape=True, cmap='Greys_r')
		
if __name__ == '__main__':
	###small test mnist
	# data = mnist()
	# imgs, _ = data(20)
	# print(imgs.shape)

	# fig = data.data2fig(imgs[:16,:,:,:])
	# plt.savefig('../Samples/test.png', bbox_inches='tight')
	# plt.close(fig)

	##small test celebA
	# data = celebA()
	# imgs = data(20)
	# print(imgs.shape)

	# fig = data.data2fig(imgs[:16,:,:,:])
	# plt.savefig('./Samples/test.png', bbox_inches='tight')
	# plt.close(fig)

	##small test imaterials
	data = imaterials(20)
	imgs = data(20)
	print(imgs.shape)

	fig = data.data2fig(imgs[:16,:,:,:])
	plt.savefig('./Samples/test.png', bbox_inches='tight')
	plt.close(fig)

