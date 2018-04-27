import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os,sys

sys.path.append('utils')
from nets import *
from datas import *

def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

class BEGAN():
	def __init__(self, generator, discriminator, data, model_path):
		self.generator = generator
		self.discriminator = discriminator
		self.data = data
		self.model_path = model_path

		# data
		self.z_dim = self.data.z_dim
		self.size = self.data.size
		self.channel = self.data.channel

		self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

		# began parameters
		self.k_t =  tf.placeholder(tf.float32, shape=[]) # weighting parameter which constantly updates during training
		gamma = 0.75  # diversity ratio, used to control model equibilibrium.
		lambda_k = 0.001 # learning rate for k. Berthelot et al. use 0.001

		# nets
		self.G_sample = self.generator(self.z)

		self.D_real = self.discriminator(self.X)
		self.D_fake = self.discriminator(self.G_sample, reuse = True)
		
		# loss
		L_real = tf.reduce_mean(tf.abs(self.X - self.D_real))
		L_fake = tf.reduce_mean(tf.abs(self.G_sample - self.D_fake))

		self.D_loss = L_real - self.k_t * L_fake
		self.G_loss = L_fake
		
		self.k_tn = self.k_t + lambda_k * (gamma*L_real - L_fake)
		self.M_global = L_real + tf.abs(gamma*L_real - L_fake)		
	
		# solver
		self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
		self.D_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.D_loss, var_list=self.discriminator.vars)
		self.G_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.G_loss, var_list=self.generator.vars)
		
		self.saver = tf.train.Saver()
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.model_name = os.path.join(self.model_path, 'began.ckpt')

		##tf summary
		tf.summary.scalar('M_global', self.M_global)
		tf.summary.scalar('k_t', self.k_t)
		tf.summary.scalar('L_real', L_real)
		tf.summary.scalar('G_loss', self.G_loss)
		tf.summary.scalar('D_loss', self.D_loss)
		#tf.summary.scalar('learning_rate', self.learning_rate)
		self.summary = tf.summary.merge_all()
		self.summary_writer = tf.summary.FileWriter(self.model_path, self.sess.graph)

	def train(self, sample_dir, training_epoches = 500000, batch_size = 16):
		fig_count = 0
		self.sess.run(tf.global_variables_initializer())
		#self.saver.restore(self.sess, self.model_name)		

		k_tn = 0
		learning_rate_initial = 1e-4
		for epoch in range(training_epoches):
			learning_rate =  learning_rate_initial * pow(0.5, epoch // 50000)
			# update D and G
			X_b = self.data(batch_size)
			_, _, k_tn = self.sess.run(
				[self.D_solver, self.G_solver, self.k_tn],
				feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim), self.k_t: min(max(k_tn, 0.), 1.), self.learning_rate: learning_rate}
				)
			# save img, model. print loss
			if epoch % 100 == 0 or epoch < 100:
				D_loss_curr, G_loss_curr, M_global_curr, summary_str = self.sess.run(
						[self.D_loss, self.G_loss, self.M_global, self.summary],
            			feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim), self.k_t: min(max(k_tn, 0.), 1.)})
				print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; M_global: {:.4}; k_t: {:.6}; learning_rate:{:.8}'.format(epoch, D_loss_curr, G_loss_curr, M_global_curr, min(max(k_tn, 0.), 1.), learning_rate))
				self.summary_writer.add_summary(summary_str, epoch)
				self.summary_writer.flush()

				if epoch % 1000 == 0:
					X_s, real, samples = self.sess.run([self.X, self.D_real, self.G_sample], feed_dict={self.X: X_b[:16,:,:,:], self.z: sample_z(16, self.z_dim)})

					fig = self.data.data2fig(X_s)
					plt.savefig('{}/{}.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
					plt.close(fig)

					fig = self.data.data2fig(real)
					plt.savefig('{}/{}_d.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
					plt.close(fig)

					fig = self.data.data2fig(samples)
					plt.savefig('{}/{}_r.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
					plt.close(fig)

					fig_count += 1

				if epoch % 5000 == 0:
					self.saver.save(self.sess, self.model_name)

if __name__ == '__main__':

	# constraint GPU
	#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	# input and output folders
	sample_dir = 'Samples/began_celebA_2000_second_try'
	model_dir = 'Models/began_celebA_2000_second_try'
	data_dir = 'celebA_2000'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	# param
	generator = G_conv()
	discriminator = D_autoencoder()
	data = celebA(data_dir)

	# run
	began = BEGAN(generator, discriminator, data, model_dir)
	began.train(sample_dir)

