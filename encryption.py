from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def byte_to_bits(byte):
	return bin(byte).lstrip('0b').rjust(8, '0')


def load_large_file():
	"""
	open a text file, and save it as ones
	and zeros in large_file which is returned an np array.
	"""
	large_file = list()

	with open('short_bible.txt', 'rb') as textfile:
		while True:
			letter = textfile.read(1)
			if not letter:
				break
			byte = ord(letter)
			try:
				for bit in byte_to_bits(byte):
					large_file.append(int(bit))
			except:
				pass

	return np.array(large_file)

def init_weights(shape):
	weights = tf.random_normal(shape, stdev=0.1)
	return tf.variable(weights)

def encryption_model(data_input, key_input):
	"""
	data is 20 bits, 
	key is 4 random bits.
	concatenate them into input_layer
	
	outputs float values
	TODO apply floor function or some other quantation so output is true binary output
	"""
	
	input_layer = tf.concat([data_input, key_input], 1)


	hidden_1 = tf.layers.dense(inputs=input_layer, units=100, activation=tf.nn.relu)
	hidden_2 = tf.layers.dense(inputs=hidden_1, units=20, activation=tf.nn.relu)


	return hidden_2

def decryption_model(encrypted_input, true_input, key_input):
	"""
	encrypted_data is 20 bits
	key is 4 random bits, same key used to encrypt
	
	output is compared to "data" using L2 loss
	
	"""

	input_layer = tf.concat([encrypted_input, key_input], 1)

	hidden_1 = tf.layers.dense(inputs=input_layer, units=100, activation=tf.nn.relu)
	hidden_2 = tf.layers.dense(inputs=hidden_1, units=100, activation=tf.nn.relu)
	predicted_file = tf.layers.dense(inputs=hidden_2, units=20, activation=tf.nn.relu)


	loss = tf.losses.mean_squared_error(true_input, predicted_file)

	return loss, predicted_file

def file_crack_model(encrypted_file, true_file):
	"""
	tries to derive file using only the encrypted file.
	may only by possible for structured true_file
	"""
	hidden_1 = tf.layers.dense(inputs=encrypted_file, units=100, activation=tf.nn.relu)
	hidden_2 = tf.layers.dense(inputs=hidden_1, units=1000, activation=tf.nn.relu)
	predicted_file = tf.layers.dense(inputs=hidden_2, units=20, activation=tf.nn.relu)

	loss = tf.losses.mean_squared_error(true_file, predicted_file)

	return loss, predicted_file
	

bits = load_large_file()
print("data loaded")
def create_batch(batch_size):
	keys = np.random.randint(0, high=2, size=(batch_size,4))
	
	datas = list()
	for i in range(batch_size):
		start = np.random.randint(bits.size-20)
		datas.append(bits[start:start+20])
	datas = np.array(datas)
	
	return datas, keys



if __name__=='__main__':

	with tf.Session() as sess:

		data_input = tf.placeholder(tf.float32, shape=[None, 20], name='data_input')
		key_input = tf.placeholder(tf.float32, shape=[None, 4], name='key_input')
		
		encrypter = encryption_model(data_input, key_input)
		loss, decrypter = decryption_model(encrypter, data_input, key_input)
		crack_loss, crack = file_crack_model(encrypter, data_input)
		
		crack_train_step = tf.train.GradientDescentOptimizer(0.5).minimize(crack_loss)
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss-crack_loss)


		init = tf.initialize_all_variables()
		sess.run(init)
		for i in range(10000):
			datas, keys = create_batch(1000)	
			_, loss_val = sess.run([train_step, loss], feed_dict={data_input: datas, key_input: keys})	
			_, crack_loss_val = sess.run([crack_train_step, crack_loss], feed_dict={data_input: datas, key_input: keys})
			if i%100 == 0:
				print("loss: ", loss_val, "file cracking loss: ", crack_loss_val)

	
			
