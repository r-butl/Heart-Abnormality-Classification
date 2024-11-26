import os.path
import time
import tensorflow as tf
import tensorflow.compat.v1 as tfe
from alexnet import AlexNet
from data import PTBXLDataset
from config import Configuration
import utils as ut

# tf.compat.v1.disable_eager_execution()

# Train class for training the model
class Trainer(object):

	def __init__(self, cfg, net, trainingset, valset, resume=False):
		self.cfg = cfg
		self.net = net

		# Datasets
		self.trainingset = trainingset
		self.valset = valset

		# Using Adam optimizer
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.LEARNING_RATE)

		# Loss function
		#	For multilabel classification
		self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

		# Threshold for prediction
		self.threshold = 0.5

		# Create global step
		self.global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)

		# Create checkpoint directory and save checkpoints
		self.epoch = tf.Variable(0, name='epoch', dtype=tf.float32, trainable=False)
		self.checkpoint_dir = self.cfg.CKPT_PATH
		self.checkpoint_encoder = os.path.join(self.checkpoint_dir, 'model')

		self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net, global_step=self.global_step)
		
		# If resume is true continue from saved checkpoint
		if resume:
			latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
			if latest_checkpoint:
				self.checkpoint.restore(latest_checkpoint)
				print(f"Restored checkpoint from {latest_checkpoint}")
			else:
				print("No checkpoint found. Starting from scratch.")

	# Loss calculaions
	def loss(self, mode, x, y):

		# Get predicted labels
		pred = self.net(x)

		# Get the losses using cross entropy loss
		loss_value = self.loss_fn(y, pred)

		return loss_value

	# Accuracy Calulations
	def accuracy(self, mode, x, y):

		# Get predicted labels
		pred = self.net(x)

		# Set to 0 or 1 based on the threshold
		y_pred_binary = tf.cast(pred >= self.threshold, tf.int32)

		# Subset accuracy - proportion of samples where all the predicted labels exactly match the true labels
		accuracy_value = tf.reduce_mean(tf.reduce_all(tf.equal(y, y_pred_binary), axis=1))

		return accuracy_value

	# Training function contining training loop
	def train(self):
		# Get start time
		start_time = time.time()
		step_time = 0.0

		# Run training loop for the number of epochs in configuration file
		for e in range(int(self.epoch.numpy()), self.cfg.EPOCHS):
			self.epoch.assign(e)

			# Run the iterator over the training dataset
			for step, (images, labels) in enumerate(self.trainingset):
				self.global_step.assign_add(1)
				step = self.global_step.numpy() + 1

				step_start_time = int(round(time.time() * 1000))

				# The big boy training code right here
				with tf.GradientTape() as tape:
					loss = self.loss('train', images, labels)
					
				gradients = tape.gradient(loss, self.net.trainable_weights)
				self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

				step_end_time = int(round(time.time() * 1000))
				step_time += step_end_time - step_start_time

				# If it is display step find training accuracy and print it
				if (step % self.cfg.DISPLAY_STEP) == 0:
					l = self.loss('train', images, labels)
					a = self.accuracy('train', images, labels).numpy()
					print ('Epoch: {:03d} Step/Batch: {:09d} Step mean time: {:04d}ms \n\tLoss: {:.7f} Training accuracy: {:.4f}'.format(e, int(step), int(step_time / step), l, a))

				# If it is Validation step find validation accuracy on valdataset and print it
				if (step % self.cfg.VALIDATION_STEP) == 0:
					val_images, val_labels = tfe.Iterator(self.valset.dataset).next()
					l = self.loss('val', val_images, val_labels)
					a = self.accuracy('val', val_images, val_labels).numpy()
					int_time = time.time() - start_time
					print ('Elapsed time: {} --- Loss: {:.7f} Validation accuracy: {:.4f}'.format(ut.format_time(int_time), l, a))

				# If it is save step, save checkpoints
				if (step % self.cfg.SAVE_STEP) == 0:
					encoder_path = self.root1.save(self.checkpoint_encoder)				
		# Save the varaibles at the end of training step
		encoder_path = self.root1.save(self.checkpoint_encoder)				
		print('\nVariables saved\n')


if __name__ == '__main__':
	i = 0
	# Make dir for logs
	if not os.path.exists("logs"):
		os.makedirs('logs')

	while os.path.exists("logs/log%s.txt" % i):
		i += 1

	# Initialize log path
	LOG_PATH = "logs/log%s.txt" % i
	def print(msg):
		with open(LOG_PATH,'a') as f:
			f.write(f'{time.ctime()}: {msg}\n')

	# Get the configuration
	cfg = Configuration()
	net = AlexNet(cfg, training=True)

	# If it is resume task, make it true
	resume  = False

	file_name = 'updated_ptbxl_database.json'
	root_path = '/home/lrbutler/Desktop/ECGSignalClassifer/ptb-xl/'

	dataset = PTBXLDataset(cfg=cfg, meta_file=file_name, root_path=root_path)
	validate = dataset.give_data(mode='validate')

	# Make the Checkpoint path
	if not os.path.exists(cfg.CKPT_PATH):
		os.makedirs(cfg.CKPT_PATH)

	# Make an object of class Trainer
	trainer = Trainer(cfg, net, validate, validate, resume)

	# Call train function on trainer class
	trainer.train()