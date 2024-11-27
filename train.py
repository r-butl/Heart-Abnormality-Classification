import os
import time
import tensorflow as tf
import datetime

from alexnet import AlexNet
from dynamic_cnn import CNN

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
		equal = tf.equal(y, y_pred_binary)
		reduce = tf.reduce_all(equal, axis=1)
		accuracy_value = tf.reduce_mean(tf.cast(reduce, tf.float32))
		return accuracy_value

	def calc_metrics_on_dataset(self, dataset):
		total_loss = 0.0
		total_accuracy = 0.0
		batches = 0

		for images, labels in dataset:
			loss = self.loss('val', images, labels)
			accuracy = self.accuracy('val', images, labels).numpy()

			total_loss += loss.numpy()
			total_accuracy += accuracy
			batches += 1

		loss = total_loss / batches
		accuracy = total_accuracy / batches

		return loss, accuracy
	
	def compute_metrics_and_log(self, epoch):

		val_loss, val_acc = self.calc_metrics_on_dataset(self.valset)
		train_loss, train_acc = self.calc_metrics_on_dataset(self.trainingset)
		
		with self.tensorboard_writer.as_default():
			tf.print(f"train_acc {train_acc:.3f} train_loss {train_loss:.3f} val_acc {val_acc:.3f} val_loss {val_loss:.3f}")
			tf.summary.scalar('train_acc', train_acc, step=epoch)
			tf.summary.scalar('train_loss', train_loss, step=epoch)
			tf.summary.scalar('val_acc', val_acc, step=epoch)
			tf.summary.scalar('val_loss', val_loss, step=epoch)

		# Return the validation loss for the early stopping function
		return val_loss
	
	def check_early_stopping(self, val_loss):

		if val_loss < self.best_val_loss - self.cfg.MIN_DELTA:  # Improvement threshold
			tf.print(f"Updating val_loss {self.best_val_loss:.3f} --> {val_loss:.3f}")
			self.best_val_loss = val_loss
			self.patience_counter = 0
			self.net.save_weights(self.cfg.BEST_WEIGHTS)  # Save best model
			tf.print(f"Validation loss improved to {val_loss:.4f}. Saved model weights.")
		else:
			self.patience_counter += 1
			tf.print(f"No improvement in validation loss for {self.patience_counter} epoch(s).")

		# Check if patience limit is reached
		if self.patience_counter >= self.cfg.PATIENCE:
			tf.print(f"Early stopping triggered...")
			return True
		
		return False

	def train(self, tensorboard_writer):

		self.tensorboard_writer = tensorboard_writer

		self.best_val_loss = float('inf')
		self.patience_counter = 0

		# Run training loop for the number of epochs in configuration file
		for e in range(int(self.epoch.numpy()), self.cfg.EPOCHS):
			self.epoch.assign(e)

			# Run the iterator over the training dataset
			for step, (images, labels) in enumerate(self.trainingset):
				self.global_step.assign_add(1)
				step = self.global_step.numpy() + 1

				# The big boy training code right here
				with tf.GradientTape() as tape:
					loss = self.loss('train', images, labels)
					
				gradients = tape.gradient(loss, self.net.trainable_weights)
				self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

			# Compute metrics every epoch
			val_loss = self.compute_metrics_and_log(epoch=e)
			
			if self.check_early_stopping(val_loss):
				break

		self.net.load_weights(self.cfg.BEST_WEIGHTS)
		print("Restored model weights from the best epoch.")

		# Save the model
		self.net.save(self.cfg.OUTPUT_MODEL)

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

	# If it is resume task, make it true
	resume  = False

	file_name = 'updated_ptbxl_database.json'
	root_path = '/home/lrbutler/Desktop/ECGSignalClassifer/ptb-xl/'

	cfg = Configuration()

	dataset = PTBXLDataset(cfg=cfg, meta_file=file_name, root_path=root_path)
	train = dataset.give_data(mode='train')
	validate = dataset.give_data(mode='validate')

	# Make the Checkpoint path
	if not os.path.exists(cfg.CKPT_PATH):
		os.makedirs(cfg.CKPT_PATH)

	conv_configs = [
		{'filters': 96, 'kernel_size': 11, 'strides': 4, 'padding': 'same', 'pool_size': 3, 'pool_strides': 1},
		{'filters': 128, 'kernel_size': 5, 'strides': 1, 'padding': 'same', 'pool_size': 3, 'pool_strides': 1},
		{'filters': 256, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'pool_size': 3, 'pool_strides': 1},
		{'filters': 256, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'pool_size': 3, 'pool_strides': 1},
	]

	fc_configs = [128, 128]

	net = CNN(conv_configs, fc_configs, num_classes=cfg.NUM_CLASSES, dropout_rate=cfg.DROPOUT, training=True)

	# Make an object of class Trainer
	trainer = Trainer(cfg=cfg, net=net, trainingset=train, valset=validate, resume=resume)

	# Tensorboard start
	run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
	log_dir = os.path.join(cfg.LOG_DIR, run_name)
	tensorboard_writer = tf.summary.create_file_writer(log_dir)

	with tensorboard_writer.as_default():
		# Log conv_configs
		conv_text = "\n".join([str(layer) for layer in conv_configs])
		tf.summary.text("Conv Layer Configurations", conv_text, step=0)

		# Log fc_configs
		fc_text = f"Fully Connected Layers: {fc_configs}"
		tf.summary.text("FC Layer Configurations", fc_text, step=0)

	# Call train function on trainer class
	trainer.train(tensorboard_writer=tensorboard_writer)