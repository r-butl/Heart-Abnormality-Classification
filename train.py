import os
import time
import datetime
import gc
import itertools

from dynamic_cnn import CNN
from alexnet import AlexNet

import tensorflow as tf

from data import PTBXLDataset
from config import Configuration
import utils as ut

# Get the list of GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Set memory growth to True for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Train class for training the model
class Trainer(object):

	def __init__(self, cfg, net, resume=False):
		self.cfg = cfg
		self.net = net

		# Using Adam optimizer
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.LEARNING_RATE)

		# Loss function
		#	For multilabel classification
		self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

		# Threshold for prediction
		self.threshold = cfg.THRESHOLD

		# Create global step
		self.global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)

	# Loss calculaions
	def compute_loss(self, x, y):

		# Get predicted labels
		pred = self.net(x)
		loss_value = self.loss_fn(y, pred)	# Get the losses using cross entropy loss
		return loss_value

	# Accuracy Calulations
	def compute_accuracy(self, x, y):

		# Get predicted labels
		pred = self.net(x)
		y_pred_binary = tf.cast(pred >= self.threshold, tf.int32) # Set to 0 or 1 based on the threshold

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
			loss = self.compute_loss(images, labels)
			accuracy = self.compute_accuracy(images, labels).numpy()

			total_loss += loss.numpy()
			total_accuracy += accuracy
			batches += 1

		loss = total_loss / batches
		accuracy = total_accuracy / batches

		return loss, accuracy
	
	def log_metric_pairs(self, loss, acc, var_name, tensorboard_writer, epoch):
		
		with tensorboard_writer.as_default():
			tf.print(f"{var_name}_acc {acc:.3f} {var_name}_loss {loss:.3f}")
			tf.summary.scalar(f'{var_name}_acc', acc, step=epoch)
			tf.summary.scalar(f'{var_name}_loss', loss, step=epoch)

	def check_early_stopping(self, val_loss):

		if val_loss < self.best_val_loss - self.cfg.MIN_DELTA:  # Improvement threshold
			self.best_val_loss = val_loss
			self.patience_counter = 0
			if not self.cross_validate:
				self.net.save(self.cfg.OUTPUT_MODEL)
				tf.print(f"Validation loss improved to {val_loss:.4f}. Saved model weights.")
		else:
			self.patience_counter += 1
			tf.print(f"No improvement in validation loss for {self.patience_counter} epoch(s).")

		# Check if patience limit is reached
		if self.patience_counter >= self.cfg.PATIENCE:
			tf.print(f"Early stopping triggered...")
			return True
		
		return False

	def train(self, trainset, valset, tensorboard_writer=None, cross_validate=False, epochs=None):
		self.best_val_loss = float('inf')
		self.patience_counter = 0
		self.cross_validate = cross_validate

		# Run training loop for the number of epochs in configuration file
		for e in range(0, self.cfg.EPOCHS if not epochs else epochs):

			# Run the iterator over the training dataset
			for step, (sample, labels) in enumerate(trainset.shuffle(buffer_size=1000)):
				self.global_step.assign_add(1)
				g_step = self.global_step.numpy() + 1

				# The big boy training code right here
				with tf.GradientTape() as tape:
					loss = self.compute_loss(sample, labels)
					
				gradients = tape.gradient(loss, self.net.trainable_weights)
				self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

			# Compute metrics every epoch
			val_loss, val_acc = self.calc_metrics_on_dataset(valset)

			if not cross_validate and tensorboard_writer:
				train_loss, train_acc = self.calc_metrics_on_dataset(trainset)
				self.log_metric_pairs(loss=val_loss, acc=val_acc, var_name='validate', tensorboard_writer=tensorboard_writer, epoch=e)
				self.log_metric_pairs(loss=train_loss, acc=train_acc, var_name='train', tensorboard_writer=tensorboard_writer, epoch=e)

			if self.check_early_stopping(val_loss):
				break

		return self.best_val_loss

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

	file_name = 'updated_ptbxl_database.json'
	root_path = '/home/lrbutler/Desktop/ECGSignalClassifer/ptb-xl/'

	cfg = Configuration()

	dataset = PTBXLDataset(cfg=cfg, meta_file=file_name, root_path=root_path)
	train = dataset.read_tfrecords('data_1_label/train_dataset_1_label.tfrecord', buffer_size=64000)
	cross_validate = False

	if cross_validate:
		batch_sizes = [32, 64, 128]

		hyperparameters = list(itertools.product(batch_sizes))

		for p in hyperparameters:
			tf.print(p)

		k_folds = 5
		dataset_size = sum(1 for _ in train)  # Calculate the total number of samples
		fold_size = dataset_size // k_folds  # Calculate the size of each fold
		fold_config_results = []

		for parameters in hyperparameters:
			fold_results = []

			for fold_idx in range(k_folds):
				# Create a fresh model for each fold
				net = AlexNet(cfg=cfg, training=True)
				trainer = Trainer(cfg=cfg, net=net)

				# Create validation dataset for the current fold
				val_dataset = train.skip(fold_idx * fold_size).take(fold_size)

				# Create training dataset by skipping the validation fold and concatenating the rest
				train_dataset = train.take(fold_idx * fold_size).concatenate(
					train.skip((fold_idx + 1) * fold_size)
				)

				# Run training for this fold
				fold_val_loss = trainer.train(
					trainset=train_dataset.batch(parameters[0]), 
					valset=val_dataset.batch(parameters[0]), 
					cross_validate=True, 
					epochs=5
					)
				tf.print(f'Cross validation fold {fold_idx} loss: {fold_val_loss}')
				fold_results.append(fold_val_loss)

				# Clean up
				del train_dataset
				del val_dataset
				del net
				del trainer
				tf.keras.backend.clear_session()
				gc.collect()

			avg_val_loss = sum(fold_results) / len(fold_results)
			print(f"Cross-validation average validation loss: {avg_val_loss}")
			fold_config_results.append((parameters, avg_val_loss))

		# Best configuration
		best_hyperparameters = max(fold_config_results, key=lambda x: x[1])
		print(f"Best configuration:\n{best_hyperparameters}")

		batch_size = best_hyperparameters[0]
	else:
		batch_size = 128
	
		
	# Tensorboard start
	run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
	log_dir = os.path.join(cfg.LOG_DIR, run_name)
	tensorboard_writer = tf.summary.create_file_writer(log_dir)

	with tensorboard_writer.as_default():
		tf.summary.scalar("Batch Size", batch_size, step=0)

	net = AlexNet(cfg=cfg, training=True)
	trainer = Trainer(cfg=cfg, net=net)

	validate = dataset.read_tfrecords('data_1_label/validate_dataset_1_label.tfrecord', buffer_size=10000)

	# Call train function on trainer class
	print(trainer.train(trainset=train.batch(batch_size), valset=validate.batch(batch_size), cross_validate=False, tensorboard_writer=tensorboard_writer, epochs=cfg.EPOCHS))
