import re
import os
import time
import numpy as np
import tensorflow as tf
from data import PTBXLDataset
from config import Configuration
from alexnet import AlexNet
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Class for Tester
class Tester(object):

	def __init__(self, cfg, net, testset):

		self.cfg = cfg
		self.net = net
		self.testset = testset
		self.threshold = cfg.THRESHOLD

	# Function for top-1  Accuracy and error
	def compute_accuracy(self, x, y):

		# Get predicted labels
		y_pred_binary = self.predict(x)

		# Subset accuracy - proportion of samples where all the predicted labels exactly match the true labels
		equal = tf.equal(y, y_pred_binary)
		reduce = tf.reduce_all(equal, axis=1)
		accuracy_value = tf.reduce_mean(tf.cast(reduce, tf.float32))
		return accuracy_value
	
	def predict(self, x):
		prediction = self.net(x)
		return tf.cast(prediction >= self.threshold, tf.int32)

	# Function for test
	def test(self):
		prediction_results = []
		labels = []

		for step, (sample, label) in enumerate(self.testset.shuffle(buffer_size=1000)):
			prediction_results.append(self.predict(sample))
			labels.append(label)

		prediction_results = np.array(prediction_results).flatten()
		labels = np.array(labels).flatten()

		return prediction_results, labels
	
	def evaluate_model(self, predicted, actual):
		# Calculate ROC Curve and AUC
		fpr, tpr, thresholds = roc_curve(actual, predicted)
		auc_score = roc_auc_score(actual, predicted)

		# Plot ROC Curve
		plt.figure(figsize=(10, 5))
		plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
		plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Chance')
		plt.title('Receiver Operating Characteristic (ROC) Curve')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend()
		plt.grid()
		plt.savefig('ROC_curve.png')

		# Convert predicted probabilities to binary predictions
		binary_predictions = [1 if p >= 0.5 else 0 for p in predicted]

		# Calculate and display Confusion Matrix
		cm = confusion_matrix(actual, binary_predictions)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
		disp.plot(cmap=plt.cm.Blues)
		plt.title('Confusion Matrix')
		plt.grid(False)
		plt.savefig('Confusion_matrix.png')

		# Calculate Accuracy
		acc = accuracy_score(actual, binary_predictions)
		tf.print(f'Accuracy Score: {acc:.2f}')
	
if __name__ == '__main__':
	i = 0

	# Path for test results
	if not os.path.exists("Tests"):
		os.makedirs('Tests')

	while os.path.exists("Tests/Test%s.txt" % i):
		i += 1

	LOG_PATH = "Tests/Test%s.txt" % i
	def print(msg):
		with open(LOG_PATH,'a') as f:
			f.write(f'{time.ctime()}: {msg}\n')

	cfg = Configuration()

	file_name = 'updated_ptbxl_database.json'
	root_path = '/home/lrbutler/Desktop/ECGSignalClassifer/ptb-xl/'
	dataset = PTBXLDataset(cfg=cfg, meta_file=file_name, root_path=root_path)
	testset = dataset.read_tfrecords('12_lead_normalized_data_1_label/test.tfrecord', buffer_size=64000).batch(1)

	shape = None
	for t in testset.take(1):
		shape = t[0].shape

	tf.print(shape)

	# Get the Alexnet form models
	net = AlexNet(cfg=cfg, training=False)
	net.build(input_shape=shape)
	net.load_weights('model.keras')

	# Create a tester object
	tester = Tester(cfg, net, testset)

	# Call test function on tester object
	pred_labels, actual_labels = tester.test()

	tester.evaluate_model(pred_labels, actual_labels)

