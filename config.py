# Configuration used for training

class Configuration(object):

	NUM_CLASSES = 5

	# Train/Validate/Text Split
	TRAIN_PERCENTAGE = 0.70
	TEST_PERCENTAGE = 0.15
	VALIDATE_PERCENTAGE = 0.15

	# Training hyperparameters
	LEARNING_RATE = 1e-4
	MOMENTUM = 0.9
	BATCH_SIZE = 64
	EPOCHS = 50

	# Display steps
	TRAIN_STEP = 10
	VALIDATION_STEP = 50
	SAVE_STEP = 5000

	# Paths for checkpoint
	CKPT_PATH = 'ckpt'
	SUMMARY_PATH = 'summary'
	LOG_DIR = 'logs'
	BEST_WEIGHTS = 'best_weights.weights.h5'
	OUTPUT_MODEL = 'model.keras'	

	# Early Stopping Config
	PATIENCE = 7
	MIN_DELTA = 0.001
	RESTORE_BEST_WEIGHTS = False

	# Net architecture hyperparamaters
	LAMBDA = 5e-4 #for weight decay
	DROPOUT = 0.5

	# Test hyperparameters
	K_PATCHES = 5
	TOP_K = 1