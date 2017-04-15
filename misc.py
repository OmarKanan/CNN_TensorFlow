
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

%matplotlib inline


# File parameters
DATA_DIR = 'data'
TRAIN_IMAGES = 'data_train.bin'
TRAIN_TEMPLATES = 'fv_train.bin'
# Data parameters
NUM_TRAIN_IMAGES = 100000
IMAGE_DIM = 48
TEMPLATE_DIM = 128


# Explore data with Numpy
with open(os.path.join(DATA_DIR, TRAIN_IMAGES), 'rb') as f:
    train_data = np.fromfile(f, dtype=np.uint8)
    train_data = train_data.reshape(NUM_TRAIN_IMAGES, IMAGE_DIM, IMAGE_DIM)
with open(os.path.join(DATA_DIR, TRAIN_TEMPLATES), 'rb') as f:
    template_data = np.fromfile(f, dtype=np.float32)
    template_data = template_data.reshape(NUM_TRAIN_IMAGES, TEMPLATE_DIM)    
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
rand = np.random.randint(0, NUM_TRAIN_IMAGES)
for i, ax in enumerate(axes):
    ax.imshow(train_data[i + rand], cmap=plt.cm.gray)
    
    
# Get graph layers
ops = re.findall("<tf.Operation '(\w+)/", str(tf.get_default_graph().get_operations()[::-1]))
unique_ops = []
for op in ops:
    if op not in unique_ops:
        unique_ops.append(op)
unique_ops

# Get parameters
for var in tf.trainable_variables():
    print(var.name, var.get_shape(), var.get_shape().num_elements())
    