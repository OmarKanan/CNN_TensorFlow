
import sys
import os
import numpy as np
import tensorflow as tf


def load_data(data_dir, images_filename, templates_filename, num_images, image_dim, template_dim):
    """Load data in Numpy arrays.
    """
    with open(os.path.join(data_dir, images_filename), 'rb') as file_:
        images = np.fromfile(file_, dtype=np.uint8)
        images = images.reshape(num_images, image_dim, image_dim, 1)
    with open(os.path.join(data_dir, templates_filename), 'rb') as file_:
        templates = np.fromfile(file_, dtype=np.float32)
        templates = templates.reshape(num_images, template_dim) 
    return images, templates
