
import sys
import os
import numpy as np
import tensorflow as tf

    
def create_logs_and_checkpoints_folders(log_dir, save_dir, name):
    """Create logs and checkpoints directories if they don't exists.
    """
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        os.mkdir(os.path.join(log_dir, name))
    elif not os.path.exists(os.path.join(log_dir, name)):
        os.mkdir(os.path.join(log_dir, name))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, name))
    elif not os.path.exists(os.path.join(save_dir, name)):
        os.mkdir(os.path.join(save_dir, name))
        
def add_description(text, save_dir, name):
    """Write run description in a text file in the checkpoints folder.
    """
    with(open(os.path.join(save_dir, name, name + '_description.txt'), mode='w')) as text_file:
        text_file.write(text)
    print("Description added at " + os.path.join(save_dir, name, name + '_description.txt'))
    
def num_parameters(model):
    """Get total number of trainable parameters.
    """
    params = 0
    for var in tf.trainable_variables():
        params += var.get_shape().num_elements()
    return params