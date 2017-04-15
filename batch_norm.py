
import sys
import os
import numpy as np
import tensorflow as tf


class Batch_Normalizer():
    """Batch normalizer using an exponential moving average for mean and variance.
    """
    def __init__(self, model, decay, epsilon): 
        self.prev_layer = model.layers[-1]
        depth = self.prev_layer.get_shape().as_list()[-1]
        self.beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta') 
        self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma') 
        self.epsilon = epsilon
        self.ema_trainer = tf.train.ExponentialMovingAverage(decay=decay)
        self.batch_mean, self.batch_var = tf.nn.moments(
            self.prev_layer, name='moments', axes=list(range(self.prev_layer.get_shape().ndims-1)))
        
    def update_ema(self):
        """Update exponential moving averages with current mean and variance.
        """
        ema_update = self.ema_trainer.apply([self.batch_mean, self.batch_var])
        with tf.control_dependencies([ema_update]):
            return tf.identity(self.batch_mean), tf.identity(self.batch_var)
        
    def load_ema(self):
        """Return current values of exponential moving averages.
        """
        mean = self.ema_trainer.average(self.batch_mean)
        variance = self.ema_trainer.average(self.batch_var)
        return mean, variance
    
    def normalize(self, is_train):
        """Normalize the current batch.
        """
        mean, variance = tf.cond(is_train, self.update_ema, self.load_ema)
        bn = tf.nn.batch_normalization(self.prev_layer, mean, variance, self.beta,
                                       self.gamma, self.epsilon)
        return bn
    