
import sys
import os
from time import time
import numpy as np
import tensorflow as tf

import data_reader
from batch_norm import Batch_Normalizer
import utils


class CNN_Model():
    """Convolutional neural network model.
    """
    def __init__(self, name, data_dir, log_dir, save_dir, image_dim, template_dim, train_images, train_templates,
                 valid_images, valid_templates, num_train_images, num_valid_images, ema_decay, bn_epsilon):
        self.name = name
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.image_dim = image_dim
        self.template_dim = template_dim
        self.data_dir = data_dir
        self.train_images = train_images
        self.train_templates = train_templates
        self.valid_images = valid_images
        self.valid_templates = valid_templates
        self.num_train_images = num_train_images
        self.num_valid_images = num_valid_images
        self.ema_decay = ema_decay
        self.bn_epsilon = bn_epsilon
        self.initialize_graph()
        
    def initialize_graph(self):
        """Initialize the graph model.
        """
        utils.create_logs_and_checkpoints_folders(self.log_dir, self.save_dir, self.name)
        self.load_data()
        tf.reset_default_graph()
        self.images = tf.placeholder(tf.float32, [None, self.image_dim, self.image_dim, 1], 'images')
        self.templates = tf.placeholder(tf.float32, [None, self.template_dim], 'templates')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='train_flag')
        self.validation_loss = tf.placeholder(tf.float32, shape=[], name='validation_loss')
        self.global_step = tf.Variable(0, trainable=False, dtype = tf.int64, name='global_step')
        self.layers = [self.images]
        
    def load_data(self):
        """Load train and validation data.
        """
        self.train_images, self.train_templates = data_reader.load_data(
            self.data_dir, self.train_images, self.train_templates, self.num_train_images, self.image_dim, self.template_dim)
        self.valid_images, self.valid_templates = data_reader.load_data(
            self.data_dir, self.valid_images, self.valid_templates, self.num_valid_images, self.image_dim, self.template_dim)
    
    def get_batch_data(self, batch_size, train=True):
        """Get train data batch.
        """
        if train:
            indices = np.random.randint(0, self.num_train_images, size=batch_size)
            images_batch = self.train_images.take(indices, axis=0)
            templates_batch = self.train_templates.take(indices, axis=0)
            return images_batch, templates_batch
        else:
            indices = np.random.randint(0, self.num_valid_images, size=batch_size)
            images_batch = self.valid_images.take(indices, axis=0)
            templates_batch = self.valid_templates.take(indices, axis=0)
            return images_batch, templates_batch
                    
    def add_conv(self, name, ksize, stride=[1, 1, 1, 1], padding='SAME', dropout=0, residual=0, prev_layer=-1, relu=True):
        """Add convolution layer.
        """
        with tf.variable_scope(name): 
            kernel = tf.get_variable('kernel', ksize, initializer=tf.truncated_normal_initializer())
            bias = tf.get_variable('bias', [ksize[-1]], initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(self.layers[prev_layer], kernel, stride, padding) + bias
            if residual:
                prev_layer = self.layers[residual]
                prev_shape = prev_layer.get_shape().as_list()
                self.add_conv('id_mapping', ksize=[1, 1, prev_shape[-1], ksize[-1]], prev_layer=residual, relu=False)
                id_mapping = self.layers.pop(-1)
                conv = conv + id_mapping
            if relu:
                conv = tf.nn.relu(conv)
            if dropout:
                self.layers.append(self.add_dropout(conv, dropout))
            else:
                self.layers.append(conv)
            
    def add_pool(self, name, pool_type, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME', dropout=0):
        """Add pooling layer.
        """
        with tf.variable_scope(name):
            if pool_type == 'max':
                pool = tf.nn.max_pool(self.layers[-1], ksize, stride, padding)
            elif pool_type == 'avg':
                pool = tf.nn.avg_pool(self.layers[-1], ksize, stride, padding)
            if dropout:
                self.layers.append(self.add_dropout(pool, dropout))
            else:
                self.layers.append(pool)
            
    def add_batch_norm(self, name, dropout=0):
        """Add batch_normalization layer.
        """
        with tf.variable_scope(name): 
            bn = Batch_Normalizer(self, self.ema_decay, self.bn_epsilon).normalize(self.is_train)
            if dropout:
                self.layers.append(self.add_dropout(bn, dropout))
            else:
                self.layers.append(bn)
            
    def add_fully_connected(self, name, size, dropout=0):
        """Add fully-connected layer.
        """
        with tf.variable_scope(name):
            input_features = np.product(self.layers[-1].get_shape().as_list()[1:])
            weights = tf.get_variable('weights', [input_features, size])
            bias = tf.get_variable('bias', [size])
            flat_input = tf.reshape(self.layers[-1], [-1, input_features])
            fc = tf.matmul(flat_input, weights) + bias
            if dropout:
                self.layers.append(self.add_dropout(fc, dropout))
            else:
                self.layers.append(fc)
            
    def add_dropout(self, tensor, dropout):
        """Add dropout to tensor.
        """
        keep_prob = tf.cond(self.is_train, lambda: tf.constant(1-dropout, tf.float32),
                            lambda: tf.constant(1, tf.float32), name='keep_prob')
        return tf.nn.dropout(tensor, keep_prob)
    
    def add_mse_loss(self, name):
        """Add quadratic loss.
        """
        with tf.variable_scope(name):
            loss = tf.squared_difference(self.templates, self.layers[-1])
            self.loss = tf.reduce_mean(loss)
            
    def add_adam_optimizer(self, name, init_learning_rate, decay):
        """Add AdamOptimizer to minimize loss.
        """
        with tf.variable_scope(name):
            if decay:
                self.learning_rate = tf.div(init_learning_rate, tf.cast(self.global_step + 1, tf.float32),
                                            name='learning_rate')
            else:
                self.learning_rate = tf.constant(init_learning_rate, name='learning_rate')
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)
            
    def add_summaries(self, name):
        """Create summaries operation.
        """
        with tf.variable_scope(name):
            s1 = tf.summary.scalar('mse_loss', self.loss)
            s2 = tf.summary.histogram('mse_loss_hist', self.loss)
            self.summary_op = tf.summary.merge([s1, s2])
            s3 = tf.summary.scalar('validation_loss', self.validation_loss)
            s4 = tf.summary.histogram('validation_loss_hist', self.validation_loss)
            self.valid_loss_summary = tf.summary.merge([s3, s4])
        
    def initialize_session(self, restore):
        """Initialize a new session.
        """
        self.sess = tf.Session()
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.name), self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.save_dir, self.name))
        if restore and ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Restored session ' + ckpt.model_checkpoint_path)
        
    def close_session(self):
        """Close current session.
        """
        self.writer.close()
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()
        
    def train(self, n_batches, step_size, batch_size, valid_batch_size, save):
        """Train model for n_epochs.
        """
        step_loss = 0
        init_step = self.sess.run(self.global_step)
        for i in range(init_step, init_step + n_batches):
            image_batch, template_batch = self.get_batch_data(batch_size)
            _, batch_loss, summary = self.sess.run([self.optimizer, self.loss, self.summary_op],
                                                   {self.images : image_batch,
                                                    self.templates : template_batch,
                                                    self.is_train: True})
            step_loss += batch_loss    
            if i > step_size:
                self.writer.add_summary(summary, i)
            
            if (i+1) % step_size == 0:
                valid_loss = self.validate_loss(i, valid_batch_size)
                print('Batch %d:' % (i+1))
                print('--> Train loss = %f' % (step_loss / step_size))
                print('--> Valid loss = %f' % valid_loss)
                step_loss = 0
                if save:
                    print('Saving session ' + os.path.join(self.save_dir, self.name, 'model.ckpt-%d' % (i+1)))
                    self.saver.save(self.sess, os.path.join(self.save_dir, self.name, 'model.ckpt'), (i+1)) 
                
    def validate_loss(self, i, valid_batch_size):
        """Get total loss on the validation data.
        """
        total_valid_loss = 0
        n_batches = int(self.num_valid_images / valid_batch_size)
        for j in range(n_batches):
            images = self.valid_images[j*valid_batch_size:(j+1)*valid_batch_size, :, :, :]
            templates = self.valid_templates[j*valid_batch_size:(j+1)*valid_batch_size, :]
            total_valid_loss += self.sess.run(self.loss, {self.images:images, self.templates:templates, self.is_train:False})
        total_valid_loss /= n_batches
        summary = self.sess.run(self.valid_loss_summary, {self.validation_loss : total_valid_loss})
        self.writer.add_summary(summary, i+1)
        return total_valid_loss
    