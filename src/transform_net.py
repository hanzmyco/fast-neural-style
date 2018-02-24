import numpy as np
import scipy.io
import tensorflow as tf
import sys
import utils


class TransformNet(object):
    def __init__(self, input_img):
        self.input_img = input_img
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        self.conv_ksize=[2,2,2,2,2,2]
        self.conv_filter=[32,32,32,32,32,32]
        self.residual_conv_ksize=[2,2,2,2]
        self.residual_conv_filter=[32,32,32,32]
        self.training = True
        self.n_test = 10000

    def conv_relu(self,inputs,filters,k_size,stride,padding,scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            in_channels = inputs.shape[-1]
            kernel = tf.get_variable('kernel',
                                     [k_size, k_size, in_channels, filters],
                                     initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases',
                                     [filters],
                                     initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
        return tf.nn.relu(conv + biases, name=scope.name)

    def residual(self,inputs,filters,k_size,stride,padding,scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            in_channels = inputs.shape[-1]
            kernel = tf.get_variable('kernel',
                                     [k_size, k_size, in_channels, filters],
                                     initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases',
                                     [filters],
                                     initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)

        conv_block=tf.nn.relu(conv + biases)
        return tf.add(conv_block,inputs,name=scope.name)

    def inference(self):
        conv1 = self.conv_relu(inputs=self.input_img,
                          filters=self.conv_filter[0],
                          k_size=self.conv_ksize[0],
                          stride=1,
                          padding='SAME',
                          scope_name='conv1')
        conv2 = self.conv_relu(inputs=conv1,
                               filters=self.conv_filter[1],
                               k_size=self.conv_ksize[1],
                               stride=2,
                               padding='SAME',
                               scope_name='conv2')
        conv3 = self.conv_relu(inputs=conv2,
                               filters=self.conv_filter[2],
                               k_size=self.conv_ksize[2],
                               stride=2,
                               padding='SAME',
                               scope_name='conv3')
        residual1=self.residual(inputs=conv3,
                                filters=self.residual_conv_filter[0],
                                k_size=self.residual_conv_ksize[0],
                                stride=2,
                                padding='SAME',
                                scope_name='residual1')
        residual2 = self.residual(inputs=residual1,
                                  filters=self.residual_conv_filter[1],
                                  k_size=self.residual_conv_ksize[1],
                                  stride=2,
                                  padding='SAME',
                                  scope_name='residual2')
        residual3 = self.residual(inputs=residual2,
                                  filters=self.residual_conv_filter[2],
                                  k_size=self.residual_conv_ksize[2],
                                  stride=2,
                                  padding='SAME',
                                  scope_name='residual3')
        residual4 = self.residual(inputs=residual3,
                                  filters=self.residual_conv_filter[3],
                                  k_size=self.residual_conv_ksize[3],
                                  stride=2,
                                  padding='SAME',
                                  scope_name='residual4')
        residual5 = self.residual(inputs=residual4,
                                  filters=self.residual_conv_filter[4],
                                  k_size=self.residual_conv_ksize[4],
                                  stride=2,
                                  padding='SAME',
                                  scope_name='residual5')
        conv4 = self.conv_relu(inputs=residual5,
                               filters=self.conv_filter[3],
                               k_size=self.conv_ksize[3],
                               stride=1,
                               padding='SAME',
                               scope_name='conv4')
        conv5 = self.conv_relu(inputs=conv4,
                               filters=self.conv_filter[4],
                               k_size=self.conv_ksize[4],
                               stride=0.5,
                               padding='SAME',
                               scope_name='conv5')
        conv6 = self.conv_relu(inputs=conv5,
                               filters=self.conv_filter[5],
                               k_size=self.conv_ksize[5],
                               stride=0.5,
                               padding='SAME',
                               scope_name='conv6')

        self.transformed_img = 255*(tf.tanh(conv6,'last_before_output')+1)/2










