import numpy as np
import scipy.io
import tensorflow as tf
import sys
import utils


class TransformNet(object):
    def __init__(self, input_img):
        self.input_img = input_img
        self.conv_ksize=[9,3,3,3,3,9]
        self.conv_filter=[32,64,128,64,3,3]
        self.residual_conv_ksize=[3,3,3,3,3]
        self.residual_conv_filter=[128,128,128,128,128]
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

    def conv_relu_transpose(self,inputs,filters,k_size,stride,padding,scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            batch_size, rows, cols, in_channels = [i.value for i in inputs.get_shape()]
            kernel = tf.get_variable('out_kernel',
                                     [k_size,k_size,filters,in_channels],
                                     initializer=tf.truncated_normal_initializer())
            new_rows,new_cols = int(rows*stride),int(cols*stride)
            # this is important
            new_shape=[tf.shape(inputs)[0],new_rows,new_cols,filters]
            print('stop here')
            tf_shape = tf.stack(new_shape)
            strides_shape=[1,stride,stride,1]

            conv = tf.nn.conv2d_transpose(inputs,kernel,tf_shape,strides_shape,padding=padding)
            conv = self._instance_norm(conv)
        return tf.reshape(tf.nn.relu(conv,name=scope.name),tf_shape)

    def _instance_norm(self,inputs, train=True):
        batch, rows, cols, channels = [i.value for i in inputs.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)
        shift = tf.Variable(tf.zeros(var_shape))
        scale = tf.Variable(tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (inputs - mu) / (sigma_sq + epsilon) ** (.5)
        return scale * normalized + shift

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

            in_channels_2=conv_block.shape[-1]

            kernel2 = tf.get_variable('kernel2',
                                     [k_size, k_size, in_channels_2, 128],
                                     initializer=tf.truncated_normal_initializer())

            second_conv_block=tf.nn.conv2d(conv_block,kernel2,strides=[1, 1,1, 1], padding=padding)
        return tf.add(second_conv_block,inputs,name=scope.name)

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
                                stride=1,
                                padding='SAME',
                                scope_name='residual1')
        residual2 = self.residual(inputs=residual1,
                                  filters=self.residual_conv_filter[1],
                                  k_size=self.residual_conv_ksize[1],
                                  stride=1,
                                  padding='SAME',
                                  scope_name='residual2')
        residual3 = self.residual(inputs=residual2,
                                  filters=self.residual_conv_filter[2],
                                  k_size=self.residual_conv_ksize[2],
                                  stride=1,
                                  padding='SAME',
                                  scope_name='residual3')
        residual4 = self.residual(inputs=residual3,
                                  filters=self.residual_conv_filter[3],
                                  k_size=self.residual_conv_ksize[3],
                                  stride=1,
                                  padding='SAME',
                                  scope_name='residual4')
        residual5 = self.residual(inputs=residual4,
                                  filters=self.residual_conv_filter[4],
                                  k_size=self.residual_conv_ksize[4],
                                  stride=1,
                                  padding='SAME',
                                  scope_name='residual5')
        conv4 = self.conv_relu_transpose(inputs=residual5,
                                         filters=self.conv_filter[3],
                                         k_size=self.conv_ksize[3],
                                         stride=2,
                                         padding='SAME',
                                         scope_name='conv4')
        conv5 = self.conv_relu_transpose(inputs=conv4,
                                         filters=self.conv_filter[4],
                                         k_size=self.conv_ksize[4],
                                         stride=2,
                                         padding='SAME',
                                         scope_name='conv5')
        '''conv6 = self.conv_relu_transpose(inputs=conv5,
                                         filters=self.conv_filter[5],
                                         k_size=self.conv_ksize[5],
                                         stride=2,
                                         padding='SAME',
                                         scope_name='conv6')
        '''
        preds=tf.tanh(conv5,'last_before_output')
        self.transformed_img=255*(preds+1)/2
        #self.transformed_img = tf.nn.tanh(output)*127.5+255./2










