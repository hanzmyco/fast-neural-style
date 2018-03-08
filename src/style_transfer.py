""" Implementation in TensorFlow of the paper 
A Neural Algorithm of Artistic Style (Gatys et al., 2016) 

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

For more details, please read the assignment handout:
https://docs.google.com/document/d/1FpueD-3mScnD0SJQDtwmOb1FrSwo1NGowkXzMwPoLH4/edit?usp=sharing
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import tensorflow as tf

import load_vgg
import utils
import transform_net

def setup():
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('outputs')

class StyleTransfer(object):
    def __init__(self, train_img_path, style_img, img_width, img_height,batch_size):
        '''
        img_width and img_height are the dimensions we expect from the generated image.
        We will resize input content image and input style image to match this dimension.
        Feel free to alter any hyperparameter here and see how it affects your training.
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.train_img_path = train_img_path
        self.batch_size = batch_size
        self.style_imgs = utils.get_resized_image(style_img, img_width, img_height)
        self.img_width=img_width
        self.img_height=img_height
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

        ###############################
        ## TO DO
        ## create global step (gstep) and hyperparameters for the model
        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        # content_w, style_w: corresponding weights for content loss and style loss
        self.content_w = 1
        self.style_w = 20
        self.skip_step = 20
        # style_layer_w: weights for different style layers. deep layers have more weights
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0] 
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                trainable=False, name='global_step')
        starter_learning_rate=100.0
        self.lr = tf.train.exponential_decay(starter_learning_rate,self.gstep,100000,0.96,staircase=True)
        #self.lr=10000
        ###############################

    def create_img_placeholder(self):
        '''
        We will use one input_img as a placeholder for the content image,
        style image, and generated image, because:
            1. they have the same dimension
            2. we have to extract the same set of features from them
        We use a variable instead of a placeholder because we're, at the same time,
        training the generated image to get the desirable result.

        Note: image height corresponds to number of rows, not columns.
        '''
        with tf.name_scope('img_placeholder') as scope:
            self.img_place_holder = tf.get_variable('img_placeholder',
                                             shape=([1, self.img_height, self.img_width, 3]),
                                             dtype=tf.float32,
                                             initializer=tf.zeros_initializer(),
                                                    trainable=False)

    def get_data(self):
        with tf.name_scope('data'):
            train_data = utils.get_image_dataset(self.batch_size,self.train_img_path,self.img_width,self.img_height)
            iterator= tf.data.Iterator.from_structure(train_data.output_types,
                                                       train_data.output_shapes)
            self.img = iterator.get_next()

            # reshape the image to make it work with tf.nn.conv2d
            #self.img = utils.get_resized_image(img,self.img_width,self.img_height)
            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.create_img_placeholder()

    def transform(self):
        self.TransformNet=transform_net.TransformNet(self.img)
        self.TransformNet.inference()

    def load_vgg(self):
        '''
        Load the saved model parameters of VGG-19, using the input_img
        as the input to compute the output at each layer of vgg.

        During training, VGG-19 mean-centered all images and found the mean pixels
        to be [123.68, 116.779, 103.939] along RGB dimensions. We have to subtract
        this mean from our images.

        '''
        self.TransformNet.transformed_img -= self.mean_pixels
        self.vgg_transformed = load_vgg.VGG(self.TransformNet.transformed_img)
        self.vgg_transformed.load()

        self.style_imgs = np.expand_dims(self.style_imgs, 0)
        self.style_imgs -= self.mean_pixels
        
        #self.vgg_style_imgs = load_vgg.VGG(self.style_imgs)
        self.vgg_style_imgs = load_vgg.VGG(self.img_place_holder)
        #self.vgg_style_imgs = load_vgg.VGG(self.style_imgs)
        self.vgg_style_imgs.load()

        self.img -= self.mean_pixels
        self.vgg_content_imgs = load_vgg.VGG(self.img)
        self.vgg_content_imgs.load()



    def _content_loss(self, P, F):
        ''' Calculate the loss between the feature representation of the
        content image and the generated image.
        
        Inputs: 
            P: content representation of the content image
            F: content representation of the generated image
            Read the assignment handout for more details

            Note: Don't use the coefficient 0.5 as defined in the paper.
            Use the coefficient defined in the assignment handout.
        '''
        ###############################
        ## TO DO  actually reshape or not it's the same
        #feature_P = tf.reshape(P, [P.shape[0], P.shape[1] * P.shape[2], P.shape[3]])
        #feature_F=tf.reshape(F,[F.shape[0],F.shape[1]*F.shape[2],F.shape[3]])
        factor=1/(P.shape[1]*P.shape[2]*P.shape[3]).value
        self.content_loss = factor*tf.reduce_sum(tf.square(tf.subtract(P,F)),[1,2,3])
        ###############################
        
    def _gram_matrix(self, F):
        """ Create and return the gram matrix for tensor F
            Hint: you'll first have to reshape F
        """
        ###############################
        ## TO DO
        image_feature= tf.reshape(F,[tf.shape(F)[0],(F.shape[1]*F.shape[2]).value,F.shape[3].value])
        return tf.matmul(tf.transpose(image_feature,perm=[0,2,1]),image_feature)
        ###############################

    def _single_style_loss(self, a, g):
        """ Calculate the style loss at a certain layer
        Inputs:
            a is the feature representation of the style image at that layer
            g is the feature representation of the generated image at that layer
        Output:
            the style loss at a certain layer (which is E_l in the paper)

        Hint: 1. you'll have to use the function _gram_matrix()
            2. we'll use the same coefficient for style loss as in the paper
            3. a and g are feature representation, not gram matrices
        """
        ###############################
        ## TO DO
        A=self._gram_matrix(a)
        G=self._gram_matrix(g)
        return tf.reduce_sum(tf.square(tf.subtract(G,A)),[1,2])/(4*(a.shape[2]*a.shape[2]*a.shape[1]*a.shape[1]*a.shape[3]*a.shape[3]).value)
        ###############################

    def _style_loss(self, A,B):
        """ Calculate the total style loss as a weighted sum 
        of style losses at all style layers
        Hint: you'll have to use _single_style_loss()
        """
        ###############################
        ## TO DO
        self.style_loss = 0
        for index in range(0,len(A)):
            self.style_loss+=self.style_layer_w[index]*self._single_style_loss(A[index],B[index])
        ###############################

    def losses(self):
        with tf.variable_scope('losses') as scope:
            with tf.Session() as sess:
                gen_img_content = getattr(self.vgg_transformed, self.content_layer)
                content_img_content = getattr(self.vgg_content_imgs, self.content_layer)
            self._content_loss(content_img_content, gen_img_content)

            with tf.Session() as sess:
                print('stop here 1')
                print('stop here 2')
                sess.run(self.img_place_holder.assign(self.style_imgs))
                style_layers = [getattr(self.vgg_style_imgs, layer) for layer in self.style_layers]
                gen_img_style_layers=[getattr(self.vgg_transformed, layer) for layer in self.style_layers]
            self._style_loss(style_layers,gen_img_style_layers)

            ##########################################
            ## TO DO: create total loss. 
            ## Hint: don't forget the weights for the content loss and style loss
            self.total_loss = tf.reduce_mean(self.content_w*self.content_loss) +tf.reduce_mean(self.style_w*self.style_loss)
            ##########################################

    def optimize(self):
        ###############################
        ## TO DO: create optimizer
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss,
                                                            global_step=self.gstep)
        ###############################:x


    def create_summary(self):
        ###############################
        ## TO DO: create summaries for all the losses
        ## Hint: don't forget to merge them
        with tf.name_scope('summaries'):
           # tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.histogram('histogram loss', self.total_loss)
            #tf.summary.scalar('content_loss', self.content_loss)
            tf.summary.histogram('histogram content_loss', self.content_loss)
#            tf.summary.scalar('styles_loss', self.style_loss)
            tf.summary.histogram('histogram styles_loss', self.style_loss)
            self.summary_op = tf.summary.merge_all()
        ###############################


    def build(self):
        self.get_data()
        self.transform()
        self.load_vgg()
        self.losses()
        self.optimize()
        self.create_summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.total_loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/fast-neural-style/', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/fast-neural-style')
        writer = tf.summary.FileWriter('./graphs/fast-neural-style', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/fast-neural-style/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)

        writer.close()

if __name__ == '__main__':
    setup()
    windows_path='C:\\Users\\hanzhan.REDMOND\\Source\\fast-neural-style\\data\\train\\'
    windows_path2='C:\\Users\\hanzhan.REDMOND\\OneDrive - Microsoft\\cocodata\\train2014\\'
    mac_path='/Users/hanz/Deep_Learning/fast-neural-style/data/test'
    linux_path='/home/hanzhan/dl/fast-neural-style/data/training/'
    machine = StyleTransfer(windows_path, '../data/styles/guernica.jpg', 256, 256,20)
    machine.build()
    machine.train(n_epochs=30)
