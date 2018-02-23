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
    def __init__(self, train_img_path, style_img, img_width, img_height):
        '''
        img_width and img_height are the dimensions we expect from the generated image.
        We will resize input content image and input style image to match this dimension.
        Feel free to alter any hyperparameter here and see how it affects your training.
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.train_img_path = train_img_path
        self.style_img = utils.get_resized_image(style_img, img_width, img_height)
        #self.initial_img = utils.generate_noise_image(self.content_img, img_width, img_height)
        self.batch_size=128
        self.img_width=img_width
        self.img_height=img_height


        ###############################
        ## TO DO
        ## create global step (gstep) and hyperparameters for the model
        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        # content_w, style_w: corresponding weights for content loss and style loss
        self.content_w = 1
        self.style_w = 20
        # style_layer_w: weights for different style layers. deep layers have more weights
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0] 
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                trainable=False, name='global_step')
        starter_learning_rate=0.1
        self.lr = tf.train.exponential_decay(starter_learning_rate,self.gstep,100000,0.96,staircase=True)
        ###############################

    def get_data(self):
        with tf.name_scope('data'):
            train_data = utils.get_image_dataset(self.batch_size,self.train_img_path)
            iterator= tf.data.Iterator.from_structure(train_data.output_types,
                                                       train_data.output_shapes)
            img,_= iterator.get_next()

            # reshape the image to make it work with tf.nn.conv2d
            self.img = utils.get_resized_image(img,self.img_width,self.img_height)
            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data

    def transform(self):
        self.TransformNet=transform_net.TransforNet(self.img)
        self.TransformNet.inference()

    def load_vgg(self):
        '''
        Load the saved model parameters of VGG-19, using the input_img
        as the input to compute the output at each layer of vgg.

        During training, VGG-19 mean-centered all images and found the mean pixels
        to be [123.68, 116.779, 103.939] along RGB dimensions. We have to subtract
        this mean from our images.

        '''
        self.vgg = load_vgg.VGG(self.TransformNet.transformed_img)
        self.vgg.load()
        #self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

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
        feature_P = tf.reshape(P, [P.shape[0], P.shape[1] * P.shape[2], P.shape[3]])
        feature_F=tf.reshape(F,[F.shape[0],F.shape[1]*F.shape[2],F.shape[3]])
        self.content_loss = 1/(P.shape[1]*P.shape[2]*P.shape[3])*tf.reduce_sum(tf.square(tf.subtract(feature_P,feature_F)))
        ###############################
        
    def _gram_matrix(self, F, N, M):
        """ Create and return the gram matrix for tensor F
            Hint: you'll first have to reshape F
        """
        ###############################
        ## TO DO
        feature=tf.reshape(F,[F.shape[1]*F.shape[2],F.shape[3]])
        return tf.matmul(tf.transpose(feature),feature)
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
        a_tensor=tf.convert_to_tensor(a)
        A=self._gram_matrix(a_tensor,a_tensor.shape[2],a_tensor.shape[0]*a_tensor.shape[1])
        G=self._gram_matrix(g,g.shape[2],g.shape[0]*g.shape[1])
        return tf.reduce_sum(tf.square(tf.subtract(G,A)))/(4*a.shape[2]*a.shape[2]*a.shape[1]*a.shape[1]*a.shape[3]*a.shape[3])
        ###############################

    def _style_loss(self, A):
        """ Calculate the total style loss as a weighted sum 
        of style losses at all style layers
        Hint: you'll have to use _single_style_loss()
        """
        ###############################
        ## TO DO
        self.style_loss = 0
        gen_img_styles=[getattr(self.vgg, layer) for layer in self.style_layers]
        for index in range(0,len(A)):
            self.style_loss+=self.style_layer_w[index]*self._single_style_loss(A[index],gen_img_styles[index])
        ###############################

    def losses(self):
        with tf.variable_scope('losses') as scope:
            with tf.Session() as sess:
                # assign content image to the input variable
                sess.run(self.input_img.assign(self.content_img))
                # gen_img_content is like a container, to get the specific layer and after running, you apply the img to the layer
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)


            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.style_img))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])                              
            self._style_loss(style_layers)

            ##########################################
            ## TO DO: create total loss. 
            ## Hint: don't forget the weights for the content loss and style loss
            self.total_loss = self.content_w*self.content_loss +self.style_w*self.style_loss
            ##########################################

    def optimize(self):
        ###############################
        ## TO DO: create optimizer
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss,
                                                            global_step=self.gstep)
        ###############################

    def create_summary(self):
        ###############################
        ## TO DO: create summaries for all the losses
        ## Hint: don't forget to merge them
        with tf.name_scope('summaries'):
            tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.histogram('histogram loss', self.total_loss)
            tf.summary.scalar('content_loss', self.content_loss)
            tf.summary.histogram('histogram content_loss', self.content_loss)
            tf.summary.scalar('styles_loss', self.style_loss)
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

    def train(self, n_iters):
        skip_step = 1

        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_style_transfer')

        with tf.Session() as sess:
            
            ###############################
            ## TO DO: 
            ## 1. initialize your variables
            ## 2. create writer to write your grapp
            ###############################
            sess.run(tf.global_variables_initializer())
            sess.run(self.input_img.assign(self.initial_img))
            writer = tf.summary.FileWriter('./graphs/style_transfer_test', tf.get_default_graph())

            ###############################
            ## TO DO: 
            ## 1. create a saver object
            ## 2. check if a checkpoint exists, restore the variables
            ##############################
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_style_transfer/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            initial_step = self.gstep.eval()

            start_time = time.time()
            for index in range(initial_step, n_iters):
                if index >= 5 and index < 20:
                    skip_step = 10
                elif index >= 20:
                    skip_step = 20
                _,l,summaries,out_image = sess.run([self.opt,self.total_loss,self.summary_op,self.input_img])
                if (index + 1) % skip_step == 0:
                    ###############################
                    ## TO DO: obtain generated image, loss, and summary
                    gen_image, total_loss, summary = out_image,l,summaries
                    ###############################
                    
                    # add back the mean pixels we subtracted before
                    gen_image = gen_image + self.vgg.mean_pixels 
                    writer.add_summary(summary, global_step=index)
                    print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                    print('   Loss: {:5.1f}'.format(total_loss))
                    print('   Took: {} seconds'.format(time.time() - start_time))
                    start_time = time.time()

                    filename = 'outputs/%d.png' % (index)
                    utils.save_image(filename, gen_image)

                    if (index + 1) % 20 == 0:
                        ###############################
                        ## TO DO: save the variables into a checkpoint
                        ###############################
                        saver.save(sess, 'checkpoints/convnet_style_transfer/style-transfer-convnet')
                        pass

if __name__ == '__main__':
    setup()
    machine = StyleTransfer('content/deadpool.jpg', 'styles/guernica.jpg', 333, 250)
    machine.build()
    machine.train(30000)
