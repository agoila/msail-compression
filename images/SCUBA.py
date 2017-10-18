''' author: samtenka
    changed: 2017-10-09
    created: 2017-10-07
    credits: www.tensorflow.org/get_started/mnist/pros
    descr: Train SCUBA lossless compression architecture
    usage: Run `python SCUBA.py`.
'''

import tensorflow as tf
import numpy as np
import glob

###############################################################################
#                         0. PROGRAM (HYPER)PARAMETERS                        #
###############################################################################

# 0.0. Data
SIDE = 28
CHANNELS = 1 

# 0.1. Architecture
EPSILON = 1.0/256
RHO = 1.0/16

# 0.2. Loss
GAMMA = 1.0
BIG_M = 100.0

# 0.3. Optimizer
TRAIN_TIME= 100000
BATCH_SIZE= 50
LEARNING_RATE = 0.00001
#LEARNING_RATE = 0.000001


###############################################################################
#                            1. READ DATASET                                  #
###############################################################################

# 0.0. MNIST is a classic image-classification dataset.  Its images are 28x28 
#      grayscale photographs of handwritten digits (0 through 9).  Note that
#      we load the labels in one-hot form.  This makes defining a loss function
#      easier: 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')
#mnist = input_data.read_data_sets('FASHION_data')

def get_batch(size=BATCH_SIZE, train=True):
    ''' Return `inputs` of shape (size, 28*28)
        randomly sampled from the full data. 
    '''
    inputs, _ = (mnist.train if train else mnist.test).next_batch(size)
    return inputs

###############################################################################
#                         2. BUILD COMPUTATION GRAPH                          #
###############################################################################

# 2.0. Placeholders for the data to which to fit the model:
TrueInputs = tf.placeholder(tf.float32, shape=[None, SIDE * SIDE])

# 2.1. MODEL HYPERPARAMETERS:
LearningRate = tf.placeholder(dtype=tf.float32)

# 2.1. MODEL PARAMETERS:
W1 = tf.get_variable('W1', shape=[4, 4, CHANNELS*2, 16], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
B1 = tf.get_variable('B1', shape=[                  16], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
W2 = tf.get_variable('W2', shape=[4, 4,         16, 32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
B2 = tf.get_variable('B2', shape=[                  32], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

W3 = tf.get_variable('W3', shape=[4, 4,         32, 32], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
B3 = tf.get_variable('B3', shape=[                  32], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

W4 = tf.get_variable('W4', shape=[4, 4,         16, 48], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
B4 = tf.get_variable('B4', shape=[              16    ], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
W5 = tf.get_variable('W5', shape=[4, 4, CHANNELS*2, 18], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
B5 = tf.get_variable('B5', shape=[      CHANNELS*2    ], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

# 2.2. BUILD CLASSIFIER:
def conv2d(x, W, stride=2):
    ''' Linear convolutional map  '''
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def deconv2d(x, W, output_shape, stride=2):
    ''' Linear DEconvolutional map  '''
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')

def lrelu(x):
    ''' leaky ReLU activation function '''
    return tf.maximum(0.1*x, x)

InputImages = tf.reshape(TrueInputs, [-1, SIDE, SIDE, CHANNELS])  #   L   x L   x  1 
NbBatches = tf.shape(InputImages)[0]
Thresholds = tf.tile(tf.reshape(tf.random_uniform(shape=[NbBatches]), [-1, 1, 1, 1]), [1, SIDE, SIDE, CHANNELS])
Mask = tf.cast(tf.where(tf.random_uniform(tf.shape(InputImages)) < Thresholds, tf.ones(tf.shape(InputImages)), tf.zeros(tf.shape(InputImages))), tf.float32)
#Masked = tf.concat([Mask*InputImages/Thresholds, Mask], axis=3)        #   L   x L   x  2 
Masked = tf.concat([Mask*InputImages, Mask], axis=3)        #   L   x L   x  2 
C1 = lrelu(conv2d(Masked, W1) + B1)                                    #   L/2 x L/2 x 16
C2 = lrelu(conv2d(C1, W2) + B2)                                        #   L/4 x L/4 x 32
C3 = lrelu(deconv2d(C2, W3, [NbBatches, SIDE//2, SIDE//2, 32]) + B3)   #   L/2 x L/2 x 32
C3_= tf.concat([C1, C3], axis=3)                                       #   L/2 x L/2 x 48 
C4 = lrelu(deconv2d(C3_, W4, [NbBatches, SIDE, SIDE, 16]) + B4)        #   L   x L   x 16
C4_= tf.concat([Masked, C4], axis=3)                                   #   L   x L   x 18  
C5 = deconv2d(C4_, W5, [NbBatches, SIDE, SIDE, 2], stride=1) + B5      #   L   x L   x  2

PredictedMeans = C5[:,:,:,:CHANNELS]
PredictedVariances = tf.square(tf.nn.softplus(2.0 + C5[:,:,:,CHANNELS:]) + EPSILON)

PredictedNegLogProbs = 0.5*(np.log(2*np.pi) + tf.log(PredictedVariances*256*256)) + (tf.square(InputImages - PredictedMeans) / (2*PredictedVariances))
PredictedNegLogProbs = -tf.log((1-RHO)*tf.exp(-PredictedNegLogProbs) + RHO*(1.0/256))

MostSureIndex = tf.argmin(tf.reshape(PredictedVariances+BIG_M*Mask, [-1, SIDE*SIDE*CHANNELS]), axis=1)
MostSureRow = MostSureIndex // (SIDE*CHANNELS) 
MostSureCol = tf.mod(MostSureIndex, SIDE*CHANNELS) // CHANNELS
MostSureChannel = tf.mod(MostSureIndex, CHANNELS)
X = tf.cast(tf.range(NbBatches), tf.int64)
MostSureIndices = tf.stack([X, MostSureRow, MostSureCol, MostSureChannel], axis=1)
GatheredImages = tf.gather_nd(InputImages, MostSureIndices)
GatheredMeans = tf.gather_nd(PredictedMeans, MostSureIndices)
GatheredVariances = tf.gather_nd(PredictedVariances, MostSureIndices)

PredictedNegLogProbsMostSure = 0.5*(np.log(2*np.pi)+tf.log(GatheredVariances*256*256)) + (tf.square(GatheredImages - GatheredMeans) / (2*GatheredVariances))
PredictedNegLogProbsMostSure = -tf.log((1-RHO)*tf.exp(-PredictedNegLogProbsMostSure) + RHO*(1.0/256))

# 2.3. Gradient Descent acts to minimize a differentiable loss:
SCUBALoss = GAMMA * tf.reduce_mean(tf.reduce_sum(PredictedNegLogProbs * (1.0 - Mask), axis=[1, 2, 3])/(0.001 + tf.reduce_sum(1.0-Mask, axis=[1, 2, 3]))) + \
                    tf.reduce_mean(PredictedNegLogProbsMostSure)

# 2.4. GRADIENT DESCENT STEP (note the change to ADAM):
Update = tf.train.RMSPropOptimizer(LearningRate, decay=0.99).minimize(SCUBALoss)

###############################################################################
#                                 3. RUN GRAPH                                #
###############################################################################

saver = tf.train.Saver()
#SAVE_PATH = 'checkpoints/scuba_fashion.ckpt'
SAVE_PATH = 'checkpoints/scuba_mnist_corrected.ckpt'

with tf.Session() as sess:
    if glob.glob(SAVE_PATH+'*'):
        print('Loading Model...\n')
        saver.restore(sess, SAVE_PATH)
    else:
        print('Initializing Model from scratch...\n')
        sess.run(tf.global_variables_initializer())
    
    rates = [] 
    for i in range(TRAIN_TIME):
        batch_inputs = get_batch() 
        sess.run(Update, feed_dict={TrueInputs:batch_inputs, LearningRate:LEARNING_RATE})

        if i%10: continue
        batch_inputs = get_batch(train=True) 
        scuba_loss = sess.run(tf.reduce_mean(PredictedNegLogProbsMostSure), feed_dict={TrueInputs:batch_inputs})
        compression_rate = np.log(256) / scuba_loss 
        print(' '*180 + '\033[1A')
        print('\033[1;34;mstep %05d, scuba loss %.3f, compression rate %.2f \t%s' % (i, scuba_loss/np.log(2), compression_rate, '-'*int(compression_rate*20)))
        print('\033[2A')
        rates.append(scuba_loss)

        if i%100: continue
        print('\nSaving Model...')
        saver.save(sess, SAVE_PATH)
        pess = np.log(256)/np.mean(rates)
        print('\t pessimistic compression_rate: \033[1;33;m %.3f \t\t%s \033[1;34;m' % (pess, '-'*int(pess*20)))
        rates = []

'''
RMSProp LR=0.0001, decay=0.99:
step 40710, scuba loss 1.442, compression rate 3.85 ----------------------------------------------------------------------------
step 40720, scuba loss 1.997, compression rate 2.78 -------------------------------------------------------
step 40730, scuba loss 1.435, compression rate 3.87 -----------------------------------------------------------------------------
step 40740, scuba loss 2.152, compression rate 2.58 ---------------------------------------------------
step 40750, scuba loss 1.403, compression rate 3.95 -------------------------------------------------------------------------------
step 40760, scuba loss 1.671, compression rate 3.32 ------------------------------------------------------------------
step 40770, scuba loss 1.569, compression rate 3.53 ----------------------------------------------------------------------
step 40780, scuba loss 1.943, compression rate 2.85 ---------------------------------------------------------
step 40790, scuba loss 2.156, compression rate 2.57 ---------------------------------------------------
'''
