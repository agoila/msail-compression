''' author: samtenka
    changed: 2017-10-10
    created: 2017-10-10
    descr: 
    usage: Run `python convolutional.py`.

1862/305/333
'''

import scipy.misc
import numpy as np
import os, glob

###############################################################################
#                            0. READ DATASET                                  #
###############################################################################

# 0.0. MNIST is a classic image-classification dataset.  Its images are 28x28 
#      grayscale photographs of handwritten digits (0 through 9).  Note that
#      we load the labels in one-hot form.  This makes defining a loss function
#      easier: 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('FASHION_data')

def get_batch(size, train=True):
    ''' Return `inputs` of shape (size, 28, 28, 3)
        randomly sampled from the full data. 
    '''
    inputs, _ = (mnist.train if train else mnist.test).next_batch(size)
    #return np.concatenate([np.reshape(inputs, [size, 28, 28, 1])]*3, axis=3)
    return np.reshape(inputs, [size, 28, 28, 1])

#plt.imshow(get_batch(1)[0])
#plt.savefig('hi.png')
#plt.savefig('hi.bmp')

M = 1000
img = get_batch(M)[:,:,:,0]

for m in range(M):
    if m%10==0: print(m)
    scipy.misc.imsave('mnist_%d.png' %m, img[m])
    scipy.misc.imsave('mnist_%d.bmp' %m, img[m])

total_png = sum(os.path.getsize(f) for 
                f in glob.glob('*.png')) 
total_bmp = sum(os.path.getsize(f) for 
                f in glob.glob('*.bmp')) 
print('png: %.2f' % (total_png/float(M)))
print('bmp: %.2f' % (total_bmp/float(M)))

os.system('rm mnist_*.png mnist_*.bmp')
