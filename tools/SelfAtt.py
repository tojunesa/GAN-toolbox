import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow import identity

class SelfAttention():
    def __init__(sigma=0.2):
        super(Conv_SN, self).__init__()
        sefl.sigma = sigma

    def build(self, input_shapes):
        _, self.h, self.w, self.channels_num = input_shapes
        
    def call(self, fm):
        #Self-attention block
        fm_q = Conv2D(self.channels_num//8, 1, 1)(fm)
        fm_q = tf.reshape(fm_q, [-1, self.h*self.w, self.channels_num//8])
    
        fm_k = Conv2D(num_channels//8, 1, 1)(fm)
        fm_k = tf.reshape(fm_k, [-1, self.h*self.w, self.channels_num//8])
    
        attn = tf.linalg.matmul(fm_q, fm_k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
    
        fm_v = Conv2D(self.channels_num//2, 1, 1)(fm)
        fm_v = tf.reshape(fm_v, [-1, self.h*self.w, self.channels_num//2])
    
        sa_output = tf.linalg.matmul(attn, fm_v)
        sa_output = tf.reshape(output, [-1, self.h*self.w, self.channels_num//2])
        sa_output = Conv2D(num_channels, 1, 1)(output)

        return fm+sigma*output
