import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow import identity

class Conv_SN(tf.keras.layers.Layer, power_iter=1):
    """
    Convolutional layer with spectrum normalization (derive from keras layer class)
    """
    def __init__(self, channels, kernel_size, strides):
        """
        Initialize convolutional kernels' parameters, padding is default to be "SAME".
        Inputs:
            channels: number of filters (output channel size)
            kernel_size: kernel size
            strides: strides
            power_iter: number of iterations used in the power iteration method to calculate the l2 norm of the weight matrix
        """
        super(Conv_SN, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.power_iter = power_iter
    
    def build(self, input_shapes):
        """
        Initialize weights, bias, and a tool vector u when an instance is called.
        Inputs:
            input_shapes: automatically fetched when the instance is called to process an input feature map
        """
        self.shape = [self.kernel_size, self.kernel_size, input_shapes[-1], self.channels]
        W_init = tf.random.normal(self.shape, mean=0.0, stddev=tf.sqrt(2 / (input_shapes[-1]+self.channels)), dtype=tf.dtypes.float32)
        self.W = tf.Variable(initial_value=W_init ,name="kernels", shape=self.shape)
        b_init = tf.zeros([self.channels])
        self.b = tf.Variable(initial_value=b_init, name="bias", shape=[self.channels])
        u_init = tf.random.normal([1,self.channels], mean=0.0, stddev=1, dtype=tf.dtypes.float32)
        self.u = tf.Variable(initial_value=u_init, trainable=False, name="u", shape=[1, self.channels])

    def call(self, fm):
        """
        Apply spectral normalization on the weight matrix.
        Then apply the normalized kernels on the input feature map.
        Inputs:
            fm: feature map (data_format="NHWC")
        Outputs: 
            outputs: output feature map
        """
        w = tf.reshape(self.W, [-1, self.shape[-1]])
        for i in range(power_iter):
            v_hat = tf.math.l2_normalize(tf.matmul(self.u, tf.transpose(w)))
            u_hat = tf.math.l2_normalize(tf.matmul(v_hat, w))
        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        W_SN = self.W/sigma
        self.W.assign(W_SN)
        self.u.assign(u_hat)
        outputs = tf.nn.conv2d(fm, self.W, strides=[1,self.strides,self.strides,1], padding="SAME", data_format="NHWC", dilations=1)+self.b
        return outputs
    
class Dense_SN(tf.keras.layers.Layer):
    """
    Dense layer with spectrum normalization (derive from keras layer class)
    """
    def __init__(self, neuron_num, power_iter = 1):
        """
        Initialize dense layer parameter: the number of neurons.
        Inputs:
            neuron_num: number of neurons
            power_iter: number of iterations used in the power iteration method to calculate the l2 norm of the weight matrix
        """
        super(Dense_SN, self).__init__()
        self.neuron_num = neuron_num
    
    def build(self, input_shapes):
        """
        Initialize weights, bias, and a tool vector u when an instance is called.
        Inputs:
            input_shapes: automatically fetched when the instance is called to process an input feature map
        """
        self.shape = [input_shapes[-1], self.neuron_num]
        W_init = tf.random.normal(self.shape, mean=0.0, stddev=tf.sqrt(2 / (input_shapes[-1]+self.neuron_num)), dtype=tf.dtypes.float32)
        self.W = tf.Variable(initial_value=W_init ,name="kernels", shape=self.shape)
        b_init = tf.zeros([self.neuron_num])
        self.b = tf.Variable(initial_value=b_init, name="bias", shape=[self.neuron_num])
        u_init = tf.random.normal([1,self.neuron_num], mean=0.0, stddev=1, dtype=tf.dtypes.float32)
        self.u = tf.Variable(initial_value=u_init, trainable=False, name="u", shape=[1, self.neuron_num])
    
    def call(self, inputs):
        """
        Apply spectral normalization on the weight matrix.
        Then apply the normalized weights on the input feature map.
        Inputs:
            fm: feature map
        Outputs: 
            outputs: output feature map
        """
        w = tf.reshape(self.W, [-1, self.shape[-1]])
        for i in range(power_iter):
            v_hat = tf.math.l2_normalize(tf.matmul(self.u, tf.transpose(w)))
            u_hat = tf.math.l2_normalize(tf.matmul(v_hat, w))
        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        W_SN = self.W/sigma
        self.W.assign(W_SN)
        self.u.assign(u_hat)
        outputs = tf.matmul(inputs, self.W)+self.b
        return outputs
    
    
    