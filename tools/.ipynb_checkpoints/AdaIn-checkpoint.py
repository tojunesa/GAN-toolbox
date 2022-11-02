import tensorflow as tf

class AdaIN(tf.keras.layers.Layer):
    #Custom keras layer with adpative instance normalization
    def __init__(self):
        super(AdaIN, self).__init__()
        
    def call(self, in_fm, style):
        #feature map shape: [N, H, W, C]
        batch_size = tf.shape(in_fm)[0]
        ch_num = tf.shape(in_fm)[-1]
        style_mean = tf.reshape(style[:,0:ch_num], [batch_size,1,1,ch_num])
        style_std = tf.reshape(style[:,ch_num:], [batch_size,1,1,ch_num])
        
        fm_mean, fm_std = tf.nn.moments(in_fm, axes=[1,2], keepdims=True)
        normalized_fm = tf.nn.batch_normalization(in_fm, fm_mean, fm_std, style_mean, style_std, variance_epsilon=1e-5)
        return normalized_fm