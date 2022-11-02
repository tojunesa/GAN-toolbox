import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, LeakyReLU
from tensorflow import identity

def ResBlock(fm):
    """
    This function defines a residual block. Two convolutional layers with 3*3 kernel size are used to process the input feature map. One convolutional layer with 1*1 kernel size is used to process the copy of the input feature map. 
    Input: 
        fm: feature map (data_format='NHWC)
        
    Output:
        fm: feature map (data_format='NHWC)
    """
    fm_copy = identity(fm)
    channel_num = fm.shape[-1]
    
    fm = Conv2D(channel_num, kernel_size=3, strides=1, padding='same')(fm)
    fm = LeakyReLU(alpha = 0.2)(fm)
    fm = Conv2D(channel_num, kernel_size=3, strides=1, padding='same')(fm)
    
    fm_copy = Conv2D(channel_num, kernel_size=1, strides=1, padding='same')(fm_copy)
    fm = Add()([fm, fm_copy])
    
    fm = LeakyReLU(alpha = 0.2)(fm)
    
    return fm