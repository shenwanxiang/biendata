import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import MaxPool2D, GlobalMaxPool2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Concatenate,Flatten, Dense, Dropout


def count_trainable_params(model):
    p = 0
    for layer in model.layers:
        if len(layer.trainable_variables) == 0:
            continue
        else:
            for variables in layer.trainable_variables:
                a = variables.shape.as_list()
                if len(a) == 1:
                    p += a[0]
                else:
                    p += a[0]*a[1]
    return p



def Inception(inputs, units = 8, strides = 1):
    """
    naive google inception block
    """
    x1 = Conv2D(units, 5, padding='same', activation = 'relu', strides = strides)(inputs)
    x2 = Conv2D(units, 3, padding='same', activation = 'relu', strides = strides)(inputs)
    x3 = Conv2D(units, 1, padding='same', activation = 'relu', strides = strides)(inputs)
    outputs = Concatenate()([x1, x2, x3])    
    return outputs



def MolMapNet(input_shape,  
                n_outputs = 1, 
                conv1_kernel_size = 13,
                dense_layers = [128, 32], 
                dense_avf = 'relu', 
                last_avf = None):
    
    
    """
    parameters
    ----------------------
    molmap_shape: w, h, c
    n_outputs: output units
    dense_layers: list, how many dense layers and units
    dense_avf: activation function for dense layers
    last_avf: activation function for last layer
    """
    
    assert len(input_shape) == 3
    inputs = Input(input_shape)
    
    conv1 = Conv2D(48,  conv1_kernel_size, padding = 'same', activation='relu', strides = 1)(inputs)
    
    conv1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(conv1) #p1
    
    incept1 = Inception(conv1, strides = 1, units = 32)
    
    incept1 = MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(incept1) #p2
    
    incept2 = Inception(incept1, strides = 1, units = 64)
    
    #flatten
    x = GlobalMaxPool2D()(incept2)
    
    ## dense layer
    for units in dense_layers:
        x = Dense(units, activation = dense_avf)(x)
        
    #last layer
    outputs = Dense(n_outputs,activation=last_avf)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

