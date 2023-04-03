import tensorflow as tf
import numpy as np
from glob import glob
import sys
sys.path.append('../')

def divergence_x(x):
    #return(x[:,1:]-x[:,:-1])
    return(x[:,1:-1]-x[:,:-2])

def divergence_y(x):
    #return(x[:,:,1:]-x[:,:,:-1])
    return(x[:,:,1:-1]-x[:,:,:-2])

def divergence_x2(x):
    return(x[:,2:]-x[:,1:-1])

def divergence_y2(x):
    return(x[:,:,2:]-x[:,:,1:-1])





def classifier(inputs, option=1, num_classes=2,kernel_size=3,pool_size=3,CROP=256):
    
    
    x = tf.keras.layers.Conv2D(32, kernel_size, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [16,32,64,128]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(pool_size, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(1024, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    return x


def make_embedding(input_shape,kernel_size=3,pool_size=3,CROP=256):
    
    
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [16,32,64,128]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(pool_size, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(32*32, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    embedding =  tf.keras.Model(inputs,x)
    embedding.load_weights('/home/joel/master_thesis/23_jan/border/checkpoints/embedding')
    
    for layer in embedding.layers:
        layer.trainable = False
    
    return embedding


"""
Creating the model
"""
def conv_block(x, n_filt, size_conv=(5,5), n_conv=3):
    """
    Applies n_conv convolutions to the input with specified size and number of filters.
    """
    for c in range(n_conv):
        x = tf.keras.layers.Conv2D(n_filt, size_conv, padding="same", activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    return x

def u_encoder(x, n_filt):
    """
    Applies conv_block and returnes max pooled and skip-connection.
    """
    x = conv_block(x, n_filt)
    return tf.keras.layers.MaxPool2D()(x), x

def u_decoder(pooled, skipped, n_filt):
    """
    Upsamples pooled and concats with skiped.
    """
    upsampled = tf.keras.layers.Convolution2DTranspose(n_filt, (2,2), strides=(2,2), padding='same')(pooled)
    return conv_block(tf.keras.layers.concatenate([upsampled, skipped]), n_filt)
    
    
def make_unet(inp, depth=3, output_channels=1):
    skipped = []
    p = inp
    for _ in range(depth):
        p, s = u_encoder(p, 2**(1+_))
        #p, s = u_encoder(p, 2*(1+_))
        skipped.append(s)
    p = conv_block(p, 2**(2+depth))
    for _ in reversed(range(depth)):
        p = u_decoder(p, skipped[_], 2**(2+_))  
        #p = u_decoder(p, skipped[_], (2**3)*(1+_))  
    p = tf.keras.layers.Conv2D(output_channels, (1,1), activation='sigmoid')(p)
    return p

def make_boundary_detector(inp):
    """
    Takes network Input layer 
    """
    outp = make_unet(inp)
    return tf.keras.models.Model(inp, outp)

def get_boundary_detector(CROP):
    border = make_boundary_detector(tf.keras.layers.Input(shape=(CROP,CROP, 1)))
    border.load_weights(f"/home/joel/master_thesis/23_jan/border/checkpoints/borders_gaussian")

    for layer in border.layers:
        layer.trainable = False
        
    return border
    



class function_type:
    def splines(y,num_classes,order):


        b_initial = tf.keras.layers.Dense(1,activation='linear')(y)
        b_initial = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(b_initial)

        a = tf.keras.layers.Dense(num_classes*order,activation = 'linear')(y)

        a = tf.keras.layers.Reshape((num_classes,order))(a)

        b = tf.keras.layers.Lambda(lambda z:tf.multiply(np.asarray(1/num_classes,dtype=np.float32),tf.ones_like(z)))(a)
        b = tf.pow(b,np.arange(1,order+1))
        b = tf.keras.layers.multiply([a,b])

        m = tf.linalg.LinearOperatorLowerTriangular(tf.ones(tf.shape(a)+(0,0,num_classes-1))).to_dense()
        b = tf.keras.layers.multiply([tf.transpose(b,perm=(0,2,1)),m])
        b = tf.keras.layers.Lambda(lambda z:tf.math.reduce_sum(z,axis=-1))(b)
        b0 = tf.keras.layers.Lambda(lambda z:tf.expand_dims(tf.zeros_like(z[:,0]),axis=-1))(b)
        b = tf.keras.layers.Concatenate(axis=1)([b0,b])
        b = tf.keras.layers.Lambda(lambda z:z[:,:-1])(b)
        b = tf.keras.layers.add((b,b_initial))
        b = tf.expand_dims(b,axis=-1)

        return a,b

    def decreasing(y,num_classes,order):
        b_initial = tf.keras.layers.Dense(1,activation='linear')(y)
        b_initial = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(b_initial)
        a = tf.keras.layers.Dense(num_classes*order,activation = 'linear')(y)
        a = tf.keras.layers.Reshape((num_classes,order))(a)
        a = tf.keras.layers.Lambda(lambda z:-tf.math.pow(z,2))(a)

        b = tf.keras.layers.Lambda(lambda z:tf.multiply(np.asarray(1/num_classes,dtype=np.float32),tf.ones_like(z)))(a)
        b = tf.pow(b,np.arange(1,order+1))
        b = tf.keras.layers.multiply([a,b])

        m = tf.linalg.LinearOperatorLowerTriangular(tf.ones(tf.shape(a)+(0,0,num_classes-1))).to_dense()
        b = tf.keras.layers.multiply([tf.transpose(b,perm=(0,2,1)),m])
        b = tf.keras.layers.Lambda(lambda z:tf.math.reduce_sum(z,axis=-1))(b)
        b0 = tf.keras.layers.Lambda(lambda z:tf.expand_dims(tf.zeros_like(z[:,0]),axis=-1))(b)
        b = tf.keras.layers.Concatenate(axis=1)([b0,b])
        b = tf.keras.layers.Lambda(lambda z:z[:,:-1])(b)
        b = tf.keras.layers.add((b,b_initial))
        b = tf.expand_dims(b,axis=-1)

        return a,b

    def flux(y,num_classes,order):
        b_initial = tf.keras.layers.Dense(1,activation='linear')(y)
        b_initial = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(b_initial)

        a = tf.keras.layers.Dense(num_classes*order,activation = 'linear')(y)
        a = tf.keras.layers.Reshape((num_classes,order))(a)
        s0 = tf.keras.layers.Dense(1,activation='sigmoid')(y)

        b = tf.keras.layers.Lambda(lambda z:tf.multiply(np.asarray(1/num_classes,dtype=np.float32),tf.ones_like(z)))(a)
        b = tf.keras.layers.multiply((b,s0))
        b = tf.pow(b,np.arange(1,order+1))
        b = tf.keras.layers.multiply([a,b])

        m = tf.linalg.LinearOperatorLowerTriangular(tf.ones(tf.shape(a)+(0,0,num_classes-1))).to_dense()
        b = tf.keras.layers.multiply([tf.transpose(b,perm=(0,2,1)),m])
        b = tf.keras.layers.Lambda(lambda z:tf.math.reduce_sum(z,axis=-1))(b)
        b0 = tf.keras.layers.Lambda(lambda z:tf.expand_dims(tf.zeros_like(z[:,0]),axis=-1))(b)
        b = tf.keras.layers.Concatenate(axis=1)([b0,b])
        b = tf.keras.layers.Lambda(lambda z:z[:,:-1])(b)
        b = tf.keras.layers.add((b,b_initial))
        b = tf.expand_dims(b,axis=-1)


        minimum = tf.ones_like(a)/(num_classes)
        fun = tf.keras.layers.multiply((s0,minimum,a))
        fun = tf.keras.layers.add((fun,b))
        coeffs = 2*(tf.cumsum(tf.ones_like(a)/(num_classes),axis=1))
        coeffs = tf.keras.layers.multiply((coeffs,s0,a))
        fun = tf.keras.layers.add((coeffs,fun))
        minimum = tf.keras.layers.Lambda(lambda z:-tf.reduce_min(z[...,0],axis=-1))(fun)
        minimum_neg = tf.keras.layers.Lambda(lambda z: tf.cast(tf.less_equal(-z,0),dtype=tf.float32))(minimum)
        minimum = tf.keras.layers.multiply((minimum,minimum_neg))

        b = tf.keras.layers.add((b,minimum))

        a_pos,b_pos = a,b

        b_middle = tf.keras.layers.Dense(1,activation='linear')(y)
        b_middle = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(b_middle)


        a = tf.keras.layers.Dense(num_classes*order,activation = 'linear')(y)
        a = tf.keras.layers.Reshape((num_classes,order))(a)
        s1 = tf.keras.layers.Lambda(lambda z: tf.ones_like(z)-z)(s0)

        b = tf.keras.layers.Lambda(lambda z:tf.multiply(np.asarray(1/num_classes,dtype=np.float32),tf.ones_like(z)))(a)
        b = tf.keras.layers.multiply((b,s1))
        b = tf.pow(b,np.arange(1,order+1))
        b = tf.keras.layers.multiply([a,b])

        m = tf.linalg.LinearOperatorLowerTriangular(tf.ones(tf.shape(a)+(0,0,num_classes-1))).to_dense()
        b = tf.keras.layers.multiply([tf.transpose(b,perm=(0,2,1)),m])
        b = tf.keras.layers.Lambda(lambda z:tf.math.reduce_sum(z,axis=-1))(b)
        b0 = tf.keras.layers.Lambda(lambda z:tf.expand_dims(tf.zeros_like(z[:,0]),axis=-1))(b)
        b = tf.keras.layers.Concatenate(axis=1)([b0,b])
        b = tf.keras.layers.Lambda(lambda z:z[:,:-1])(b)
        b = tf.keras.layers.add((b,b_middle))
        b = tf.expand_dims(b,axis=-1)


        maximum = tf.ones_like(a)/(num_classes)
        fun = tf.keras.layers.multiply((s1,maximum,a))
        fun = tf.keras.layers.add((fun,b))
        coeffs = 2*(tf.cumsum(tf.ones_like(a)/(num_classes),axis=1))
        coeffs = tf.keras.layers.multiply((coeffs,s1))
        coeffs = tf.keras.layers.add((coeffs,s0))
        coeffs = tf.keras.layers.multiply((a,coeffs))
        fun = tf.keras.layers.add((coeffs,fun))
        maximum = tf.keras.layers.Lambda(lambda z:-tf.reduce_max(z[...,0],axis=-1))(fun)
        maximum_neg = tf.keras.layers.Lambda(lambda z: tf.cast(tf.greater_equal(-z,0),dtype=tf.float32))(maximum)
        maximum = tf.keras.layers.multiply((maximum,maximum_neg))
        maximum = tf.keras.layers.Lambda(lambda z:tf.expand_dims(z,axis=-1))(maximum)

        b = tf.keras.layers.add((b,maximum))

        a_neg,b_neg = a,b

        a = tf.keras.layers.Concatenate(axis=1)((a_pos,a_neg))
        b = tf.keras.layers.Concatenate(axis=1)((b_pos,b_neg))

        return a,b





def get_model(arch,it_lim,image_size,typ='gaussian',num_classes=1,CROP = 256,order = 1,gamma=1,with_embedding = True, boundary_detector=True):


    input_shape = image_size + (1,)
    
    
    inputs = tf.keras.Input(shape=input_shape,name='input')
    
    if with_embedding:
        x = make_embedding(input_shape)(inputs)
    else:
        x = classifier(inputs, num_classes=num_classes,CROP=CROP)
    
    y = tf.keras.layers.Flatten(name='y')(x)
    
    a,b = getattr(function_type,arch)(y,num_classes,order)
    
    if arch == 'flux':
        num_classes = 2*num_classes
    

    partition_low = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[:-1])
    partition_low = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_low,0),0),0)
    partition_low = tf.cast(partition_low,tf.float32)
    partition_up = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[1:])
    partition_up = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_up,0),0),0)
    partition_up = tf.cast(partition_up,tf.float32)


    


    ct = tf.keras.layers.Concatenate(name='coeff_spline')((b,a))
    ct = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct)

    outputs = inputs
    
    if boundary_detector:
        boundary = get_boundary_detector(CROP)
 

    for num_it in range(it_lim):
        
        zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(inputs)[:,1],axis=-3)
        
        if boundary_detector:
            deltaS = boundary(outputs)
            deltaE = boundary(outputs)
            
        else:
 
            difS = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=1))(outputs)
            difE = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=2))(outputs)
            deltaS = tf.keras.layers.Concatenate(axis=1)([difS,zeros_x])
            deltaE = tf.keras.layers.Concatenate(axis=2)([difE,zeros_y])

        
        dS_n = tf.keras.layers.Lambda(lambda z: tf.math.sqrt(z))(deltaS)
        dE_n = tf.keras.layers.Lambda(lambda z: tf.math.sqrt(z))(deltaE)

        dS2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2),name=f'dx_{num_it}')(dS_n)
        dE2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dE_n)

        ineq1_S = tf.greater_equal(dS2, partition_low)
        ineq2_S = tf.less_equal(dS2,partition_up)
        ineq1_E = tf.greater_equal(dE2, partition_low)
        ineq2_E = tf.less_equal(dE2,partition_up)

        interval_S = tf.cast(tf.math.logical_and(ineq1_S,ineq2_S),tf.float32)
        interval_E = tf.cast(tf.math.logical_and(ineq1_E,ineq2_E),tf.float32)

        power_norm_S = tf.pow(dS2,tf.constant(np.asarray(np.arange(1,order+1),dtype='float32')))
        power_norm_S = tf.keras.layers.Lambda(lambda z:tf.math.mod(z,1/num_classes))(power_norm_S)
        cte_S = tf.ones_like(inputs)
        power_norm_S = tf.keras.layers.Concatenate(axis=-1)((cte_S,power_norm_S))
        power_norm_S = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-2))(power_norm_S)
        power_norm_E = tf.pow(dE2,tf.constant(np.asarray(np.arange(1,order+1),dtype='float32')))
        power_norm_E = tf.keras.layers.Lambda(lambda z:tf.math.mod(z,1/num_classes))(power_norm_E)
        cte_E = tf.ones_like(inputs)
        power_norm_E = tf.keras.layers.Concatenate(axis=-1)((cte_E,power_norm_E))
        power_norm_E = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-2))(power_norm_E)

        
        spline_S = tf.keras.layers.multiply([ct,power_norm_S])
        spline_S = tf.keras.layers.Lambda(lambda z:tf.math.reduce_sum(z,axis=-1))(spline_S)
        spline_S = tf.keras.layers.multiply([spline_S,interval_S])
        spline_E = tf.keras.layers.multiply([ct,power_norm_E])
        spline_E = tf.keras.layers.Lambda(lambda z:tf.math.reduce_sum(z,axis=-1))(spline_E)
        spline_E = tf.keras.layers.multiply([spline_E,interval_E])


        gS = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z,axis=-1),name=f'gS_{num_it}')(spline_S)
        gS = tf.expand_dims(gS,axis=-1)
        gE = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z,axis=-1),name=f'gE_{num_it}')(spline_E)
        gE = tf.expand_dims(gE,axis=-1)
        
        difS = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=1))(outputs)
        difE = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=2))(outputs)
        deltaS = tf.keras.layers.Concatenate(axis=1)([difS,zeros_x])
        deltaE = tf.keras.layers.Concatenate(axis=2)([difE,zeros_y])
        E = tf.keras.layers.multiply((gE,deltaE))
        S = tf.keras.layers.multiply((gS,deltaS))

        NS = S
        EW = E
        NS = tf.keras.layers.Concatenate(axis=1)([zeros_x,NS])
        EW = tf.keras.layers.Concatenate(axis=2)([zeros_y,EW])
        NS = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=1))(NS)
        EW = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=2))(EW)

        mult = tf.keras.layers.Lambda(lambda z: tf.multiply(tf.cast(gamma,dtype=tf.float32),z))(tf.ones_like(NS))
        NS_mod = NS
        EW_mod = EW

        adding = tf.keras.layers.add([NS,EW])
        adding = tf.keras.layers.multiply((mult,adding))

        outputs = tf.keras.layers.add([outputs,adding])
        
    
        
    return tf.keras.models.Model(inputs, outputs)



