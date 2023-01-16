import tensorflow as tf
import numpy as np
from glob import glob

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
        skipped.append(s)
    p = conv_block(p, 2**(2+depth))
    for _ in reversed(range(depth)):
        p = u_decoder(p, skipped[_], 2**(2+_))  
    p = tf.keras.layers.Conv2D(output_channels, (1,1), activation='sigmoid')(p)
    return p

def get_unet(inp):
    """
    Takes network Input layer 
    """
    outp = make_unet(inp)
    return tf.keras.models.Model(inp, outp)



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


def lambdas(input_shape,it_lim = 1, option=1, num_classes=2,kernel_size=3,pool_size=3,CROP=256):
    inputs = tf.keras.Input(shape=input_shape)
    x = classifier(inputs, option=option, num_classes=num_classes,kernel_size=kernel_size,
                   pool_size=pool_size,CROP=CROP)

    x = tf.keras.layers.Dropout(0.5)(x)
    lambda_value = tf.keras.layers.Dense(1, activation='linear', name=f'lambdas',kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=1, seed=None),bias_initializer = tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=1, seed=None))(x)
    lambda_value = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,-2))(lambda_value)

    outputs = inputs
    
    
    for num_it in range(it_lim):

        dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(outputs)
        dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(outputs)
        dz = tf.keras.layers.Lambda(lambda z: z[:,1:-1,2:] - z[:,1:-1,1:-1])(outputs)
        dw = tf.keras.layers.Lambda(lambda z: z[:,1:-1,:-2] - z[:,1:-1,1:-1])(outputs)

        dx2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx)
        dy2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy)
        dz2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dz)
        dw2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dw)

        norm = tf.keras.layers.add([dx2,dy2,dz2,dw2])
        dx2 = tf.keras.layers.multiply([lambda_value,dx2])
        dy2 = tf.keras.layers.multiply([lambda_value,dy2])
        dz2 = tf.keras.layers.multiply([lambda_value,dz2])
        dw2 = tf.keras.layers.multiply([lambda_value,dw2])

        if option == 1:
            coeff_x = tf.keras.layers.Lambda(lambda z: tf.math.exp(-z),name=f'coeffx_{num_it}')(dx2)
            coeff_y = tf.keras.layers.Lambda(lambda z: tf.math.exp(-z),name=f'coeffy_{num_it}')(dy2)
            coeff_z = tf.keras.layers.Lambda(lambda z: tf.math.exp(-z),name=f'coeffz_{num_it}')(dz2)
            coeff_w = tf.keras.layers.Lambda(lambda z: tf.math.exp(-z),name=f'coeffw_{num_it}')(dw2)
        elif option == 2:
            coeff_x = tf.keras.layers.Lambda(lambda z: 1./(1. + z),name=f'coeffx_{num_it}')(dx2)
            coeff_y = tf.keras.layers.Lambda(lambda z: 1./(1. + z),name=f'coeffy_{num_it}')(dy2)
            coeff_z = tf.keras.layers.Lambda(lambda z: 1./(1. + z),name=f'coeffz_{num_it}')(dz2)
            coeff_w = tf.keras.layers.Lambda(lambda z: 1./(1. + z),name=f'coeffw_{num_it}')(dw2)
            
            

        outputs_x = tf.keras.layers.multiply([coeff_x,dx])
        outputs_y = tf.keras.layers.multiply([coeff_y,dy])
        outputs_z = tf.keras.layers.multiply([coeff_z,dz])
        outputs_w = tf.keras.layers.multiply([coeff_w,dw])
        
        outputs_it = tf.keras.layers.add([outputs_x,outputs_y,outputs_z,outputs_w])
        zeros_y = tf.expand_dims(tf.zeros_like(outputs_it)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-3)
        pad_y = tf.keras.layers.Concatenate(axis=-2)([zeros_y,outputs_it,zeros_y])
        outputs_it = tf.keras.layers.Concatenate(axis=1)([zeros_x,pad_y,zeros_x])
        outputs = tf.keras.layers.add([0.1*outputs_it,outputs])
    
    return tf.keras.models.Model(inputs, outputs)

def border_lambdas(input_shape,border_classifier,it_lim = 1, option=1, num_classes=2,kernel_size=3,pool_size=3,CROP=256):
    inputs = tf.keras.Input(shape=input_shape)
    x = classifier(inputs, option=option, num_classes=num_classes,kernel_size=kernel_size,
                   pool_size=pool_size,CROP=CROP)

    x = tf.keras.layers.Dropout(0.5)(x)
    lambda_value_1 = tf.keras.layers.Dense(1, activation='linear', name=f'lambdas_1',kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=1, seed=None),bias_initializer = tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=1, seed=None))(x)
    lambda_value_2 = tf.keras.layers.Dense(1, activation='linear', name=f'lambdas_2',kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=1, seed=None),bias_initializer = tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=1, seed=None))(x)
    lambda_value_1 = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,-2))(lambda_value_1)
    lambda_value_2 = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,-2))(lambda_value_2)


    lam_1 = tf.expand_dims(lambda_value_1,axis=-1)
    lam_1 = tf.expand_dims(lam_1,axis=-1)
    lam_2 = tf.expand_dims(lambda_value_2,axis=-1)
    lam_2 = tf.expand_dims(lam_2,axis=-1)

    class_1 = tf.cast(border_classifier(inputs)>0.5,'float32')
    class_2 = tf.cast(border_classifier(inputs)<=0.5,'float32')
    class_1 = tf.keras.layers.multiply([lam_1,class_1])
    class_2 = tf.keras.layers.multiply([lam_2,class_2])

    lambda_value = tf.keras.layers.add([class_1,class_2],name='lambda_val')

    outputs = inputs


    for num_it in range(it_lim):

        dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(outputs)
        dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(outputs)
        dz = tf.keras.layers.Lambda(lambda z: z[:,1:-1,2:] - z[:,1:-1,1:-1])(outputs)
        dw = tf.keras.layers.Lambda(lambda z: z[:,1:-1,:-2] - z[:,1:-1,1:-1])(outputs)

        dx2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx)
        dy2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy)
        dz2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dz)
        dw2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dw)

        norm = tf.keras.layers.add([dx2,dy2,dz2,dw2])
        dx2 = tf.keras.layers.multiply([lambda_value[:,1:-1,1:-1],dx2])
        dy2 = tf.keras.layers.multiply([lambda_value[:,1:-1,1:-1],dy2])
        dz2 = tf.keras.layers.multiply([lambda_value[:,1:-1,1:-1],dz2])
        dw2 = tf.keras.layers.multiply([lambda_value[:,1:-1,1:-1],dw2])

        if option == 1:
            coeff_x = tf.keras.layers.Lambda(lambda z: tf.math.exp(-z),name=f'coeffx_{num_it}')(dx2)
            coeff_y = tf.keras.layers.Lambda(lambda z: tf.math.exp(-z),name=f'coeffy_{num_it}')(dy2)
            coeff_z = tf.keras.layers.Lambda(lambda z: tf.math.exp(-z),name=f'coeffz_{num_it}')(dz2)
            coeff_w = tf.keras.layers.Lambda(lambda z: tf.math.exp(-z),name=f'coeffw_{num_it}')(dw2)
        elif option == 2:
            coeff_x = tf.keras.layers.Lambda(lambda z: 1./(1. + z),name=f'coeffx_{num_it}')(dx2)
            coeff_y = tf.keras.layers.Lambda(lambda z: 1./(1. + z),name=f'coeffy_{num_it}')(dy2)
            coeff_z = tf.keras.layers.Lambda(lambda z: 1./(1. + z),name=f'coeffz_{num_it}')(dz2)
            coeff_w = tf.keras.layers.Lambda(lambda z: 1./(1. + z),name=f'coeffw_{num_it}')(dw2)



        outputs_x = tf.keras.layers.multiply([coeff_x,dx])
        outputs_y = tf.keras.layers.multiply([coeff_y,dy])
        outputs_z = tf.keras.layers.multiply([coeff_z,dz])
        outputs_w = tf.keras.layers.multiply([coeff_w,dw])

        outputs_it = tf.keras.layers.add([outputs_x,outputs_y,outputs_z,outputs_w])
        zeros_y = tf.expand_dims(tf.zeros_like(outputs_it)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-3)
        pad_y = tf.keras.layers.Concatenate(axis=-2)([zeros_y,outputs_it,zeros_y])
        outputs_it = tf.keras.layers.Concatenate(axis=1)([zeros_x,pad_y,zeros_x])
        outputs = tf.keras.layers.add([0.1*outputs_it,outputs])
    
    return tf.keras.models.Model(inputs, outputs)
    
def splines(input_shape, it_lim = 1,num_classes=10,order1 = 1,CROP=256):
    order = order1
    inputs = tf.keras.Input(shape=input_shape)
    x = classifier(inputs, num_classes=num_classes,CROP=CROP)

    x = tf.keras.layers.Dropout(0.5)(x)
    y = tf.keras.layers.Flatten()(x)

    #parts = tf.keras.layers.Dense(num_classes+1, activation='softmax')(y)
    parts = tf.keras.layers.Lambda(lambda z:tf.constant(np.linspace(0,1,num_classes+1)))(y)
    parts = tf.keras.layers.Lambda(lambda z:z/tf.math.reduce_sum(z))(parts)

    outputs = inputs

    for num_it in range(it_lim):
    
        dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(outputs)
        dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(outputs)
        dz = tf.keras.layers.Lambda(lambda z: z[:,1:-1,2:] - z[:,1:-1,1:-1])(outputs)
        dw = tf.keras.layers.Lambda(lambda z: z[:,1:-1,:-2] - z[:,1:-1,1:-1])(outputs)

        dx2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx)
        dy2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy)
        dz2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dz)
        dw2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dw)

        norm = tf.keras.layers.add([dx2,dy2,dz2,dw2],name=f'norm_{num_it}')


        triangular = tf.constant(tf.linalg.LinearOperatorLowerTriangular(tf.ones((num_classes+1,num_classes+1))).to_dense())
        triangular = tf.math.maximum(0,triangular-tf.transpose(triangular))
        partition = tf.keras.layers.multiply([tf.expand_dims(triangular,axis=0),tf.expand_dims(parts,axis=-2)])
        partition = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z,axis=2))(partition)
        partition = tf.keras.layers.multiply([tf.math.reduce_max(norm,axis=(1,2)),partition],name=f'partition_{num_it}')

        partition_low = partition[:,:-1]
        partition_low = tf.expand_dims(tf.expand_dims(partition_low,axis=1),axis=1)
        partition_up = partition[:,1:]
        partition_up = tf.expand_dims(tf.expand_dims(partition_up,axis=1),axis=1)
        ineq1_x = tf.greater_equal(dx, partition_low)
        ineq2_x = tf.less(dx,partition_up)
        ineq1_y = tf.greater_equal(dy, partition_low)
        ineq2_y = tf.less(dy,partition_up)
        ineq1_z = tf.greater_equal(dz, partition_low)
        ineq2_z = tf.less(dz,partition_up)
        ineq1_w = tf.greater_equal(dw, partition_low)
        ineq2_w = tf.less(dw,partition_up)
        partition = tf.expand_dims(tf.expand_dims(partition,axis=1),axis=1)

        interval_x = tf.cast(tf.math.logical_and(ineq1_x,ineq2_x),'float32')
        interval_y = tf.cast(tf.math.logical_and(ineq1_y,ineq2_y),'float32')
        interval_z = tf.cast(tf.math.logical_and(ineq1_z,ineq2_z),'float32')
        interval_w = tf.cast(tf.math.logical_and(ineq1_w,ineq2_w),'float32')


        if num_it == 0:
        
            ct = tf.keras.layers.Dense(num_classes*(order+1),activation='linear')(y)
            ct = tf.keras.layers.Reshape((num_classes,order+1),name=f'coeff_spline_{num_it}')(ct)
            ct = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct)

            power_norm = tf.pow(norm,tf.constant(np.asarray(np.arange(order+1),dtype='float32')))
            power_norm = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-2))(power_norm)

            spline = tf.keras.layers.multiply([ct,power_norm])
            spline = tf.keras.layers.Lambda(lambda z: tf.unstack(z,axis=-1))(spline)
            spline = tf.keras.layers.add(spline)
            spline_x = tf.keras.layers.multiply([spline,interval_x])
            spline_y = tf.keras.layers.multiply([spline,interval_y])
            spline_z = tf.keras.layers.multiply([spline,interval_z])
            spline_w = tf.keras.layers.multiply([spline,interval_w])
        
        

            coeff_x = tf.expand_dims(tf.keras.layers.add(tf.unstack(spline_x,axis=-1),name=f'coeff_x_{num_it}'),axis=-1)
            coeff_y = tf.expand_dims(tf.keras.layers.add(tf.unstack(spline_y,axis=-1),name=f'coeff_y_{num_it}'),axis=-1)
            coeff_z = tf.expand_dims(tf.keras.layers.add(tf.unstack(spline_z,axis=-1),name=f'coeff_z_{num_it}'),axis=-1)
            coeff_w = tf.expand_dims(tf.keras.layers.add(tf.unstack(spline_w,axis=-1),name=f'coeff_w_{num_it}'),axis=-1)

        outputs_x = tf.keras.layers.multiply([coeff_x,dx])
        outputs_y = tf.keras.layers.multiply([coeff_y,dy])
        outputs_z = tf.keras.layers.multiply([coeff_z,dz])
        outputs_w = tf.keras.layers.multiply([coeff_w,dw])

        outputs_it = tf.keras.layers.add([outputs_x,outputs_y,outputs_z,outputs_w])
        zeros_y = tf.expand_dims(tf.zeros_like(outputs_it)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-3)
        pad_y = tf.keras.layers.Concatenate(axis=-2)([zeros_y,outputs_it,zeros_y])
        outputs_it = tf.keras.layers.Concatenate(axis=1)([zeros_x,pad_y,zeros_x])
        outputs = tf.keras.layers.add([0.1*outputs_it,outputs])
        outputs = tf.keras.layers.Lambda(lambda z: tf.clip_by_value(z,0,1))(outputs)

    return tf.keras.models.Model(inputs, outputs)

def border_splines(input_shape,border_classifier, it_lim = 1,num_classes=10,order1 = 1,CROP=256):
    order = order1
    inputs = tf.keras.Input(shape=input_shape)
    x = classifier(inputs, num_classes=num_classes,CROP=CROP)

    x = tf.keras.layers.Dropout(0.5)(x)
    y = tf.keras.layers.Flatten()(x)

    parts = tf.keras.layers.Lambda(lambda z:tf.constant(np.linspace(0,1,num_classes+1)))(y)
    parts = tf.keras.layers.Lambda(lambda z:z/tf.math.reduce_sum(z))(parts)

    outputs = inputs

    for num_it in range(it_lim):

        dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(outputs)
        dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(outputs)
        dz = tf.keras.layers.Lambda(lambda z: z[:,1:-1,2:] - z[:,1:-1,1:-1])(outputs)
        dw = tf.keras.layers.Lambda(lambda z: z[:,1:-1,:-2] - z[:,1:-1,1:-1])(outputs)

        dx2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx)
        dy2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy)
        dz2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dz)
        dw2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dw)

        norm = tf.keras.layers.add([dx2,dy2,dz2,dw2],name=f'norm_{num_it}')


        triangular = tf.constant(tf.linalg.LinearOperatorLowerTriangular(tf.ones((num_classes+1,num_classes+1))).to_dense())
        triangular = tf.math.maximum(0,triangular-tf.transpose(triangular))
        partition = tf.keras.layers.multiply([tf.expand_dims(triangular,axis=0),tf.expand_dims(parts,axis=-2)])
        partition = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z,axis=2))(partition)
        partition = tf.keras.layers.multiply([tf.math.reduce_max(norm,axis=(1,2)),partition],name=f'partition_{num_it}')

        partition_low = partition[:,:-1]
        partition_low = tf.expand_dims(tf.expand_dims(partition_low,axis=1),axis=1)
        partition_up = partition[:,1:]
        partition_up = tf.expand_dims(tf.expand_dims(partition_up,axis=1),axis=1)
        ineq1_x = tf.greater_equal(dx, partition_low)
        ineq2_x = tf.less(dx,partition_up)
        ineq1_y = tf.greater_equal(dy, partition_low)
        ineq2_y = tf.less(dy,partition_up)
        ineq1_z = tf.greater_equal(dz, partition_low)
        ineq2_z = tf.less(dz,partition_up)
        ineq1_w = tf.greater_equal(dw, partition_low)
        ineq2_w = tf.less(dw,partition_up)
        partition = tf.expand_dims(tf.expand_dims(partition,axis=1),axis=1)

        interval_x = tf.cast(tf.math.logical_and(ineq1_x,ineq2_x),'float32')
        interval_y = tf.cast(tf.math.logical_and(ineq1_y,ineq2_y),'float32')
        interval_z = tf.cast(tf.math.logical_and(ineq1_z,ineq2_z),'float32')
        interval_w = tf.cast(tf.math.logical_and(ineq1_w,ineq2_w),'float32')

        if num_it == 0:

            ct_low = tf.keras.layers.Dense(num_classes*(order+1),activation='linear')(y)
            ct_low = tf.keras.layers.Reshape((num_classes,order+1),name=f'coeff_spline_{num_it}_low')(ct_low)
            ct_low = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct_low)

            ct_up = tf.keras.layers.Dense(num_classes*(order+1),activation='linear')(y)
            ct_up = tf.keras.layers.Reshape((num_classes,order+1),name=f'coeff_spline_{num_it}_up')(ct_up)
            ct_up = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct_up)


            class_up = tf.expand_dims(tf.cast(border_classifier(inputs)>0.5,'float32'),axis=-1)
            class_low = tf.expand_dims(tf.cast(border_classifier(inputs)<=0.5,'float32'),axis=-1)

            power_norm = tf.pow(norm,tf.constant(np.asarray(np.arange(order+1),dtype='float32')))
            power_norm = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-2))(power_norm)

            spline_low = tf.keras.layers.multiply([class_low[:,1:-1,1:-1],ct_low,power_norm])
            spline_low = tf.keras.layers.Lambda(lambda z: tf.unstack(z,axis=-1))(spline_low)
            spline_low = tf.keras.layers.add(spline_low)
            spline_x_low = tf.keras.layers.multiply([spline_low,interval_x])
            spline_y_low = tf.keras.layers.multiply([spline_low,interval_y])
            spline_z_low = tf.keras.layers.multiply([spline_low,interval_z])
            spline_w_low = tf.keras.layers.multiply([spline_low,interval_w])

            spline_up = tf.keras.layers.multiply([class_up[:,1:-1,1:-1],ct_up,power_norm])
            spline_up = tf.keras.layers.Lambda(lambda z: tf.unstack(z,axis=-1))(spline_up)
            spline_up = tf.keras.layers.add(spline_up)
            spline_x_up = tf.keras.layers.multiply([spline_up,interval_x])
            spline_y_up = tf.keras.layers.multiply([spline_up,interval_y])
            spline_z_up = tf.keras.layers.multiply([spline_up,interval_z])
            spline_w_up = tf.keras.layers.multiply([spline_up,interval_w])

            spline_x = tf.keras.layers.add([spline_x_low,spline_x_up],name = 'spline_x')
            spline_y = tf.keras.layers.add([spline_y_low,spline_y_up],name = 'spline_y')
            spline_z = tf.keras.layers.add([spline_z_low,spline_z_up],name = 'spline_z')
            spline_w = tf.keras.layers.add([spline_w_low,spline_w_up],name = 'spline_w')



            coeff_x = tf.expand_dims(tf.keras.layers.add(tf.unstack(spline_x,axis=-1),name=f'coeff_x_{num_it}'),axis=-1)
            coeff_y = tf.expand_dims(tf.keras.layers.add(tf.unstack(spline_y,axis=-1),name=f'coeff_y_{num_it}'),axis=-1)
            coeff_z = tf.expand_dims(tf.keras.layers.add(tf.unstack(spline_z,axis=-1),name=f'coeff_z_{num_it}'),axis=-1)
            coeff_w = tf.expand_dims(tf.keras.layers.add(tf.unstack(spline_w,axis=-1),name=f'coeff_w_{num_it}'),axis=-1)





        outputs_x = tf.keras.layers.multiply([coeff_x,dx])
        outputs_y = tf.keras.layers.multiply([coeff_y,dy])
        outputs_z = tf.keras.layers.multiply([coeff_z,dz])
        outputs_w = tf.keras.layers.multiply([coeff_w,dw])


        outputs_it = tf.keras.layers.add([outputs_x,outputs_y,outputs_z,outputs_w])
        zeros_y = tf.expand_dims(tf.zeros_like(outputs_it)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-3)
        pad_y = tf.keras.layers.Concatenate(axis=-2)([zeros_y,outputs_it,zeros_y])
        outputs_it = tf.keras.layers.Concatenate(axis=1)([zeros_x,pad_y,zeros_x])
        outputs = tf.keras.layers.add([0.1*outputs_it,outputs])
        outputs = tf.keras.layers.Lambda(lambda z: tf.clip_by_value(z,0,1))(outputs)

    return tf.keras.models.Model(inputs, outputs)
    
    
def decreasing(input_shape, it_lim = 1,num_classes=10,CROP=256):

    inputs = tf.keras.Input(shape=input_shape)
    x = classifier(inputs, num_classes=num_classes,CROP=CROP)

    x = tf.keras.layers.Dropout(0.2)(x)
    y = tf.keras.layers.Flatten()(x)

    y1 = tf.keras.layers.MaxPool1D(3)(tf.expand_dims(y,axis=-1))
    y1 = tf.squeeze(y1,axis=-1)

    dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(inputs)
    dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(inputs)
    dz = tf.keras.layers.Lambda(lambda z: z[:,1:-1,2:] - z[:,1:-1,1:-1])(inputs)
    dw = tf.keras.layers.Lambda(lambda z: z[:,1:-1,:-2] - z[:,1:-1,1:-1])(inputs)

    dx2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx)
    dy2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy)
    dz2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dz)
    dw2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dw)

    norm = tf.keras.layers.add([dx2,dy2,dz2,dw2],name='norm')

    #parts = tf.keras.layers.Dense(num_classes+1, activation='softmax')(y)
    parts = tf.keras.layers.Lambda(lambda z:tf.constant(np.linspace(0,1,num_classes+1)))(y)
    parts = tf.keras.layers.Lambda(lambda z:z/tf.math.reduce_sum(z))(parts)

    triangular = tf.constant(tf.linalg.LinearOperatorLowerTriangular(tf.ones((num_classes+1,num_classes+1))).to_dense())
    triangular = tf.math.maximum(0,triangular-tf.transpose(triangular))
    partition = tf.keras.layers.multiply([tf.expand_dims(triangular,axis=0),tf.expand_dims(parts,axis=-2)])
    partition = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z,axis=2))(partition)
    partition = tf.keras.layers.multiply([tf.math.reduce_max(norm,axis=(1,2)),partition],name='partition')

    initial = tf.keras.layers.Dense(1,activation = "linear")(y1)
    initial = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,2),name='initial')(initial)
    triangular = tf.constant(tf.linalg.LinearOperatorLowerTriangular(tf.ones((num_classes,num_classes))).to_dense())

    flat_norm = tf.keras.layers.Conv2D(num_classes,(5,5))(norm)
    flat_norm = tf.keras.layers.MaxPool2D(pool_size=(5,5))(flat_norm)
    flat_norm = tf.keras.layers.Flatten()(flat_norm)
    simple = tf.keras.layers.Dense(num_classes,activation = "sigmoid")(flat_norm)-1
    simple = tf.keras.layers.multiply([tf.expand_dims(triangular,axis=0),tf.expand_dims(simple,axis=-2)],name="mult1")+1
    simple = tf.keras.layers.Lambda(lambda z: tf.math.reduce_prod(z,axis=2),name = 'simple_extract')(simple)
    simple = tf.keras.layers.multiply([simple,initial])
    simple = tf.expand_dims(tf.expand_dims(simple,axis=-2),axis=-2)
    

    partition_low = partition[:,:-1]
    partition_low = tf.expand_dims(tf.expand_dims(partition_low,axis=1),axis=1)
    partition_up = partition[:,1:]
    partition_up = tf.expand_dims(tf.expand_dims(partition_up,axis=1),axis=1)

    outputs = inputs
    partition = tf.expand_dims(tf.expand_dims(partition,axis=1),axis=1)

    for num_it in range(it_lim):

        dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(outputs)
        dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(outputs)
        dz = tf.keras.layers.Lambda(lambda z: z[:,1:-1,2:] - z[:,1:-1,1:-1])(outputs)
        dw = tf.keras.layers.Lambda(lambda z: z[:,1:-1,:-2] - z[:,1:-1,1:-1])(outputs)

        dx2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx)
        dy2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy)
        dz2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dz)
        dw2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dw)

        ineq1_x = tf.greater_equal(dx, partition_low)
        ineq2_x = tf.less(dx,partition_up)
        ineq1_y = tf.greater_equal(dy, partition_low)
        ineq2_y = tf.less(dy,partition_up)
        ineq1_z = tf.greater_equal(dz, partition_low)
        ineq2_z = tf.less(dz,partition_up)
        ineq1_w = tf.greater_equal(dw, partition_low)
        ineq2_w = tf.less(dw,partition_up)


        interval_x = tf.cast(tf.math.logical_and(ineq1_x,ineq2_x),'float32')
        interval_y = tf.cast(tf.math.logical_and(ineq1_y,ineq2_y),'float32')
        interval_z = tf.cast(tf.math.logical_and(ineq1_z,ineq2_z),'float32')
        interval_w = tf.cast(tf.math.logical_and(ineq1_w,ineq2_w),'float32')


        simple_x = tf.keras.layers.multiply([interval_x,simple])
        simple_x = tf.keras.layers.multiply([simple_x,initial],name=f"simple_x_{num_it}")
        simple_y = tf.keras.layers.multiply([interval_y,simple])
        simple_y = tf.keras.layers.multiply([simple_y,initial],name=f"simple_y_{num_it}")
        simple_z = tf.keras.layers.multiply([interval_z,simple])
        simple_z = tf.keras.layers.multiply([simple_z,initial],name=f"simple_z_{num_it}")
        simple_w = tf.keras.layers.multiply([interval_w,simple])
        simple_w = tf.keras.layers.multiply([simple_w,initial],name=f"simple_w_{num_it}")

        coeff_x = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_x,axis=-1),name=f'coeff_x_{num_it}'),axis=-1)
        coeff_y = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_y,axis=-1),name=f'coeff_y_{num_it}'),axis=-1)
        coeff_z = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_z,axis=-1),name=f'coeff_z_{num_it}'),axis=-1)
        coeff_w = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_w,axis=-1),name=f'coeff_w_{num_it}'),axis=-1)

        outputs_x = tf.keras.layers.multiply([coeff_x,dx])
        outputs_y = tf.keras.layers.multiply([coeff_y,dy])
        outputs_z = tf.keras.layers.multiply([coeff_z,dz])
        outputs_w = tf.keras.layers.multiply([coeff_w,dw])

        

        outputs_it = tf.keras.layers.add([outputs_x,outputs_y,outputs_z,outputs_w])
        zeros_y = tf.expand_dims(tf.zeros_like(outputs_it)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-3)
        pad_y = tf.keras.layers.Concatenate(axis=-2)([zeros_y,outputs_it,zeros_y])
        outputs_it = tf.keras.layers.Concatenate(axis=1)([zeros_x,pad_y,zeros_x])
        outputs = tf.keras.layers.add([0.1*outputs_it,outputs])
        
    return tf.keras.models.Model(inputs, outputs)

def border_decreasing(input_shape,border_classifier, it_lim = 1,num_classes=10,CROP=256):

    it_lim = min(it_lim,7)
    inputs = tf.keras.Input(shape=input_shape)
    x = classifier(inputs, num_classes=num_classes,CROP=CROP)

    x = tf.keras.layers.Dropout(0.2)(x)
    y = tf.keras.layers.Flatten()(x)

    y1 = tf.keras.layers.MaxPool1D(3)(tf.expand_dims(y,axis=-1))
    y1 = tf.squeeze(y1,axis=-1)

    dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(inputs)
    dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(inputs)
    dz = tf.keras.layers.Lambda(lambda z: z[:,1:-1,2:] - z[:,1:-1,1:-1])(inputs)
    dw = tf.keras.layers.Lambda(lambda z: z[:,1:-1,:-2] - z[:,1:-1,1:-1])(inputs)

    dx2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx)
    dy2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy)
    dz2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dz)
    dw2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dw)

    norm = tf.keras.layers.add([dx2,dy2,dz2,dw2],name='norm')

    parts = tf.keras.layers.Lambda(lambda z:tf.constant(np.linspace(0,1,num_classes+1)))(y)
    parts = tf.keras.layers.Lambda(lambda z:z/tf.math.reduce_sum(z))(parts)

    triangular = tf.constant(tf.linalg.LinearOperatorLowerTriangular(tf.ones((num_classes+1,num_classes+1))).to_dense())
    triangular = tf.math.maximum(0,triangular-tf.transpose(triangular))
    partition = tf.keras.layers.multiply([tf.expand_dims(triangular,axis=0),tf.expand_dims(parts,axis=-2)])
    partition = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z,axis=2))(partition)
    partition = tf.keras.layers.multiply([tf.math.reduce_max(norm,axis=(1,2)),partition],name='partition')

    initial_low = tf.keras.layers.Dense(1,activation = "linear")(y1)
    initial_low = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,2),name='initial_low')(initial_low)
    initial_up = tf.keras.layers.Dense(1,activation = "linear")(y1)
    initial_up = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,2),name='initial_up')(initial_up)
    triangular = tf.constant(tf.linalg.LinearOperatorLowerTriangular(tf.ones((num_classes,num_classes))).to_dense())

    class_up = tf.cast(border_classifier(inputs)>0.5,'float32')
    class_low = tf.cast(border_classifier(inputs)<=0.5,'float32')

    flat_norm_low = tf.keras.layers.multiply([class_low[:,1:-1,1:-1],norm])
    flat_norm_low = tf.keras.layers.Conv2D(num_classes,(5,5))(flat_norm_low)
    flat_norm_low = tf.keras.layers.MaxPool2D(pool_size=(5,5))(flat_norm_low)
    flat_norm_low = tf.keras.layers.Conv2D(num_classes,(2,2))(flat_norm_low)
    flat_norm_low = tf.keras.layers.MaxPool2D(pool_size=(2,2))(flat_norm_low)
    flat_norm_low = tf.keras.layers.Flatten()(flat_norm_low)
    simple_low = tf.keras.layers.Dense(num_classes,activation = "sigmoid")(flat_norm_low)-1
    simple_low = tf.keras.layers.multiply([tf.expand_dims(triangular,axis=0),tf.expand_dims(simple_low,axis=-2)])+1
    simple_low = tf.keras.layers.Lambda(lambda z: tf.math.reduce_prod(z,axis=2))(simple_low)
    simple_low = tf.keras.layers.multiply([simple_low,initial_low])
    simple_low = tf.expand_dims(tf.expand_dims(simple_low,axis=-2),axis=-2)

    flat_norm_up = tf.keras.layers.multiply([class_up[:,1:-1,1:-1],norm])
    flat_norm_up = tf.keras.layers.Conv2D(num_classes,(5,5))(flat_norm_up)
    flat_norm_up = tf.keras.layers.MaxPool2D(pool_size=(5,5))(flat_norm_up)
    flat_norm_up = tf.keras.layers.Conv2D(num_classes,(2,2))(flat_norm_up)
    flat_norm_up = tf.keras.layers.MaxPool2D(pool_size=(2,2))(flat_norm_up)
    flat_norm_up = tf.keras.layers.Flatten()(flat_norm_up)
    simple_up = tf.keras.layers.Dense(num_classes,activation = "sigmoid")(flat_norm_up)-1
    simple_up = tf.keras.layers.multiply([tf.expand_dims(triangular,axis=0),tf.expand_dims(simple_up,axis=-2)])+1
    simple_up = tf.keras.layers.Lambda(lambda z: tf.math.reduce_prod(z,axis=2))(simple_up)
    simple_up = tf.keras.layers.multiply([simple_up,initial_up])
    simple_up = tf.expand_dims(tf.expand_dims(simple_up,axis=-2),axis=-2)

    partition_low = partition[:,:-1]
    partition_low = tf.expand_dims(tf.expand_dims(partition_low,axis=1),axis=1)
    partition_up = partition[:,1:]
    partition_up = tf.expand_dims(tf.expand_dims(partition_up,axis=1),axis=1)

    outputs = inputs
    partition = tf.expand_dims(tf.expand_dims(partition,axis=1),axis=1)

    for num_it in range(it_lim):

        dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(outputs)
        dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(outputs)
        dz = tf.keras.layers.Lambda(lambda z: z[:,1:-1,2:] - z[:,1:-1,1:-1])(outputs)
        dw = tf.keras.layers.Lambda(lambda z: z[:,1:-1,:-2] - z[:,1:-1,1:-1])(outputs)

        dx2 = tf.keras.layers.Lambda(lambda z:tf.math.abs(z))(dx)
        dy2 = tf.keras.layers.Lambda(lambda z:tf.math.abs(z))(dy)
        dz2 = tf.keras.layers.Lambda(lambda z:tf.math.abs(z))(dz)
        dw2 = tf.keras.layers.Lambda(lambda z:tf.math.abs(z))(dw)

        ineq1_x = tf.greater_equal(dx2, partition_low)
        ineq2_x = tf.less(dx,partition_up)
        ineq1_y = tf.greater_equal(dy2, partition_low)
        ineq2_y = tf.less(dy,partition_up)
        ineq1_z = tf.greater_equal(dz2, partition_low)
        ineq2_z = tf.less(dz,partition_up)
        ineq1_w = tf.greater_equal(dw2, partition_low)
        ineq2_w = tf.less(dw,partition_up)


        interval_x = tf.cast(tf.math.logical_and(ineq1_x,ineq2_x),'float32')
        interval_y = tf.cast(tf.math.logical_and(ineq1_y,ineq2_y),'float32')
        interval_z = tf.cast(tf.math.logical_and(ineq1_z,ineq2_z),'float32')
        interval_w = tf.cast(tf.math.logical_and(ineq1_w,ineq2_w),'float32')



        simple_x_low = tf.keras.layers.multiply([class_low[:,1:-1,1:-1],interval_x,simple_low])
        simple_x_low = tf.keras.layers.multiply([simple_x_low,initial_low],name=f"simple_x_{num_it}_low")
        simple_y_low = tf.keras.layers.multiply([class_low[:,1:-1,1:-1],interval_y,simple_low])
        simple_y_low = tf.keras.layers.multiply([simple_y_low,initial_low],name=f"simple_y_{num_it}_low")
        simple_z_low = tf.keras.layers.multiply([class_low[:,1:-1,1:-1],interval_z,simple_low])
        simple_z_low = tf.keras.layers.multiply([simple_z_low,initial_low],name=f"simple_z_{num_it}_low")
        simple_w_low = tf.keras.layers.multiply([class_low[:,1:-1,1:-1],interval_w,simple_low])
        simple_w_low = tf.keras.layers.multiply([simple_w_low,initial_low],name=f"simple_w_{num_it}_low")

        simple_x_up = tf.keras.layers.multiply([class_up[:,1:-1,1:-1],interval_x,simple_up])
        simple_x_up = tf.keras.layers.multiply([simple_x_up,initial_up],name=f"simple_x_{num_it}_up")
        simple_y_up = tf.keras.layers.multiply([class_up[:,1:-1,1:-1],interval_y,simple_up])
        simple_y_up = tf.keras.layers.multiply([simple_y_up,initial_up],name=f"simple_y_{num_it}_up")
        simple_z_up = tf.keras.layers.multiply([class_up[:,1:-1,1:-1],interval_z,simple_up])
        simple_z_up = tf.keras.layers.multiply([simple_z_up,initial_up],name=f"simple_z_{num_it}_up")
        simple_w_up = tf.keras.layers.multiply([class_up[:,1:-1,1:-1],interval_w,simple_up])
        simple_w_up = tf.keras.layers.multiply([simple_w_up,initial_up],name=f"simple_w_{num_it}_up")

        simple_x = tf.keras.layers.add([simple_x_low,simple_x_up],name=f'simple_x_{num_it}')
        simple_y = tf.keras.layers.add([simple_y_low,simple_y_up],name=f'simple_y_{num_it}')
        simple_z = tf.keras.layers.add([simple_z_low,simple_z_up],name=f'simple_z_{num_it}')
        simple_w = tf.keras.layers.add([simple_w_low,simple_w_up],name=f'simple_w_{num_it}')

        coeff_x = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_x,axis=-1),name=f'coeff_x_{num_it}'),axis=-1)
        coeff_y = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_y,axis=-1),name=f'coeff_y_{num_it}'),axis=-1)
        coeff_z = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_z,axis=-1),name=f'coeff_z_{num_it}'),axis=-1)
        coeff_w = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_w,axis=-1),name=f'coeff_w_{num_it}'),axis=-1)

        outputs_x = tf.keras.layers.multiply([coeff_x,dx])
        outputs_y = tf.keras.layers.multiply([coeff_y,dy])
        outputs_z = tf.keras.layers.multiply([coeff_z,dz])
        outputs_w = tf.keras.layers.multiply([coeff_w,dw])



        outputs_it = tf.keras.layers.add([outputs_x,outputs_y,outputs_z,outputs_w])
        zeros_y = tf.expand_dims(tf.zeros_like(outputs_it)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-3)
        pad_y = tf.keras.layers.Concatenate(axis=-2)([zeros_y,outputs_it,zeros_y])
        outputs_it = tf.keras.layers.Concatenate(axis=1)([zeros_x,pad_y,zeros_x])
        outputs = tf.keras.layers.add([0.1*outputs_it,outputs])


    return tf.keras.models.Model(inputs, outputs)


def get_model(arch,it_lim,image_size,typ='gaussian',var = 1,num_classes=1,CROP = 256,option = 1,order = 1):


    if arch == "lambdas":
        #iteration = lambdas
        #denoising = iteration(input_shape=image_size + (1,), num_classes=num_classes,option=option,CROP=CROP)
        return lambdas(image_size + (1,),it_lim = it_lim, option=option,CROP=CROP)

    if arch == "splines":
        #iteration = splines
        #denoising = iteration(input_shape=image_size + (1,), num_classes=num_classes,order1=order,CROP=CROP)
        return splines(image_size + (1,), it_lim = it_lim,num_classes=num_classes,order1 = order,CROP=CROP)

    if arch == "decreasing":
        #iteration = decreasing
        #denoising = iteration(input_shape=image_size + (1,), num_classes=num_classes,CROP=CROP)
        return decreasing(image_size + (1,), it_lim = it_lim,num_classes=num_classes,CROP=CROP)
    
    if arch == "unet":
        return get_unet(tf.keras.layers.Input(shape=image_size + (1,)))
    
    if arch == "border_lambdas":
        border_classifier = get_unet(tf.keras.layers.Input(shape=image_size + (1,)))
        available = glob(f'../../11_oct/border_classifying/checkpoints/classifier_{typ}_*.index')
        trained = [float(i.split('_')[-1][:-6]) for i in available]
        comparison = np.array(trained) - var
        border_classifier.load_weights(available[np.argmin(np.where(comparison >=0,comparison,1e16))][:-6])
        
        for layer in border_classifier.layers:
            layer.trainable = False

        return border_lambdas(image_size + (1,),border_classifier=border_classifier,it_lim = it_lim, option=option,CROP=CROP)
    
    
    if arch == "border_splines":
        border_classifier = get_unet(tf.keras.layers.Input(shape=image_size + (1,)))
        available = glob(f'../../11_oct/border_classifying/checkpoints/classifier_{typ}_*.index')
        trained = [float(i.split('_')[-1][:-6]) for i in available]
        comparison = np.array(trained) - var
        border_classifier.load_weights(available[np.argmin(np.where(comparison >=0,comparison,1e16))][:-6])
        
        for layer in border_classifier.layers:
            layer.trainable = False

        return border_splines(image_size + (1,),border_classifier=border_classifier, it_lim = it_lim,num_classes=num_classes,order1 = order,CROP=CROP)
    
    if arch == "border_splines":
        border_classifier = get_unet(tf.keras.layers.Input(shape=image_size + (1,)))
        available = glob(f'../../11_oct/border_classifying/checkpoints/classifier_{typ}_*.index')
        trained = [float(i.split('_')[-1][:-6]) for i in available]
        comparison = np.array(trained) - var
        border_classifier.load_weights(available[np.argmin(np.where(comparison >=0,comparison,1e16))][:-6])
        
        for layer in border_classifier.layers:
            layer.trainable = False

        return border_decreasing(image_size + (1,),border_classifier=border_classifier, it_lim = it_lim,num_classes=num_classes,CROP=CROP)


    return None


