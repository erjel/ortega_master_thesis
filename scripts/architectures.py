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


def splines(input_shape, it_lim = 10,num_classes=10,order1 = 1,CROP=256,gamma=1,pre_smoothing=False,size=5):
    order = order1
    
    inputs = tf.keras.Input(shape=input_shape,name='input')
    #input_emb = tf.keras.layers.Input(shape = (1),name='input_emb')
    x = classifier(inputs, num_classes=num_classes,CROP=CROP)

    y = tf.keras.layers.Flatten(name='y')(x)

    partition_low = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[:-1])
    partition_low = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_low,0),0),0)
    partition_low = tf.cast(partition_low,tf.float32)
    partition_up = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[1:])
    partition_up = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_up,0),0),0)
    partition_up = tf.cast(partition_up,tf.float32)

    deltaS = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    deltaE = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    NS = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    EW = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    gS = tf.keras.layers.Lambda(lambda z:tf.ones_like(z))(inputs)
    gE = tf.keras.layers.Lambda(lambda z:tf.ones_like(z))(inputs)


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


    ct = tf.keras.layers.Concatenate(name='coeff_spline')((b,a))
    ct = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct)

    outputs = inputs
    
    if pre_smoothing:
        skip = tf.keras.layers.Conv2D(64,(size,size),padding='same',use_bias=False)(outputs)
        outputs = tf.keras.layers.MaxPool2D((2,2))(skip)
        outputs = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2))(outputs)
        outputs = tf.keras.layers.concatenate([outputs, skip])
        outputs = tf.keras.layers.Conv2D(64,(2,2),padding='same')(outputs)
        outputs = tf.keras.layers.Conv2D(1,(1,1), activation='sigmoid')(outputs)
    

    for num_it in range(it_lim):


        difS = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=1))(outputs)
        difE = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=2))(outputs)
        zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(inputs)[:,1],axis=-3)
        deltaS = tf.keras.layers.Concatenate(axis=1)([difS,zeros_x])
        deltaE = tf.keras.layers.Concatenate(axis=2)([difE,zeros_y])

        dS_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaS)
        dE_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaE)

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


    
def decreasing(input_shape, it_lim = 1,num_classes=10,CROP=256):

    inputs = tf.keras.Input(shape=input_shape,name='input')
    #input_emb = tf.keras.layers.Input(shape = (1),name='input_emb')

    #emb = tf.keras.layers.Dense(num_classes,activation='relu')(input_emb)
    #emb_initial = tf.keras.layers.Dense(1,activation = 'linear')(emb)
    #emb_initial = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,2))(emb_initial)
    #emb = tf.keras.layers.Dense(num_classes,activation='sigmoid',name='embedding')(emb)
    x = classifier(inputs, num_classes=num_classes,CROP=CROP)

    y = tf.keras.layers.Flatten()(x)

    y1 = y

    initial = tf.keras.layers.Dense(1,activation = "linear")(y1)
    initial = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,2),name='initial')(initial)
    simple = tf.keras.layers.Dense(num_classes,activation = "sigmoid")(y1)
    simple = tf.keras.layers.Lambda(lambda z:tf.stack([z for it in range(num_classes)],axis=-2))(simple)-1
    triangular = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(simple)).to_dense()
    #emb = tf.keras.layers.Lambda(lambda z:tf.stack([z for it in range(num_classes)],axis=-2))(emb)-1
    #emb = tf.keras.layers.multiply([triangular,emb])+1
    #emb = tf.keras.layers.Lambda(lambda z: tf.math.reduce_prod(z,axis=-1))(emb)
    simple = tf.keras.layers.multiply([triangular,simple])+1
    simple = tf.keras.layers.Lambda(lambda z: tf.math.reduce_prod(z,axis=-1))(simple)
    simple = tf.keras.layers.multiply([simple,initial],name = 'simple_extract')
    #simple = tf.keras.layers.multiply([simple,initial,emb,emb_initial],name = 'simple_extract')


    partition_low = tf.constant(np.power(np.linspace(0,1,num_classes+1),2)[:-1])
    partition_low = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_low,0),0),0)
    partition_low = tf.cast(partition_low,tf.float32)
    partition_up = tf.constant(np.power(np.linspace(0,1,num_classes+1),2)[1:])
    partition_up = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_up,0),0),0)
    partition_up = tf.cast(partition_up,tf.float32)

    outputs = inputs

    for num_it in range(it_lim):

        dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(outputs)
        dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(outputs)
        dz = tf.keras.layers.Lambda(lambda z: z[:,1:-1,2:] - z[:,1:-1,1:-1])(outputs)
        dw = tf.keras.layers.Lambda(lambda z: z[:,1:-1,:-2] - z[:,1:-1,1:-1])(outputs)

        dx_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(dx)
        dy_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(dy)
        dz_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(dz)
        dw_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(dw)

        dx_n = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2),name=f'dx_{num_it}')(dx_n)
        dy_n = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy_n)
        dz_n = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dz_n)
        dw_n = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dw_n)


        ineq1_x = tf.greater_equal(dx_n, partition_low)
        ineq2_x = tf.less_equal(dx_n,partition_up)
        ineq1_y = tf.greater_equal(dy_n, partition_low)
        ineq2_y = tf.less_equal(dy_n,partition_up)
        ineq1_z = tf.greater_equal(dz_n, partition_low)
        ineq2_z = tf.less_equal(dz_n,partition_up)
        ineq1_w = tf.greater_equal(dw_n, partition_low)
        ineq2_w = tf.less_equal(dw_n,partition_up)

        interval_x = tf.cast(tf.math.logical_and(ineq1_x,ineq2_x),tf.float32)
        interval_y = tf.cast(tf.math.logical_and(ineq1_y,ineq2_y),tf.float32)
        interval_z = tf.cast(tf.math.logical_and(ineq1_z,ineq2_z),tf.float32)
        interval_w = tf.cast(tf.math.logical_and(ineq1_w,ineq2_w),tf.float32)


        simple_x = tf.keras.layers.multiply([interval_x,simple],name=f"simple_x_{num_it}")
        simple_y = tf.keras.layers.multiply([interval_y,simple],name=f"simple_y_{num_it}")
        simple_z = tf.keras.layers.multiply([interval_z,simple],name=f"simple_z_{num_it}")
        simple_w = tf.keras.layers.multiply([interval_w,simple],name=f"simple_w_{num_it}")

        coeff_x = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_x,axis=-1),name=f'coeff_x_{num_it}'),axis=-1)
        coeff_y = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_y,axis=-1),name=f'coeff_y_{num_it}'),axis=-1)
        coeff_z = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_z,axis=-1),name=f'coeff_z_{num_it}'),axis=-1)
        coeff_w = tf.expand_dims(tf.keras.layers.add(tf.unstack(simple_w,axis=-1),name=f'coeff_w_{num_it}'),axis=-1)

        outputs_x = tf.keras.layers.multiply([coeff_x,dx])
        outputs_y = tf.keras.layers.multiply([coeff_y,dy])
        outputs_z = tf.keras.layers.multiply([coeff_z,dz])
        outputs_w = tf.keras.layers.multiply([coeff_w,dw])

        outputs_it = tf.keras.layers.add([outputs_x,outputs_y,outputs_z,outputs_w])
        new_outputs = tf.ones_like(outputs_it)
        zeros_y = tf.expand_dims(tf.zeros_like(outputs_it)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-3)
        pad_y = tf.keras.layers.Concatenate(axis=-2)([zeros_y,outputs_it,zeros_y])
        new_pad_y = tf.keras.layers.Concatenate(axis=-2)([zeros_y,new_outputs,zeros_y])
        outputs_it = tf.keras.layers.Concatenate(axis=1)([zeros_x,pad_y,zeros_x])
        new_outputs = tf.keras.layers.Concatenate(axis=1)([zeros_x,new_pad_y,zeros_x])
        new_outputs = tf.keras.layers.multiply([new_outputs,outputs])
        new_outputs = tf.keras.layers.add([0.1*outputs_it,new_outputs])
        new_outputs = tf.keras.layers.Lambda(lambda z: tf.clip_by_value(z,0,1))(new_outputs)
        outputs = new_outputs

        
    return tf.keras.models.Model(inputs, outputs)

def decreasing_taylor(input_shape, it_lim = 1,num_classes=10,CROP=256,gamma=1):

    inputs = tf.keras.Input(shape=input_shape,name='input')
    #input_emb = tf.keras.layers.Input(shape = (1),name='input_emb')

    #emb = tf.keras.layers.Dense(num_classes,activation='relu')(input_emb)
    #emb = tf.keras.layers.Dense(num_classes,activation='sigmoid',name='embedding')(emb)
    x = classifier(inputs, num_classes=num_classes,CROP=CROP)

    y = tf.keras.layers.Flatten()(x)

    y1 = y

    dx = tf.keras.layers.Lambda(lambda z: z[:,:-2,1:-1] - z[:,1:-1,1:-1])(inputs)
    dy = tf.keras.layers.Lambda(lambda z: z[:,2:,1:-1] - z[:,1:-1,1:-1])(inputs)
    dx_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(dx)
    dy_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(dy)
    dx_n = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx_n)
    dy_n = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy_n)

    cte = tf.keras.layers.Dense(1,activation = 'relu')(y1)
    #cte_emb = tf.keras.layers.Dense(1,activation = 'relu')(emb)
    #cte = tf.keras.layers.add([cte,cte_emb])
    cte = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,2))(cte)

    linear = tf.keras.layers.Dense(1,activation = 'relu')(y1)
    #linear_emb = tf.keras.layers.Dense(1,activation = 'relu')(emb)
    #linear = tf.keras.layers.add([linear,linear_emb])
    quad = tf.keras.layers.Dense(num_classes-2,activation='linear')(y1)
    #quad_emb = tf.keras.layers.Dense(num_classes-2,activation='linear')(emb)
    #quad = tf.keras.layers.add([quad,quad_emb])

    fact = tf.constant(np.arange(2,num_classes,1),dtype=tf.float32)
    fact = tf.stack([fact for it in range(num_classes-2)],axis=-2)-1
    triangular = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(fact)).to_dense()
    fact = tf.keras.layers.multiply([triangular,fact])+1
    fact = tf.keras.layers.Lambda(lambda z: tf.math.reduce_prod(z,axis=-1))(fact)
    quad = tf.keras.layers.Lambda(lambda z: z[0]/z[1],name='quad')([quad,fact])
    sum_quad = tf.keras.layers.add(tf.unstack(quad,axis=-1))

    for dim in range(2):
        quad = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=1))(quad)

    val_quad_x = tf.pow(dx_n,tf.constant(np.asarray(np.arange(1,num_classes-1),dtype='float32')))
    val_quad_y = tf.pow(dy_n,tf.constant(np.asarray(np.arange(1,num_classes-1),dtype='float32')))

    quad_val_x = tf.keras.layers.multiply([quad,val_quad_x])
    quad_val_x = tf.keras.layers.Lambda(lambda z: tf.unstack(z,axis=-1))(quad_val_x)
    quad_val_x = tf.keras.layers.add(quad_val_x)
    quad_val_y = tf.keras.layers.multiply([quad,val_quad_y])
    quad_val_y = tf.keras.layers.Lambda(lambda z: tf.unstack(z,axis=-1))(quad_val_y)
    quad_val_y = tf.keras.layers.add(quad_val_y)

    bound_x = tf.reduce_max(quad_val_x,axis=(1,2))
    bound_y = tf.reduce_max(quad_val_y,axis=(1,2))
    bound = tf.math.abs(tf.reduce_max([bound_x,bound_y],axis=0))

    linear = tf.keras.layers.add((linear,tf.expand_dims(bound,axis=-1)),name='linear')
    linear = tf.keras.layers.Lambda(lambda z: -z)(linear)

    negative = tf.keras.layers.add([cte,linear,sum_quad])
    negative = tf.cast(tf.less(negative,0),dtype=tf.float32)
    negative = tf.keras.layers.multiply([sum_quad,negative])
    cte = tf.keras.layers.add([cte,-negative],name='cte')


    outputs = inputs

    for num_it in range(it_lim):
        
        difS = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=1))(outputs)
        difE = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=2))(outputs)
        zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(inputs)[:,1],axis=-3)
        deltaS = tf.keras.layers.Concatenate(axis=1)([difS,zeros_x])
        deltaE = tf.keras.layers.Concatenate(axis=2)([difE,zeros_y])

        dS_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaS)
        dE_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaE)

        dS2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2),name=f'dx_{num_it}')(dS_n)
        dE2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dE_n)

        val_quad_S = tf.pow(dS2,tf.constant(np.asarray(np.arange(2,num_classes),dtype='float32')))
        val_quad_E = tf.pow(dE2,tf.constant(np.asarray(np.arange(2,num_classes),dtype='float32')))

        quad_val_S = tf.keras.layers.multiply([quad,val_quad_S])
        quad_val_S = tf.keras.layers.Lambda(lambda z: tf.unstack(z,axis=-1))(quad_val_S)
        quad_val_S = tf.keras.layers.add(quad_val_S)
        quad_val_S = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-1))(quad_val_S)
        quad_val_E = tf.keras.layers.multiply([quad,val_quad_E])
        quad_val_E = tf.keras.layers.Lambda(lambda z: tf.unstack(z,axis=-1))(quad_val_E)
        quad_val_E = tf.keras.layers.add(quad_val_E)
        quad_val_E = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-1))(quad_val_E)
        

        linear_S = tf.keras.layers.multiply((linear,dS2))
        linear_E = tf.keras.layers.multiply((linear,dE2))

        gS = tf.keras.layers.add([cte,linear_S,quad_val_S],name=f'gS_{num_it}')
        gE = tf.keras.layers.add([cte,linear_E,quad_val_E],name=f'gE_{num_it}')

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

        
    #return tf.keras.models.Model([inputs,input_emb], outputs)
    return tf.keras.models.Model(inputs, outputs)


def decreasing_linear(input_shape, it_lim = 1,num_classes=10,CROP=256,order=1,gamma=1):

    inputs = tf.keras.Input(shape=input_shape,name='input')
    #input_emb = tf.keras.layers.Input(shape = (1),name='input_emb')

    x = classifier(inputs, num_classes=num_classes,CROP=CROP)

    y = tf.keras.layers.Flatten(name='y')(x)

    partition_low = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[:-1])
    partition_low = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_low,0),0),0)
    partition_low = tf.cast(partition_low,tf.float32)
    partition_up = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[1:])
    partition_up = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_up,0),0),0)
    partition_up = tf.cast(partition_up,tf.float32)

    deltaS = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    deltaE = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    NS = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    EW = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    gS = tf.keras.layers.Lambda(lambda z:tf.ones_like(z))(inputs)
    gE = tf.keras.layers.Lambda(lambda z:tf.ones_like(z))(inputs)

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


    ct = tf.keras.layers.Concatenate(name='coeff_spline')((b,a))
    ct = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct)

    outputs = inputs

    for num_it in range(it_lim):


        difS = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=1))(outputs)
        difE = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=2))(outputs)
        zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(inputs)[:,1],axis=-3)
        deltaS = tf.keras.layers.Concatenate(axis=1)([difS,zeros_x])
        deltaE = tf.keras.layers.Concatenate(axis=2)([difE,zeros_y])

        dS_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaS)
        dE_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaE)

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

        E = tf.keras.layers.multiply((gE,deltaE))
        S = tf.keras.layers.multiply((gS,deltaS))

        NS = S
        EW = E
        NS = tf.keras.layers.Concatenate(axis=1)([zeros_x,NS])
        EW = tf.keras.layers.Concatenate(axis=2)([zeros_y,EW])
        NS = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=1))(NS)
        EW = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=2))(EW)

        #mult = gamma*tf.ones_like(NS)
        mult = tf.keras.layers.Lambda(lambda z: tf.multiply(tf.cast(gamma,dtype=tf.float32),z))(tf.ones_like(NS))
        NS_mod = NS
        EW_mod = EW

        adding = tf.keras.layers.add([NS,EW])
        adding = tf.keras.layers.multiply((mult,adding))

        outputs = tf.keras.layers.add([outputs,adding])

        
    #return tf.keras.models.Model([inputs,input_emb], outputs)
    return tf.keras.models.Model(inputs, outputs)



def autoencoder(input_shape, it_lim = 10,num_classes=10,order1 = 1,CROP=256,gamma=1):
    
    StableDiffusion = keras_cv.models.StableDiffusion(img_height=CROP, img_width=CROP,jit_compile=True)
    
    order = order1
    
    inputs = tf.keras.Input(shape=input_shape,name='input')
    x = tf.keras.layers.Lambda(lambda z:tf.squeeze(tf.stack([z,z,z],axis=-1)))(inputs)
    x = StableDiffusion.image_encoder(x)
    y = tf.keras.layers.Flatten(name='y')(x)

    partition_low = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[:-1])
    partition_low = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_low,0),0),0)
    partition_low = tf.cast(partition_low,tf.float32)
    partition_up = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[1:])
    partition_up = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_up,0),0),0)
    partition_up = tf.cast(partition_up,tf.float32)

    deltaS = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    deltaE = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    NS = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    EW = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    gS = tf.keras.layers.Lambda(lambda z:tf.ones_like(z))(inputs)
    gE = tf.keras.layers.Lambda(lambda z:tf.ones_like(z))(inputs)


    b_initial = tf.keras.layers.Dense(2**10,activation='relu')(y)
    b_initial = tf.keras.layers.Dropout(0.2)(b_initial)
    b_initial = tf.keras.layers.Dense(2**7,activation='relu')(b_initial)
    b_initial = tf.keras.layers.Dropout(0.2)(b_initial)
    b_initial = tf.keras.layers.Dense(1,activation='linear')(b_initial)
    b_initial = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(b_initial)

    a = tf.keras.layers.Dense(2**10,activation = 'relu')(y)
    a = tf.keras.layers.Dropout(0.2)(a)
    a = tf.keras.layers.Dense(2**7,activation = 'relu')(y)
    a = tf.keras.layers.Dropout(0.2)(a)
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


    ct = tf.keras.layers.Concatenate(name='coeff_spline')((b,a))
    ct = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct)

    outputs = inputs
    
    for num_it in range(it_lim):


        difS = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=1))(outputs)
        difE = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=2))(outputs)
        zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(inputs)[:,1],axis=-3)
        deltaS = tf.keras.layers.Concatenate(axis=1)([difS,zeros_x])
        deltaE = tf.keras.layers.Concatenate(axis=2)([difE,zeros_y])

        dS_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaS)
        dE_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaE)

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

def convex(input_shape, it_lim = 10,num_classes=10,order1 = 1,CROP=256,gamma=1,pre_smoothing=False,size=5):
    order = order1

    inputs = tf.keras.Input(shape=input_shape,name='input')
    x = classifier(inputs, num_classes=num_classes,CROP=CROP)

    y = tf.keras.layers.Flatten(name='y')(x)

    partition_low = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[:-1])
    partition_low = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_low,0),0),0)
    partition_low = tf.cast(partition_low,tf.float32)
    partition_up = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[1:])
    partition_up = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_up,0),0),0)
    partition_up = tf.cast(partition_up,tf.float32)

    deltaS = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    deltaE = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    NS = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    EW = tf.keras.layers.Lambda(lambda z:tf.zeros_like(z))(inputs)
    gS = tf.keras.layers.Lambda(lambda z:tf.ones_like(z))(inputs)
    gE = tf.keras.layers.Lambda(lambda z:tf.ones_like(z))(inputs)


    b_initial = tf.keras.layers.Dense(1,activation='linear')(y)
    b_initial = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(b_initial)

    a_initial = tf.keras.layers.Dense(1,activation='linear')(y)
    a = tf.keras.layers.Dense(num_classes*order,activation = 'linear')(y)
    a = tf.keras.layers.Lambda(lambda z: tf.pow(z,2))(a)
    a = tf.keras.layers.Reshape((num_classes,order))(a)
    a = tf.keras.layers.Lambda(lambda z: tf.cumsum(z,axis=1))(a)

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


    ct = tf.keras.layers.Concatenate(name='coeff_spline')((b,a))
    ct = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct)

    outputs = inputs

    if pre_smoothing:
        skip = tf.keras.layers.Conv2D(64,(size,size),padding='same',use_bias=False)(outputs)
        outputs = tf.keras.layers.MaxPool2D((2,2))(skip)
        outputs = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2))(outputs)
        outputs = tf.keras.layers.concatenate([outputs, skip])
        outputs = tf.keras.layers.Conv2D(64,(2,2),padding='same')(outputs)
        outputs = tf.keras.layers.Conv2D(1,(1,1), activation='sigmoid')(outputs)


    for num_it in range(it_lim):


        difS = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=1))(outputs)
        difE = tf.keras.layers.Lambda(lambda z: tf.experimental.numpy.diff(z,axis=2))(outputs)
        zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-1)
        zeros_x = tf.expand_dims(tf.zeros_like(inputs)[:,1],axis=-3)
        deltaS = tf.keras.layers.Concatenate(axis=1)([difS,zeros_x])
        deltaE = tf.keras.layers.Concatenate(axis=2)([difE,zeros_y])

        dS_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaS)
        dE_n = tf.keras.layers.Lambda(lambda z: tf.math.abs(z)/tf.reduce_max(tf.math.abs(z)))(deltaE)

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





def get_model(arch,it_lim,image_size,typ='gaussian',var = 1,num_classes=1,CROP = 256,option = 1,order = 1, pre_smoothing=False, size=5):


    

    if arch == "splines":
        return splines(image_size + (1,), it_lim = it_lim,num_classes=num_classes,order1 = order,CROP=CROP, pre_smoothing=pre_smoothing, size=size)

    if arch == "decreasing":
        return decreasing(image_size + (1,), it_lim = it_lim,num_classes=num_classes,CROP=CROP)
    
    if arch == "decreasing_taylor":
        return decreasing_taylor(image_size + (1,), it_lim = it_lim,num_classes=num_classes,CROP=CROP)
    
    if arch == "decreasing_linear":
        return decreasing_linear(image_size + (1,), it_lim = it_lim,num_classes=num_classes,CROP=CROP)
    
    if arch == "convex":
        return convex(image_size + (1,), it_lim = it_lim,num_classes=num_classes,CROP=CROP)
    
    if arch == "autoencoder":
        import keras_cv
        model = autoencoder(image_size + (1,), it_lim = it_lim,num_classes=num_classes,order1 = order,CROP=CROP)
        #model.get_layer('image_encoder').trainable = False
        return model
   

    return None


