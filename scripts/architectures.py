import tensorflow as tf
import numpy as np
from glob import glob
import sys
import os
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


class losses:
    def Linfty(y_true,y_pred):
        error = tf.math.log(tf.math.reduce_mean(tf.exp(tf.abs(y_true-y_pred))))
        return error

    def W1infty(y_true,y_pred):
        error = tf.math.log(tf.math.reduce_mean(tf.exp(tf.abs(y_true-y_pred))))

        dx_true,dy_true = tf.image.image_gradients(y_true)
        dx_pred,dy_pred = tf.image.image_gradients(y_pred)

        error += tf.math.log(tf.math.reduce_mean(tf.exp(tf.abs(dx_true-dx_pred))))
        error += tf.math.log(tf.math.reduce_mean(tf.exp(tf.abs(dy_true-dy_pred))))

        return error

    def H1(y_true,y_pred):
        error = tf.math.reduce_mean(tf.pow(y_true-y_pred,2))

        dx_true,dy_true = tf.image.image_gradients(y_true)
        dx_pred,dy_pred = tf.image.image_gradients(y_pred)

        error += tf.math.reduce_mean(tf.pow(dx_true-dx_pred,2))
        error += tf.math.reduce_mean(tf.pow(dy_true-dy_pred,2))
        error = tf.sqrt(error)

        return error

    def L2(y_true,y_pred):
        error = tf.math.reduce_mean(tf.pow(y_true-y_pred,2))  

        return error

    def probability(y_true,y_pred):
        error = -tf.image.ssim(y_true,y_pred,255)

        return error
    
    def product(y_true,y_pred):
        
        return -tf.image.psnr(y_true,y_pred,255)*tf.image.ssim(y_true,y_pred,255) 

    def psnr(y_true,y_pred):
        
        return -tf.image.psnr(y_true,y_pred,255) 
    
    def Hpsnr(y_true,y_pred):
        
        error = -tf.image.psnr(y_true,y_pred,255)
        m = tf.reduce_max(tf.abs(tf.image.image_gradients(y_true)))
        gt = tf.image.image_gradients(y_true)
        gp = tf.image.image_gradients(y_pred)
        error += -tf.image.psnr(gt[0],gp[0],m)
        error += -tf.image.psnr(gt[1],gp[1],m)
        
        return error 
    
    def Hproduct(y_true,y_pred):
        
        error = -tf.image.psnr(y_true,y_pred,255)*tf.image.ssim(y_true,y_pred,255)
        m = tf.reduce_max(tf.abs(tf.image.image_gradients(y_true)))
        gt = tf.image.image_gradients(y_true)
        gp = tf.image.image_gradients(y_pred)
        gt = [tf.cast(g,tf.float32) for g in gt]
        gp = [tf.cast(g,tf.float32) for g in gp]
        error += -tf.image.psnr(gt[0],gp[0],m)*tf.image.ssim(gt[0],gt[1],m)
        error += -tf.image.psnr(gt[1],gp[1],m)*tf.image.ssim(gt[1],gt[1],m)
        
        return error 




def classifier(inputs, option=1, num_classes=2,kernel_size=3,pool_size=3,CROP=256,latent_size=1024):
    
    
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

    x = tf.keras.layers.SeparableConv2D(latent_size, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    return x



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

def u_encoder(x, n_filt,degree=5):
    """
    Applies conv_block and returnes max pooled and skip-connection.
    """
    x = conv_block(x, n_filt,size_conv = (degree,degree))
    return tf.keras.layers.MaxPool2D()(x), x

def u_decoder(pooled, skipped, n_filt,degree=5):
    """
    Upsamples pooled and concats with skiped.
    """
    upsampled = tf.keras.layers.Convolution2DTranspose(n_filt, (degree,degree), strides=(2,2), padding='same')(pooled)
    return conv_block(tf.keras.layers.concatenate([upsampled, skipped]), n_filt)
    
def making_awareness(p,q,n_filt,degree=5):
    p = tf.keras.layers.Concatenate()([p,q])
    p = conv_block(p,n_filt,size_conv = (degree,degree))
    return p
    
def make_unet(input_shape, depth=5, output_channels=1,degree=5,nfilt=2,type_training='basic',it_lim=10):
    skipped = []
    inp = tf.keras.Input(input_shape,name='input')

    p = inp
    for _ in range(depth):
        p, s = u_encoder(p, 2**(nfilt+_),degree=degree)
        skipped.append(s)
    p = conv_block(p, 2**(2+depth))
    
    if 'aware' in type_training:
        inputs_aware = tf.keras.Input(shape=(1),name='aware')
        size = int((input_shape[0]/(2**depth))**2)
        aware = tf.keras.layers.Embedding(it_lim +1,size)(inputs_aware)
        aware = tf.keras.layers.Reshape((input_shape[0]//(2**depth),input_shape[0]//(2**depth),1))(aware)
        aware = conv_block(aware, 2**(2+depth))
        p = making_awareness(p,aware, 2**(2+depth))
    for _ in reversed(range(depth)):
        p = u_decoder(p, skipped[_], 2**(nfilt+_),degree=degree)  
        if 'aware' in type_training:
            aware = tf.keras.layers.Convolution2DTranspose(2**(nfilt+_), (degree,degree), strides=(2,2), padding='same')(aware)
            p = making_awareness(p,aware, 2**(nfilt+_))
            
    p = tf.keras.layers.Conv2D(output_channels, (1,1), activation='sigmoid')(p)
    
    if 'aware' in type_training:
        return tf.keras.Model([inp,inputs_aware],p,name='differential_operator')
    else:
        return tf.keras.Model(inp,p,name='differential_operator')
    




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
        #s0 = tf.keras.layers.Dense(1,activation='sigmoid')(y)
        s0 = tf.keras.layers.Lambda(lambda z: 0.5*tf.ones_like(z))(b_initial)

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
    
    def fluxdecreasing(y,num_classes,order):
        b_initial = tf.keras.layers.Dense(1,activation='linear')(y)
        b_initial = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(b_initial)

        a = tf.keras.layers.Dense(num_classes*order,activation = 'linear')(y)
        a = tf.keras.layers.Reshape((num_classes,order))(a)
        a = tf.keras.layers.Lambda(lambda z:-tf.math.pow(z,2))(a)
        s0 = tf.keras.layers.Lambda(lambda z: 0.5*tf.ones_like(z))(b_initial)

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
        a = tf.keras.layers.Lambda(lambda z:-tf.math.pow(z,2))(a)
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


def differential_operator(input_shape,num_filters,degree,typee,CROP=256,depth=3,nfilt=2,
                          type_training='basic',it_lim=10):
    
    
    if typee == 'unet':
        return make_unet(input_shape,degree=2*degree+1,depth=depth,nfilt=nfilt,
                         type_training=type_training,it_lim=it_lim)
    
    if typee == 'noconstraints':
        
        outputs = tf.keras.Input(shape=input_shape,name='input')        
        diff_op = tf.keras.layers.Conv2D(num_filters,(2*degree+1,2*degree+1),padding='same',use_bias=False,name='diff')(outputs)
        
        if 'aware' in type_training:
            depth = 3
            size = int((input_shape[0]/(2**depth))**2)
            inputs_aware = inputs_aware = tf.keras.Input(shape=(1),name='aware')
            aware = tf.keras.layers.Embedding(it_lim +1,size)(inputs_aware)
            aware = tf.keras.layers.Reshape((input_shape[0]//(2**depth),input_shape[0]//(2**depth),1))(aware)
            aware = conv_block(aware, 2**(2+depth))
            for _ in reversed(range(depth)):
                aware = tf.keras.layers.Convolution2DTranspose(2**(nfilt+_), (degree,degree), strides=(2,2), padding='same')(aware)

            diff_op = making_awareness(diff_op,aware, 2**(nfilt+_))
            diff_op = tf.keras.layers.Conv2D(output_channels, (1,1), activation='sigmoid')(diff_op)
        
            return tf.keras.Model([outputs,inputs_aware],diff_op,name='differential_operator')
        
        else:
            return tf.keras.Model(outputs,diff_op,name='differential_operator')
    
    
    


def get_model(arch,it_lim,image_size,BATCH_SIZE,type_training,loss,num_classes=10,order = 1,gamma=1,degree=3,
             typee='noconstraints',known_variance=True,latent_size=1024,depth=3,nfilt=2):
    
    class diffusor(tf.keras.Model):
    
        def __init__(self, arch,it_lim,image_size,loss,num_classes=10,order = 1,gamma=1,degree=3,
                 typee='noconstraints',known_variance=True,latent_size=1024,classifier=classifier,
                     type_training = 'basic',depth=3,nfilt=2,BATCH_SIZE=BATCH_SIZE):

            super().__init__()
            self.it_lim = it_lim
            self.order = order
            self.gamma = gamma
            self.degree = degree
            self.typee = typee
            self.known_variance = known_variance
            self.loss = loss
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
            self.type_training = type_training
            self.BATCH_SIZE = BATCH_SIZE


            CROP = image_size[0]


            input_shape = image_size + (1,)   
            inputs = tf.keras.Input(shape=input_shape,name='input')
            if typee != 'classic':
                num_filters = np.sum(2**np.arange(1,degree+1))
                differential_model = differential_operator(input_shape,num_filters,degree,typee,CROP=CROP,
                            depth=depth,nfilt=nfilt,type_training=type_training,it_lim=it_lim)
            else:

                dS_n,dE_n = tf.keras.layers.Lambda(lambda z: tf.image.image_gradients(z))(inputs)
                dS = tf.keras.layers.Lambda(lambda z: tf.pow(z,2))(dS_n)
                dE = tf.keras.layers.Lambda(lambda z: tf.pow(z,2))(dE_n)
                diff_op = tf.keras.layers.add((dS,dE))
                differential_model = tf.keras.Model(inputs,diff_op)

            self.differential_model = differential_model


            x = tf.keras.layers.Lambda(lambda z:z/tf.math.reduce_std(z,axis=(1,2,3),keepdims=True))(inputs)
            x = classifier(x, num_classes=num_classes,CROP=CROP,latent_size=latent_size)
            y = tf.keras.layers.Flatten(name='y')(x)

            if known_variance:
                inputs_var = tf.keras.Input(shape=(1),name='var')
                embedded = tf.keras.layers.Embedding(101,latent_size)(inputs_var)
                embedded = tf.keras.layers.Flatten(name='embedded')(embedded)
                y = tf.keras.layers.Lambda(lambda z:tf.multiply(z[0],z[1]))([y,embedded])

            a,b = getattr(function_type,arch)(y,num_classes,order)    
            b = tf.keras.layers.Lambda(lambda z: tf.multiply(tf.ones_like(z[0]),z[1]))([a,b])
            if 'flux' in arch:
                num_classes = 2*num_classes


            partition_low = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[:-1])
            partition_low = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_low,0),0),0)
            partition_low = tf.cast(partition_low,tf.float32)
            partition_up = tf.constant(np.power(np.linspace(0,1,num_classes+1),1)[1:])
            partition_up = tf.expand_dims(tf.expand_dims(tf.expand_dims(partition_up,0),0),0)
            partition_up = tf.cast(partition_up,tf.float32)

            self.partition_low = partition_low
            self.partition_up = partition_up


            ct = tf.keras.layers.Concatenate(name='coeff_spline')((b,a))
            ct = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct)

            if known_variance:
                self.diffusivity = tf.keras.Model([inputs,inputs_var],ct)
            else:
                self.diffusivity = tf.keras.Model(inputs,ct)

            ct = tf.keras.Input(shape=(1,1,num_classes,2))
            diff_op = tf.keras.Input(shape=input_shape)
            outputs = tf.keras.Input(shape=input_shape)
            ineq1 = tf.greater_equal(diff_op, self.partition_low)
            ineq2 = tf.less_equal(diff_op, self.partition_up)

            interval = tf.cast(tf.math.logical_and(ineq1,ineq2),tf.float32)

            power_norm = tf.constant(np.asarray(np.arange(1,self.order+1),dtype='float32'))
            power_norm = tf.keras.layers.Lambda(lambda z:tf.pow(z[0],z[1]))([diff_op,power_norm])
            cte = tf.ones_like(diff_op)
            power_norm = tf.keras.layers.Concatenate(axis=-1)((cte,power_norm))
            power_norm = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-2))(power_norm)

            spline = tf.keras.layers.multiply([ct,power_norm])
            spline = tf.keras.layers.Lambda(lambda z:tf.math.reduce_sum(z,axis=-1))(spline)
            spline = tf.keras.layers.multiply([spline,interval])

            g = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z,axis=-1))(spline)
            g = tf.expand_dims(g,axis=-1)


            deltaS,deltaE = tf.keras.layers.Lambda(lambda z: tf.image.image_gradients(z))(outputs)
            E = tf.keras.layers.multiply((g,deltaE))
            S = tf.keras.layers.multiply((g,deltaS))

            NS = S
            EW = E
            zeros_y = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-1)
            zeros_x = tf.expand_dims(tf.zeros_like(outputs)[:,1],axis=-3)
            NS = tf.keras.layers.Concatenate(axis=1)([zeros_x,NS])
            EW = tf.keras.layers.Concatenate(axis=2)([zeros_y,EW])
            NS = tf.keras.layers.Lambda(lambda z: z[:,1:] - z[:,:-1])(NS)
            EW = tf.keras.layers.Lambda(lambda z: z[:,:,1:] - z[:,:,:-1])(EW)

            mult = tf.keras.layers.Lambda(lambda z: tf.multiply(tf.cast(self.gamma,dtype=tf.float32),z))(tf.ones_like(NS))

            adding = tf.keras.layers.add([NS,EW])
            adding = tf.keras.layers.multiply((mult,adding))

            self.diffusion = tf.keras.Model([diff_op,outputs,ct],adding)

        def call(self, inputs):
            ct = self.diffusivity(inputs)

            if self.known_variance:
                outputs = inputs['input']
            else:
                outputs = inputs

            for num_it in range(self.it_lim):

                if 'aware' in self.type_training:
                    stage = (self.it_lim-num_it)*tf.ones(self.BATCH_SIZE)
                    diff_op = self.differential_model([outputs,stage])
                else:
                    diff_op = self.differential_model(outputs)
                adding = self.diffusion([diff_op,outputs,ct])
                outputs = tf.keras.layers.add([outputs,adding])

            return outputs

        def train_step(self, data):
            x,y = data

            with tf.GradientTape() as tape:
                ct = self.diffusivity(x,training =True)

                if self.known_variance:
                    outputs = x['input']
                else:
                    outputs = x

                for num_it in range(self.it_lim):

                    if 'aware' in self.type_training:
                        stage = (self.it_lim-num_it)*tf.ones(self.BATCH_SIZE)
                        diff_op = self.differential_model([outputs,stage],training=True)
                    else:
                        diff_op = self.differential_model(outputs,training=True)
                    adding = self.diffusion([diff_op,outputs,ct],training=True)

                    if 'greedy' in self.type_training:
                        outputs = adding + tf.stop_gradient(outputs)
                    else:
                        outputs = tf.keras.layers.add([outputs,adding])

                loss = self.loss(y, outputs)

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, data):

            x,y = data
            y_pred = self(x)
            loss = self.loss(y,y_pred)
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}
        
    loss = getattr(losses,loss)

    model = diffusor(arch,it_lim,image_size,loss,num_classes=num_classes,order = order,gamma=gamma,degree=degree,
                 typee=typee,known_variance=known_variance,latent_size=latent_size,classifier=classifier,
                     type_training = type_training,depth=depth,nfilt=nfilt,BATCH_SIZE=BATCH_SIZE)

    return model 

