def open_frame_poisson(frame_num,var=1):
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    input_channel = img/np.amax(img)
    poisson_noise = np.sqrt(img) * np.random.normal(0, var, img.shape)
    
    poisson = np.clip(np.copy(img) + poisson_noise,0,255)
    poisson = poisson/np.amax(img)
    
    
    return input_channel,poisson
    
    
    return input_channel,sp

def divergence_x(x):
    return(x[:,1:]-x[:,:-1])

def divergence_y(x):
    return(x[:,:,1:]-x[:,:,:-1])

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

def make_model(input_shape, option=1,num_classes=1):
    """
    Takes network Input layer 
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = make_unet(inputs)
    
    lambdas = tf.keras.layers.Conv2D(num_classes, 3, activation="relu", padding="same",name='lambdas')(x)
    lambdas = tf.keras.layers.Lambda(lambda z: tf.math.pow(z,2))(lambdas)
    
    
    dx = tf.keras.layers.Lambda(divergence_x)(inputs)
    dy = tf.keras.layers.Lambda(divergence_y)(inputs)

    dx = tf.keras.layers.Lambda(lambda z:tf.image.resize(z,[CROP,CROP]))(dx)
    dy = tf.keras.layers.Lambda(lambda z:tf.image.resize(z,[CROP,CROP]))(dy)
    
    dx2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx)
    dy2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy)
    
    norm = tf.keras.layers.add([dx2,dy2])
    
    if option == 1:
        coeff = tf.keras.layers.multiply([lambdas,norm])
        coeff = tf.keras.layers.Lambda(lambda z: tf.math.exp(-z),name='coeff')(coeff)
    elif option == 2:
        coeff = tf.keras.layers.multiply([lambdas,norm])
        coeff = tf.keras.layers.Lambda(lambda z: 1./(1. + z),name='coeff')(coeff)
        

    outputs_x = tf.keras.layers.multiply([coeff,dx])
    outputs_y = tf.keras.layers.multiply([coeff,dy])

    outputs_x = tf.keras.layers.Lambda(divergence_x)(outputs_x)
    outputs_y = tf.keras.layers.Lambda(divergence_y)(outputs_y)

    outputs_x = tf.keras.layers.Lambda(lambda x:tf.image.resize(x,[CROP,CROP]))(outputs_x)
    outputs_y = tf.keras.layers.Lambda(lambda x:tf.image.resize(x,[CROP,CROP]))(outputs_y)

    outputs = tf.keras.layers.add([outputs_x,outputs_y])
    outputs = tf.keras.layers.add([outputs,inputs])
    
    return tf.keras.models.Model(inputs, outputs)




if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from glob import glob
    from tqdm import tqdm
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    import matplotlib 
    from collections import Counter
    import scipy
    from scipy.stats import wasserstein_distance
    from sklearn.mixture import GaussianMixture
    import matplotlib.cm as cm
    from sklearn.preprocessing import normalize
    import sys
    import tensorflow as tf
    import cv2
   

    
    path = "/gpfs/soma_fs/home/ortega/scripts/PeronaMalik/thesis"
    
    test = glob(f'{path}/images/test/*.jpg')
    train = glob(f'{path}/images/train/*.jpg')
    val = glob(f'{path}/images/val/*.jpg')

    epochs = 2000
    def loss_function(y_true, y_pred):
        return 2 - tf.image.ssim(y_true,y_pred,max_val = 1.) - tf.math.tanh(tf.image.psnr(y_true,y_pred,1)/50)

    def metric(y_true, y_pred):
        return tf.image.ssim(y_true,y_pred,max_val = 1.)

    CROP = 256
    image_size = (CROP,CROP)
    option = int(sys.argv[2])
    kernel_size = int(sys.argv[3])
    
    windows = np.arange(1,10,2)

    var = int(100*windows[int(sys.argv[1])-1])/100
    X_train,Y_train = [],[]

    for _ in train:
        im = open_frame_poisson(_,var=var)
        X_train.append(np.expand_dims(im[1],axis=-1))
        Y_train.append(np.expand_dims(im[0],axis=-1))

    X_test,Y_test = [],[]

    for _ in test:
        im = open_frame_poisson(_,var=var)
        X_test.append(np.expand_dims(im[1],axis=-1))
        Y_test.append(np.expand_dims(im[0],axis=-1))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        filepath= f"{path}/11_oct/checkpoints/u_poisson_{var}_{option}_kernel{kernel_size}",
        save_weights_only=True,
        verbose = True,
        save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2, verbose=1)
    ]



    model = make_model(input_shape=image_size + (1,),option=option, num_classes=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_function,
        metrics=[metric],
    )

    history = model.fit(
        x=X_train,y=Y_train, epochs=epochs, callbacks=callbacks, validation_data=(X_test,Y_test),
    )
    np.save(f"{path}/11_oct/history/u_poisson_{var}_{option}_kernel{kernel_size}.npy",np.array(list(history.history.values())))