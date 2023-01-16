def augment(yx, crop=256, do_flips=True, do_rotate=True, do_scale=False):
    
    ch, cw = yx[0].shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cw/2,ch/2),np.random.randint(-90,90),1)
    yx[0] = cv2.warpAffine(yx[0],rotation_matrix, (ch,cw),cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    yx[1] = cv2.warpAffine(yx[1],rotation_matrix, (ch,cw),cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return yx

def open_frame(frame_num,var=1):
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    input_channel = img/np.amax(img)
    gausBlur = cv2.blur(input_channel, (var,var)) 
    gausBlur = gausBlur/np.amax(gausBlur)
    
    return [input_channel, gausBlur]


N_REPEAT_FRAME = 1

def sample_images(frame_nums):
    while True:
        try:
            img = open_frame(np.random.choice(frame_nums),var)
        except Exception as e:
            print(f'Exception {e} on file')
            #continue
            break
        for n in range(N_REPEAT_FRAME):
            img = open_frame(np.random.choice(frame_nums),var)
            a =  augment(img, do_flips=True, do_rotate=True, do_scale=True)
            yield a
            
            
def get_data_generator(sampler,var):
    def get_data():
        while True:
            yx = next(sampler)
            x,y = yx[1],yx[0]
            yield np.expand_dims(x,axis=-1),np.expand_dims(y,axis=-1)
        
    return get_data

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

def iteration(input_shape, option=1,num_classes=1):
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
   

    CROP = 256
    image_size = (CROP,CROP)
    option = int(sys.argv[2])
    kernel_size = int(sys.argv[3])
    windows = np.arange(4,15,1)

    var = int(windows[int(sys.argv[1])-1])    
    path = "/gpfs/soma_fs/home/ortega/scripts/PeronaMalik/thesis"
    
    test = glob(f'{path}/images/test/*.jpg')
    train = glob(f'{path}/images/train/*.jpg')
    val = glob(f'{path}/images/val/*.jpg')
    
    BATCH_SIZE = 50

    dg_train = tf.data.Dataset.from_generator(
        get_data_generator(sample_images(train),var),
        output_types=(tf.float32, tf.float32),
        output_shapes=((CROP, CROP, 1),(CROP, CROP, 1)) )

    dg_val = tf.data.Dataset.from_generator(
        get_data_generator(sample_images(test),var),
        output_types=(tf.float32, tf.float32),
        output_shapes=((CROP, CROP, 1),(CROP, CROP, 1)) )

    gen_batch_train = dg_train.batch(BATCH_SIZE)
    gen_batch_val = dg_val.batch(BATCH_SIZE)

    epochs = 100
    def loss_function(y_true, y_pred):
        return 2 - tf.image.ssim(y_true,y_pred,max_val = 1.) - tf.math.tanh(tf.image.psnr(y_true,y_pred,1)/50)

    def metric(y_true, y_pred):
        return tf.image.ssim(y_true,y_pred,max_val = 1.)
    
    it_lim = 10
    class Model(tf.keras.Model):    

        def __init__(self, network):
            super(Model, self).__init__()
            self.network = network
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")


        def call(self, inputs):
            return self.network(inputs)

        def train_step(self, data):
            x, y = data
            z = x        

            with tf.GradientTape() as tape:
                loss = self._compute_loss(data,True)

            gradients = tape.gradient(loss, self.network.trainable_weights)

            self.optimizer.apply_gradients(
                zip(gradients, self.network.trainable_weights)
            )


            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, data):

            x, y = data
            z = x

            for i in range(it_lim):
                z = self.network(z,training=False)
            loss = self._compute_loss((x, y),False)
            self.compiled_loss(y,z)

            # Update the metrics.
            self.compiled_metrics.update_state(y, z)

            return {"loss":loss}




        def _compute_loss(self, data,train):

            x, y = data
            z = x
            for i in range(it_lim):
                z = self.network(z,training=train)

            return tf.keras.losses.MeanSquaredError()(y,z)

    
    

    callbacks = [tf.keras.callbacks.ModelCheckpoint(
    filepath= f"{path}/11_oct/checkpoints/u_deblurring_cartesian_{var}_{option}_kernel{kernel_size}",
    save_weights_only=True,
    verbose = True,
    save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, verbose=1)
    ]



    denoising = iteration(input_shape=image_size + (1,), num_classes=1,)
    model = Model(denoising)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='mse',)

    history = model.fit(
        gen_batch_train,
        epochs=epochs,
        steps_per_epoch=100,
        validation_data=gen_batch_val,
        validation_steps=20,
        callbacks=callbacks
    )
    np.save(f"{path}/11_oct/history/u_debluring_cartesian_{var}_{option}_kernel{kernel_size}.npy",np.array(list(history.history.values())))