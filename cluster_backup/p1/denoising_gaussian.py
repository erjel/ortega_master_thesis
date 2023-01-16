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
    gauss = np.random.normal(0,var**2,img.shape)
    gauss = gauss.reshape(img.shape[0],img.shape[1]).astype('uint8')
    img_gauss = np.clip(np.copy(img) + gauss,0,255)
    img_gauss = np.asarray(img_gauss,dtype = np.float32)
    
    input_channel = img/np.amax(img)
    img_gauss = img_gauss/np.amax(img)
    
    return np.array([input_channel, img_gauss])

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

def iteration(input_shape, option=1, num_classes=10,order1 = 1):
    order = order1 + 1
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [16,32,64,128]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    y = tf.keras.layers.Flatten()(x)
    
    dx = tf.keras.layers.Lambda(divergence_x)(inputs)
    dy = tf.keras.layers.Lambda(divergence_y)(inputs)

    dx = tf.keras.layers.Lambda(lambda z:tf.image.resize(z,[CROP,CROP]))(dx)
    dy = tf.keras.layers.Lambda(lambda z:tf.image.resize(z,[CROP,CROP]))(dy)

    dx2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dx)
    dy2 = tf.keras.layers.Lambda(lambda z:tf.math.pow(z,2))(dy)

    norm = tf.keras.layers.add([dx2,dy2])
    
    parts = tf.keras.layers.Dense(num_classes+1, activation='softmax')(y)
    
    triangular = tf.constant(tf.linalg.LinearOperatorLowerTriangular(tf.ones((num_classes+1,num_classes+1))).to_dense())
    triangular = tf.math.maximum(0,triangular-tf.transpose(triangular))
    partition = tf.keras.layers.multiply([tf.expand_dims(triangular,axis=0),tf.expand_dims(parts,axis=-2)])
    partition = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z,axis=2))(partition)
    partition = tf.keras.layers.multiply([tf.math.reduce_max(norm,axis=(1,2)),partition])
    
    partition_low = partition[:,:-1]
    partition_low = tf.expand_dims(tf.expand_dims(partition_low,axis=1),axis=1)
    partition_up = partition[:,1:]
    partition_up = tf.expand_dims(tf.expand_dims(partition_up,axis=1),axis=1)
    ineq1 = tf.greater_equal(norm, partition_low)
    ineq2 = tf.less(norm,partition_up)
    partition = tf.expand_dims(tf.expand_dims(partition,axis=1),axis=1)

    interval = tf.cast(tf.math.logical_and(ineq1,ineq2),'float32')
    
    ct = tf.keras.layers.Dense(num_classes*order,activation='linear')(y)
    ct = tf.keras.layers.Reshape((num_classes,order))(ct)
    ct = tf.keras.layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z,axis=1),axis=1))(ct)
    
    power_norm = tf.pow(norm,tf.constant(np.asarray(np.arange(order),dtype='float32')))
    power_norm = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-2))(power_norm)
        
    spline = tf.keras.layers.multiply([ct,power_norm])
    spline = tf.keras.layers.Lambda(lambda z: tf.unstack(z,axis=-1))(spline)
    spline = tf.keras.layers.add(spline)
    spline = tf.keras.layers.multiply([spline,interval])
        
    coeff = tf.expand_dims(tf.keras.layers.add(tf.unstack(spline,axis=-1)),axis=-1)

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

    epochs = 50
    def loss_function(y_true, y_pred):
        return 2 - tf.image.ssim(y_true,y_pred,max_val = 1.) - tf.math.tanh(tf.image.psnr(y_true,y_pred,1)/50)

    def metric(y_true, y_pred):
        return tf.image.ssim(y_true,y_pred,max_val = 1.)
    
    it_lim = 1
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

    CROP = 256
    image_size = (CROP,CROP)
    num_classes = int(sys.argv[2])
    order = int(sys.argv[3])
    
    windows = np.arange(0.05,3,0.15)

    var = int(100*windows[int(sys.argv[1])-1])/100
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

    callbacks = [tf.keras.callbacks.ModelCheckpoint(
    filepath= f"{path}/11_oct/checkpoints/p{order}_gaussian_{var}_{num_classes}",
    save_weights_only=True,
    verbose = True,
    save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, verbose=1)
]



    denoising = iteration(input_shape=image_size + (1,), num_classes=num_classes,order1=order)
    model = Model(denoising)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='mse',)

    history = model.fit(
        gen_batch_train,
        epochs=epochs,
        steps_per_epoch=100,
        validation_data=gen_batch_val,
        validation_steps=20,
        callbacks=callbacks
    )
    np.save(f"{path}/11_oct/history/p{order}_gaussian_{var}_{num_classes}.npy",np.array(list(history.history.values())))