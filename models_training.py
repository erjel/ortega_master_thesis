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
        error = -tf.image.ssim(y_true,y_pred,1)

        return error
    
    def product(y_true,y_pred):
        return -tf.image.psnr(y_true,y_pred,1)*tf.image.ssim(y_true,y_pred,1)

    def psnr(y_true,y_pred):
        return -tf.image.psnr(y_true,y_pred,1)



if __name__ == '__main__':
    
    from argparse import Namespace
    import os
    from pathlib import Path

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import skimage.io as io
    import skimage.filters as flt
    from skimage.metrics import structural_similarity as SSIM
    # since we can't use imports
    import numpy as np
    import scipy.ndimage.filters as flt
    import warnings
    import cv2
    from glob import glob 

    import pylab as pl
    from time import sleep
    import tensorflow as tf
    from tqdm import tqdm
    from collections import Counter
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl


    import sys
    ROOT_DIR = Path(os.environ['PWD'])
    sys.path.append(str(ROOT_DIR / 'scripts'))
    print(sys.path)
    from open_frame import open_frame as OF
    from pm_algorithm import anisodiff
    from data_augmentation import get_generators
    from architectures import get_model

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


    CROP = 256
    image_size = (CROP,CROP)
    typ = 'gaussian'
    job = int(sys.argv[1])
    total_jobs = int(sys.argv[2])
    it_lim = 10
    
    current_jobs = 0

    losses_names = ['H1','L2','W1infty','Linfty','product','psnr']
    architectures = ['splines','decreasing','flux']

    options = []

    for N_REPEAT_FRAME1 in np.arange(2,32,2):
        for arch in architectures:
            for loss in losses_names:
                for num_classes in [5,10,15,20,25,50]:
                    for degree in np.arange(1,6):
                        for f1,factor in enumerate([1]):
                            for use_polynomial in [True,False]:
                                for polynomial_degree in range(0,3):
                                    use_polynomial = not polynomial_degree == 0

                                    options.append(
                                        Namespace(
                                            N_REPEAT_FRAME1=N_REPEAT_FRAME1,
                                            arch=arch,
                                            loss=loss,
                                            num_classes=num_classes,
                                            degree=degree,
                                            f1=f1,
                                            factor=factor,
                                            use_polynomial=use_polynomial,
                                            polynomial_degree=polynomial_degree
                                        )
                                    )

    print(f'Run hyperparameter scan for {len(options)} options')
    for option in options:
        gen_batch_train,gen_batch_val = get_generators(typ,var1_d=0,var1_u=55,CROP1=CROP,BATCH_SIZE=30,
                                                        N_REPEAT_FRAME1=option.N_REPEAT_FRAME1)

        if option.use_polynomial:                          
            checkpoint_dir = f"{ROOT_DIR}/7_apr/{option.arch}/checkpoints/{option.arch}{option.loss}_{typ}_{option.num_classes}_{option.N_REPEAT_FRAME1}_{it_lim}_t{option.degree}_{option.f1}_t{option.polynomial_degree}"
        else:
            checkpoint_dir = f"{ROOT_DIR}/7_apr/{option.arch}/checkpoints/{option.arch}{option.loss}_{typ}_{option.num_classes}_{option.N_REPEAT_FRAME1}_{it_lim}_t{option.degree}_{option.f1}_f"
        history_file = f"{checkpoint_dir.replace('checkpoint', 'history')}.npy"

        current_jobs += 1
        print(
            current_jobs,
            checkpoint_dir
        )

        if current_jobs != job:
            continue

        if len(
                glob(history_file)
            ) > 0:
            print('done')
            continue
                
        tries = 0
        
        while tries < 5:

            model = get_model(
                option.arch,
                it_lim=it_lim,
                image_size=image_size,
                num_classes = option.num_classes,
                second=True,
                degree1=option.degree,
                factor=option.factor
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=getattr(losses,option.loss)
            )

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath= checkpoint_dir,
                    save_weights_only=True,
                    verbose = True,
                    save_best_only=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, verbose=1,patience=5),
                tf.keras.callbacks.TerminateOnNaN()
            ]

            history = model.fit(
                gen_batch_train,
                epochs=50,
                steps_per_epoch=100,
                validation_data=gen_batch_val,
                validation_steps=10,
                shuffle=False,
                use_multiprocessing=True,
                callbacks=callbacks,
                workers=1
            )
            
            if not np.isnan(history.history['val_loss'][-1]):
                np.save(
                    history_file,
                    np.array([history.history['loss'],history.history['val_loss']])
                )
                break
            else:
                tries += 1