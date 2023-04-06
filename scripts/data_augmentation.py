import numpy as np
import cv2
import tensorflow as tf
import open_frame as OF
from glob import glob




def augment(yx, crop=256, do_flips=True, do_rotate=True, do_scale=True):
    
    if do_flips:
        if np.random.uniform(0,1) > 0.5:
            if np.random.uniform(0,1) > 0.5:
                for i in range(len(yx)):
                    yx[i] = cv2.flip(yx[i],0)
            else:
                for i in range(len(yx)):
                    yx[i] = cv2.flip(yx[i],1)
 

    ch, cw = yx[0].shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cw/2,ch/2),np.random.rand()*360*float(do_rotate),
                            1+float(do_scale)*(np.random.uniform(-0.2,0.2)))
    for i in range(len(yx)):
        yx[i] = cv2.warpAffine(yx[i],rotation_matrix, (ch,cw),cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return yx
    
N_REPEAT_FRAME = 1

def sample_images(frame_nums):
    while True:
        try:
            var = np.random.choice([15,25,50])
            var += np.random.normal(0,10)
            var = np.clip(var,var_d,var_u)
            img = open_frame(np.random.choice(frame_nums),var,CROP)
        except Exception as e:
            print(f'Exception {e} on file')
            #continue
            break
            
        
            
        for n in range(N_REPEAT_FRAME):
            
            a =  augment(np.copy(img),crop = CROP)
            yield a
            
def get_data_generator(sampler):
    def get_data():
        while True:
            yx = next(sampler)
            
            x,y = yx[1],yx[0]
            x = np.expand_dims(x,axis=-1)
            y = np.expand_dims(y,axis=-1)
               
            if same:
                x = y
            yield x,y
                    
    return get_data

    
    
def get_generators(typ,var1_d,var1_u,BATCH_SIZE = 50, CROP1 = 256,
                  training=False,autoencoder=False):

    test = glob('../../images/test/*.jpg')
    train = glob('../../images/train/*.jpg')


    global open_frame
    open_frame = getattr(OF,typ)
    
    global var_d
    var_d = var1_d

    global var_u
    var_u = var1_u

    global CROP
    CROP = CROP1

    
    global same
    same = autoencoder
    
    

    dg_train = tf.data.Dataset.from_generator(
        get_data_generator(sample_images(train)),
        output_types=(tf.float32, tf.float32),
        output_shapes=((CROP, CROP, 1),(CROP, CROP, 1)) )

    dg_val = tf.data.Dataset.from_generator(
        get_data_generator(sample_images(test)),
        output_types=(tf.float32, tf.float32),
        output_shapes=((CROP, CROP, 1),(CROP, CROP, 1)) )

    gen_batch_train = dg_train.batch(BATCH_SIZE)
    gen_batch_val = dg_val.batch(BATCH_SIZE)

    return (gen_batch_train,gen_batch_val)

