import numpy as np
import cv2
import tensorflow as tf
from open_frame import open_frame as OF
from glob import glob




def augment(yx, crop=256, do_flips=True, do_rotate=True, do_scale=False):
    
    if do_flips:
        if np.random.uniform(0,1) > 0.5:
            if np.random.uniform(0,1) > 0.5:
                yx[0] = cv2.flip(yx[0], 0)
                yx[1] = cv2.flip(yx[1], 0)
            else:
                yx[0] = cv2.flip(yx[0], 1)
                yx[1] = cv2.flip(yx[1], 1)
 
    
    ch, cw = yx[0].shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cw/2,ch/2),np.random.randint(-90,90),1)
    yx[0] = cv2.warpAffine(yx[0],rotation_matrix, (ch,cw),cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    yx[1] = cv2.warpAffine(yx[1],rotation_matrix, (ch,cw),cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return yx
    
N_REPEAT_FRAME = 1

def sample_images(frame_nums):
    while True:
        try:
            img = open_frame(np.random.choice(frame_nums),var,CROP)
        except Exception as e:
            print(f'Exception {e} on file')
            #continue
            break
        for n in range(N_REPEAT_FRAME):
            img = open_frame(np.random.choice(frame_nums),var,CROP)
            a =  augment(img, do_flips=True, do_rotate=True, do_scale=True,crop = CROP)
            yield a
            
def get_data_generator(sampler):
    def get_data():
        while True:
            yx = next(sampler)
            x,y = yx[1],yx[0]
            yield np.expand_dims(x,axis=-1),np.expand_dims(y,axis=-1)
        
    return get_data
    
    
def get_generators(typ,var1,BATCH_SIZE = 50, CROP1 = 256):

	test = glob('../../images/test/*.jpg')
	train = glob('../../images/train/*.jpg')

	global open_frame 
	open_frame = OF(typ,var1)
	
	global var
	var = var1
	
	global CROP
	CROP = CROP1
	
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


def get_data_classifier_borders(sampler):
    def get_data():
        while True:
            yx = next(sampler)
            x,y = yx[1],yx[0]
            divx = y[1:] - y[:-1]
            divy = y[:,1:] - y[:,:-1]
            div = cv2.resize(np.maximum(np.abs(divx[:,:-1]),np.abs(divy[:-1])),(CROP,CROP))>0.09
            div = np.asarray(div,dtype='float')
            yield np.expand_dims(x,axis=-1),np.expand_dims(div,axis=-1)
        
    return get_data


def get_classifiers_borders(typ,var1,BATCH_SIZE = 50, CROP1 = 256):

    test = glob('../../images/test/*.jpg')
    train = glob('../../images/train/*.jpg')

    global open_frame 
    open_frame = OF(typ,var1)

    global var
    var = var1

    global CROP
    CROP = CROP1

    dg_train = tf.data.Dataset.from_generator(
        get_data_classifier_borders(sample_images(train)),
        output_types=(tf.float32, tf.float32),
        output_shapes=((CROP, CROP, 1),(CROP, CROP, 1)) )

    dg_val = tf.data.Dataset.from_generator(
        get_data_classifier_borders(sample_images(test)),
        output_types=(tf.float32, tf.float32),
        output_shapes=((CROP, CROP, 1),(CROP, CROP, 1)) )

    gen_batch_train = dg_train.batch(BATCH_SIZE)
    gen_batch_val = dg_val.batch(BATCH_SIZE)
    
    return (gen_batch_train,gen_batch_val)
