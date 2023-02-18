import numpy as np
import cv2
import tensorflow as tf
import open_frame as OF
from glob import glob




def augment(yx, crop=256, do_flips=True, do_rotate=True, do_scale=False):
    
    if do_flips:
        if np.random.uniform(0,1) > 0.5:
            if np.random.uniform(0,1) > 0.5:
                for i in range(len(yx)):
                    yx[i] = cv2.flip(yx[i],0)
            else:
                for i in range(len(yx)):
                    yx[i] = cv2.flip(yx[i],1)
 
    if do_rotate:
        ch, cw = yx[0].shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cw/2,ch/2),np.random.randint(-90,90),1)
        for i in range(len(yx)):
            yx[i] = cv2.warpAffine(yx[i],rotation_matrix, (ch,cw),cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return yx
    
N_REPEAT_FRAME = 1

def sample_images(frame_nums,model):
    while True:
        try:
            if train_var:
                var = np.random.choice([10,20,50])
            else:
                #var = abs(var_u - np.random.exponential(var_d))
                var = np.random.uniform(var_d,var_u)
            img = open_frame(np.random.choice(frame_nums),var,CROP)
        except Exception as e:
            print(f'Exception {e} on file')
            #continue
            break
            
        #img = open_frame(np.random.choice(frame_nums),var,CROP)
        r = np.random.uniform(0,1)
        if r > 0.5:
            if model != 0:
                img[1] = np.squeeze(model(np.array([np.expand_dims(img[1],axis=-1)])))
            
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
            if pre:
                x[1] = tf.nn.conv2d(np.array([x[1]]),kernel,[1,1,1,1],"SAME")
            yield x,y
                    
    return get_data

    
    
def get_generators(typ,var1_d,var1_u,model=0,BATCH_SIZE = 50, CROP1 = 256,pre_smoothing=False,size=5,sigma=1,
                  training=False):

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
    
    global pre
    pre = pre_smoothing
    
    global train_var
    train_var = training
    
    if pre:
        mesh = np.meshgrid(np.arange(size),np.arange(size))
        center = size//2
        kernel1 = (np.power(mesh[0] - center,2) + np.power(mesh[1] - center,2))
        kernel1 = np.exp(-kernel1/(4*sigma))
        kernel1 = kernel1/(4*np.pi*sigma)
        kernel1 = np.expand_dims(np.expand_dims(kernel1,-1),-1)
        
        global kernel
        kernel = kernel1
    

    dg_train = tf.data.Dataset.from_generator(
        get_data_generator(sample_images(train,model)),
        output_types=(tf.float32, tf.float32),
        output_shapes=((CROP, CROP, 1),(CROP, CROP, 1)) )

    dg_val = tf.data.Dataset.from_generator(
        get_data_generator(sample_images(test,model)),
        output_types=(tf.float32, tf.float32),
        output_shapes=((CROP, CROP, 1),(CROP, CROP, 1)) )

    gen_batch_train = dg_train.batch(BATCH_SIZE)
    gen_batch_val = dg_val.batch(BATCH_SIZE)

    return (gen_batch_train,gen_batch_val)

