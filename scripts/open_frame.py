import numpy as np
import cv2

def gaussian(frame_num,var=1,CROP = 256):
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    img = img - np.amin(img)
    gauss = np.random.normal(0,var,img.shape)
    gauss = gauss.reshape(img.shape[0],img.shape[1])
    img_gauss = np.clip(np.copy(img) + gauss,0,255)
    img_gauss = np.asarray(img_gauss,dtype = np.float32)
    
    input_channel = img/np.amax(img)
    img_gauss = np.clip(img_gauss/np.amax(img),0,1)
    
    return np.array([input_channel, img_gauss])


def poisson(frame_num,var=1,CROP = 256):
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    img = img - np.amin(img)
    input_channel = img/np.amax(img)
    poisson_noise = np.sqrt(img) * np.random.normal(0, var, img.shape)
    
    poisson = np.clip(np.copy(img) + poisson_noise,0,255)
    poisson = np.clip(poisson/np.amax(img),0,1)
    
    
    return [input_channel,poisson]


def sp(frame_num,var=0.01,CROP = 256):
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    img = img - np.amin(img)
    input_channel = img/np.amax(img)
    salty = np.random.choice([True,False],img.shape,p=[var,1-var])
    peppery = np.random.choice([True,False],img.shape,p=[var,1-var])
    
    sp = np.copy(input_channel)
    sp[salty] = 1
    sp[peppery] = 0
    
    
    return [input_channel,sp]
    
    
def inpainting(frame_num,var=0.5,CROP = 256):
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    img = img - np.amin(img)
    input_channel = img/np.amax(img)
    wx,wy = np.asarray(np.random.uniform(var/5,var,2)*img.shape,dtype='int')
    x,y = np.random.choice(img.shape[0]-1-wx),np.random.choice(img.shape[1]-1-wy)
    missing = np.copy(input_channel)
    missing[x:x+wx,y:y+wy] = 0
    return [input_channel, missing]
    
    
def deblurring(frame_num,var=1,CROP = 256):
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    img = img - np.amin(img)
    input_channel = img/np.amax(img)
    gausBlur = cv2.blur(input_channel, (int(var),int(var))) 
    gausBlur = gausBlur/np.amax(gausBlur)
    
    return [input_channel, gausBlur]

def uniform(frame_num,var=10,CROP=256):
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    img = img - np.amin(img)
    unif = np.random.uniform(-var,var,img.shape)
    unif = unif.reshape(img.shape[0],img.shape[1])
    img_gauss = np.clip(np.copy(img) + unif,0,255)
    img_gauss = np.asarray(img_gauss,dtype = np.float32)
    
    input_channel = img/np.amax(img)
    img_gauss = np.clip(img_gauss/np.amax(img),0,1)
    
    return np.array([input_channel, img_gauss])

def uniform_sign(frame_num,var=10,CROP=256):
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    img = img - np.amin(img)
    if var > 0:
        unif = np.random.uniform(0,var,img.shape)
    else:
        unif = np.random.uniform(var,0,img.shape)
    unif = unif.reshape(img.shape[0],img.shape[1])
    img_gauss = np.clip(np.copy(img) + unif,0,255)
    img_gauss = np.asarray(img_gauss,dtype = np.float32)
    
    input_channel = img/np.amax(img)
    img_gauss = np.clip(img_gauss/np.amax(img),0,1)
    
    return np.array([input_channel, img_gauss])
    
def open_frame(typ,var):


    if typ == "gaussian":
        return gaussian
    elif typ == "poisson":
        return poisson
    elif typ == "sp":
        return sp
    elif typ == "inpainting":
        return inpainting
    elif typ == "deblurring":
        return deblurring
    elif typ == "uniform":
        return uniform
    elif typ == "uniform_sign":
        return uniform_sign
    else:
        return None

