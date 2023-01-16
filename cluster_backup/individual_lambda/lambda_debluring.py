def anisodiff(img,niter=1,lambd=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):

    img = img.astype('float32')
    imgout = img.copy()

    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    if ploton:

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in np.arange(1,niter):

        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        if 0<sigma:
            deltaSf=flt.gaussian_filter(deltaS,sigma);
            deltaEf=flt.gaussian_filter(deltaE,sigma);
        else: 
            deltaSf=deltaS;
            deltaEf=deltaE;

        if option == 1:
            gS = np.exp(-(deltaSf/lambd)**2.)/step[0]
            gE = np.exp(-(deltaEf/lambd)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaSf/lambd)**2.)/step[0]
            gE = 1./(1.+(deltaEf/lambd)**2.)/step[1]
        elif option == 3:
            gS = lambd/step[0]
            gE = lambd/step[1]

        E = gE*deltaE
        S = gS*deltaS

        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]


        imgout += gamma*(NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()

            

    return imgout

def open_frame(frame_num,var=1):
    CROP = 256
    input_path = frame_num
    
    img = np.asarray(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2GRAY), dtype=np.float32)
    img = cv2.resize(img,(CROP,CROP))
    input_channel = img/np.amax(img)
    gausBlur = cv2.blur(input_channel, (var,var)) 
    gausBlur = gausBlur/np.amax(gausBlur)
    
    return input_channel, gausBlur





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
   
    option = int(sys.argv[2])
    CROP = 256
    
    path = "/gpfs/soma_fs/home/ortega/scripts/PeronaMalik/thesis"
    
    test = glob(f'{path}/images/test/*.jpg')
    train = glob(f'{path}/images/train/*.jpg')
    val = glob(f'{path}/images/val/*.jpg')

    lamb_var = []
    for var in tqdm(np.arange(2,14,1)):
        er = []
        for lambd in np.arange(0.001,100,0.5):
            er.append([])
            for i in range(20):
                im = open_frame(np.random.choice(train),var=var)
                pm = anisodiff(im[1],niter=10,lambd=lambd,gamma=0.1,step=(1.,1.),sigma=0, option=option,ploton=False)
                er[-1].append(np.sqrt(np.sum(np.power(np.subtract(im[0],pm),2))))

        er = np.mean(er,axis=-1)
        er = np.convolve(er,np.ones(5)/5,mode='same')

        lamb_var.append(er)
        
    np.save(f'/gpfs/soma_fs/home/ortega/scripts/PeronaMalik/thesis/11_oct/individual_lambda/lambdas/debluring_{option}.npy',lamb_var)