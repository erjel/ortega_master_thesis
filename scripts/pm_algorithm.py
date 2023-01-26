import numpy as np
import cv2
import tensorflow as tf

def anisodiff(img,niter=1,lambd=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):

    '''
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
    '''
    
    if option == 1:
        def f(lamb,b):
            return tf.math.exp(-1* (np.power(lamb,2))/(np.power(b,2)))
        
    if option == 2:
        def f(lamb,b):
            return 1./(1.+np.power(lamb/b,2))
        
        
    img_new = np.zeros(img.shape) 
    for t in range(niter): 
        dx = img[:-2,1:-1] - img[1:-1,1:-1] 
        dy = img[2:,1:-1] - img[1:-1,1:-1] 
        dz = img[1:-1,2:] - img[1:-1,1:-1] 
        dw = img[1:-1,:-2] - img[1:-1,1:-1] 
        
        
        img_new[1:-1,1:-1] = img[1:-1,1:-1] +gamma * (f(dx,lambd)*dx + f (dy,lambd)*dy + f (dz,lambd)*dz + f (dw,lambd)*dw) 
        img = img_new 
        
    return img_new
    
def anisodiff_nn(img,niter=1,lambd=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):

    img = img.astype('float32')
    imgout = img.copy()

    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
    lambd = cv2.resize(lambd,np.squeeze(deltaS).shape)
    lambd = np.expand_dims(lambd,axis=-1)

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

            

    return imgout
