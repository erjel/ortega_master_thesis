# master thesis

This is a repository showing the current state of my master thesis' code and although it contains the trained weights of the nn's I use,
it doesn't contain the training scripts since those give no further informations and was done in another computer. The idea behind the existence
of this repository is so anyone can pull it and start using it right away. It is a work in progress, so it is quite prone to change. 

The topic for my thesis is image reconstruction through learning functions and the idea is to corrupt the BSDS500 data set [1] with a given type of noise which could be Gaussian, Poisson, salt and pepper, blur or inpainting.
Afterwards I will use deep learning and the anisotropic diffusion equation given by

$$ \partial_t u =\text{div}(f(t,||u||)\nabla u)$$

to get rid of the added corruption.

## Corrupting the images

Due to the limited amount of available images in the BSDS500, I built a generator so I could apply data augmentation at the same time.
The generator's inputs are the type of noise I want to add, the variance, kernel size or proportion of image loss depending on the type I'm adding
and whether I want to apply vertical and horizontal flips or rotations as a data augmentation feature.

## Applying the anisotropic diffusion equation

The reason to choose anisotropic diffusion to do the reconstruction are the well-known properties of this algorithm when it comes to smooth images 
while at the same time preserve edges, meaning that in theory, we would be able to smooth out the images' peaks generated by the added noise 
and keep the information our original image had.

On the other hand, the way I'm trying to improve this algorithm is by using neural networks, usually based in a Unet arquitecture, to find
the function $f$ in the equation which would produce the best results in the denoising task.

## Description of the folders

What you will find in the "images" folders is the BSDS500 database witht the result of merging training and validation sets and labelling them
as "train" and the original "test" set. I did this so I could have more images to train and becaise I can just generate as many testing images as
I need from the corresponding pool.

In order to avoid redundance on the code, I created the "scripts" folder, so I could just import my functions and architectures in the folders
I needed.

As for the "11_oct" folder, I put the different arquitectures I have been trying out, each of them producing functions with different properties,
namely being an optimized version of a canonical function used in the Perona-Malik algorithm, being a spline (currently just order 1) or monotone decreasing.

Finally, the "summary_141222.pdf" is meant to give an idea of how the reconstruction process iwas going on up till 14.12.2022


## References

[1] https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
