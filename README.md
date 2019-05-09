# Adversarial Autoencoders (AAE)
 This is a Tensorflow 2.0 implementation of [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644) by [Alireza Makhzani](http://www.alireza.ai/) et al. (ICLR 2016). This repository contains reproduce of several experiments mentioned in the paper.
 
## Requirements
- Python 3
- [TensorFlow 2.0+](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)

## Results
### Unsupervised AAE deterministic
Latent space
![Latent space](figs/unsupervised_aae_deterministic/latent.png)
Reconstruction
![Latent space](figs/unsupervised_aae_deterministic/reconstruction.png)
Sampled
![Latent space](figs/unsupervised_aae_deterministic/sampling.png)

### Unsupervised AAE deterministic convolutional
Latent space
![Latent space](figs/unsupervised_aae_deterministic_convolutional/latent.png)
Reconstruction
![Latent space](figs/unsupervised_aae_deterministic_convolutional/reconstruction.png)
Sampled
![Latent space](figs/unsupervised_aae_deterministic_convolutional/sampling.png)

### Unsupervised AAE deterministic convolutional using WGAN loss function
Latent space
![Latent space](figs/unsupervised_aae_deterministic_convolutional_wasserstein/latent.png)
Reconstruction
![Latent space](figs/unsupervised_aae_deterministic_convolutional_wasserstein/reconstruction.png)
Sampled
![Latent space](figs/unsupervised_aae_deterministic_convolutional_wasserstein/sampling.png)

### Unsupervised AAE deterministic
Latent space
![Latent space](figs/supervised_aae_deterministic/latent.png)
Reconstruction
![Latent space](figs/supervised_aae_deterministic/reconstruction.png)
Sampled
![Latent space](figs/supervised_aae_deterministic/style.png)

### Unsupervised AAE deterministic convolutional
Latent space
![Latent space](figs/supervised_aae_deterministic_convolutional/latent.png)
Reconstruction
![Latent space](figs/supervised_aae_deterministic_convolutional/reconstruction.png)
Sampled
![Latent space](figs/supervised_aae_deterministic_convolutional/style.png)
