<p align="center">
    <img src="assets/keras_gan.png" width="480"\>
</p>

GAN

实现最原始的，基于多层感知器构成的生成器和判别器，组成的生成对抗网络模型（Generative Adversarial）。

参考论文：《Generative Adversarial Networks》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

 

AC-GAN

实现辅助分类-生成对抗网络（Auxiliary Classifier Generative Adversarial Network）。

参考论文：《Conditional Image Synthesis With Auxiliary Classifier GANs》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py

 

BiGAN

实现双向生成对抗网络（Bidirectional Generative Adversarial Network）。

参考论文：《Adversarial Feature Learning》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py

 

BGAN

实现边界搜索生成对抗网络（Boundary-Seeking Generative Adversarial Networks）。

参考论文：《Boundary-Seeking Generative Adversarial Networks》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/bgan/bgan.py

 

CC-GAN

实现基于上下文的半监督生成对抗网络（Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks）。

参考论文：《Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/ccgan/ccgan.py

 

CoGAN

实现耦合生成对抗网络（Coupled generative adversarial networks）。

参考论文：《Coupled Generative Adversarial Networks》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/cogan/cogan.py

 

CycleGAN

实现基于循环一致性对抗网络（Cycle-Consistent Adversarial Networks）的不成对的Image-to-Image 翻译。

参考论文：《Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py

 

DCGAN

实现深度卷积生成对抗网络（Deep Convolutional Generative Adversarial Network）。

参考论文：《Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

 

DualGAN

实现对偶生成对抗网络（DualGAN),基于无监督的对偶学习进行Image-to-Image翻译。

参考论文：《DualGAN: Unsupervised Dual Learning for Image-to-Image Translation》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/dualgan/dualgan.py

 

InfoGAN

实现的信息最大化的生成对抗网络（InfoGAN），基于信息最大化生成对抗网络的可解释表示学习。

参考论文：《InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/infogan/infogan.py

 

LSGAN

实现最小均方误差的生成对抗网络（Least Squares Generative Adversarial Networks）。

参考论文：《Least Squares Generative Adversarial Networks》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/lsgan/lsgan.py

 

SGAN

实现半监督生成对抗网络（Semi-Supervised Generative Adversarial Network）。

参考论文：《Semi-Supervised Learning with Generative Adversarial Networks》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/sgan/sgan.py

 

WGAN

实现 Wasserstein GAN。

参考论文：《Wasserstein GAN》

代码地址：https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py

















## Keras-GAN
Collection of Keras implementations of Generative Adversarial Networks (GANs) suggested in research papers. These models are in some cases simplified versions of the ones ultimately described in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. Contributions and suggestions of GAN varieties to implement are very welcomed.

<b>See also:</b> [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#ac-gan)
    + [Adversarial Autoencoder](#adversarial-autoencoder)
    + [Bidirectional GAN](#bigan)
    + [Boundary-Seeking GAN](#bgan)
    + [Conditional GAN](#cgan)
    + [Context-Conditional GAN](#cc-gan)
    + [Context Encoder](#context-encoder)
    + [Coupled GANs](#cogan)
    + [CycleGAN](#cyclegan)
    + [Deep Convolutional GAN](#dcgan)
    + [DiscoGAN](#discogan)
    + [DualGAN](#dualgan)
    + [Generative Adversarial Network](#gan)
    + [InfoGAN](#infogan)
    + [LSGAN](#lsgan)
    + [Pix2Pix](#pix2pix)
    + [PixelDA](#pixelda)
    + [Semi-Supervised GAN](#sgan)
    + [Super-Resolution GAN](#srgan)
    + [Wasserstein GAN](#wgan)
    + [Wasserstein GAN GP](#wgan-gp)     

## Installation
    $ git clone https://github.com/eriklindernoren/Keras-GAN
    $ cd Keras-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations   
### AC-GAN
Implementation of _Auxiliary Classifier Generative Adversarial Network_.

[Code](acgan/acgan.py)

Paper: https://arxiv.org/abs/1610.09585

#### Example
```
$ cd acgan/
$ python3 acgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/acgan.gif" width="640"\>
</p>

### Adversarial Autoencoder
Implementation of _Adversarial Autoencoder_.

[Code](aae/aae.py)

Paper: https://arxiv.org/abs/1511.05644

#### Example
```
$ cd aae/
$ python3 aae.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/aae.png" width="640"\>
</p>

### BiGAN
Implementation of _Bidirectional Generative Adversarial Network_.

[Code](bigan/bigan.py)

Paper: https://arxiv.org/abs/1605.09782

#### Example
```
$ cd bigan/
$ python3 bigan.py
```

### BGAN
Implementation of _Boundary-Seeking Generative Adversarial Networks_.

[Code](bgan/bgan.py)

Paper: https://arxiv.org/abs/1702.08431

#### Example
```
$ cd bgan/
$ python3 bgan.py
```

### CC-GAN
Implementation of _Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks_.

[Code](ccgan/ccgan.py)

Paper: https://arxiv.org/abs/1611.06430

#### Example
```
$ cd ccgan/
$ python3 ccgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/ccgan.png" width="640"\>
</p>

### CGAN
Implementation of _Conditional Generative Adversarial Nets_.

[Code](cgan/cgan.py)

Paper:https://arxiv.org/abs/1411.1784

#### Example
```
$ cd cgan/
$ python3 cgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/cgan.gif" width="640"\>
</p>

### Context Encoder
Implementation of _Context Encoders: Feature Learning by Inpainting_.

[Code](context_encoder/context_encoder.py)

Paper: https://arxiv.org/abs/1604.07379

#### Example
```
$ cd context_encoder/
$ python3 context_encoder.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/context_encoder.png" width="640"\>
</p>

### CoGAN
Implementation of _Coupled generative adversarial networks_.

[Code](cogan/cogan.py)

Paper: https://arxiv.org/abs/1606.07536

#### Example
```
$ cd cogan/
$ python3 cogan.py
```

### CycleGAN
Implementation of _Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_.

[Code](cyclegan/cyclegan.py)

Paper: https://arxiv.org/abs/1703.10593

<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan.png" width="640"\>
</p>

#### Example
```
$ cd cyclegan/
$ bash download_dataset.sh apple2orange
$ python3 cyclegan.py
```   

<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan_gif.gif" width="640"\>
</p>


### DCGAN
Implementation of _Deep Convolutional Generative Adversarial Network_.

[Code](dcgan/dcgan.py)

Paper: https://arxiv.org/abs/1511.06434

#### Example
```
$ cd dcgan/
$ python3 dcgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/dcgan2.png" width="640"\>
</p>

### DiscoGAN
Implementation of _Learning to Discover Cross-Domain Relations with Generative Adversarial Networks_.

[Code](discogan/discogan.py)

Paper: https://arxiv.org/abs/1703.05192

<p align="center">
    <img src="http://eriklindernoren.se/images/discogan_architecture.png" width="640"\>
</p>

#### Example
```
$ cd discogan/
$ bash download_dataset.sh edges2shoes
$ python3 discogan.py
```   

<p align="center">
    <img src="http://eriklindernoren.se/images/discogan.png" width="640"\>
</p>

### DualGAN
Implementation of _DualGAN: Unsupervised Dual Learning for Image-to-Image Translation_.

[Code](dualgan/dualgan.py)

Paper: https://arxiv.org/abs/1704.02510

#### Example
```
$ cd dualgan/
$ python3 dualgan.py
```

### GAN
Implementation of _Generative Adversarial Network_ with a MLP generator and discriminator.

[Code](gan/gan.py)

Paper: https://arxiv.org/abs/1406.2661

#### Example
```
$ cd gan/
$ python3 gan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/gan_mnist5.gif" width="640"\>
</p>

### InfoGAN
Implementation of _InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets_.

[Code](infogan/infogan.py)

Paper: https://arxiv.org/abs/1606.03657

#### Example
```
$ cd infogan/
$ python3 infogan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/infogan.png" width="640"\>
</p>

### LSGAN
Implementation of _Least Squares Generative Adversarial Networks_.

[Code](lsgan/lsgan.py)

Paper: https://arxiv.org/abs/1611.04076

#### Example
```
$ cd lsgan/
$ python3 lsgan.py
```

### Pix2Pix
Implementation of _Image-to-Image Translation with Conditional Adversarial Networks_.

[Code](pix2pix/pix2pix.py)

Paper: https://arxiv.org/abs/1611.07004

<p align="center">
    <img src="http://eriklindernoren.se/images/pix2pix_architecture.png" width="640"\>
</p>

#### Example
```
$ cd pix2pix/
$ bash download_dataset.sh facades
$ python3 pix2pix.py
```   

<p align="center">
    <img src="http://eriklindernoren.se/images/pix2pix2.png" width="640"\>
</p>

### PixelDA
Implementation of _Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks_.

[Code](pixelda/pixelda.py)

Paper: https://arxiv.org/abs/1612.05424

#### MNIST to MNIST-M Classification
Trains a classifier on MNIST images that are translated to resemble MNIST-M (by performing unsupervised image-to-image domain adaptation). This model is compared to the naive solution of training a classifier on MNIST and evaluating it on MNIST-M. The naive model manages a 55% classification accuracy on MNIST-M while the one trained during domain adaptation gets a 95% classification accuracy.

```
$ cd pixelda/
$ python3 pixelda.py
```

| Method       | Accuracy  |
| ------------ |:---------:|
| Naive        | 55%       |
| PixelDA      | 95%       |

### SGAN
Implementation of _Semi-Supervised Generative Adversarial Network_.

[Code](sgan/sgan.py)

Paper: https://arxiv.org/abs/1606.01583

#### Example
```
$ cd sgan/
$ python3 sgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/sgan.png" width="640"\>
</p>

### SRGAN
Implementation of _Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_.

[Code](srgan/srgan.py)

Paper: https://arxiv.org/abs/1609.04802

<p align="center">
    <img src="http://eriklindernoren.se/images/superresgan.png" width="640"\>
</p>


#### Example
```
$ cd srgan/
<follow steps at the top of srgan.py>
$ python3 srgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/srgan.png" width="640"\>
</p>

### WGAN
Implementation of _Wasserstein GAN_ (with DCGAN generator and discriminator).

[Code](wgan/wgan.py)

Paper: https://arxiv.org/abs/1701.07875

#### Example
```
$ cd wgan/
$ python3 wgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/wgan2.png" width="640"\>
</p>

### WGAN GP
Implementation of _Improved Training of Wasserstein GANs_.

[Code](wgan_gp/wgan_gp.py)

Paper: https://arxiv.org/abs/1704.00028

#### Example
```
$ cd wgan_gp/
$ python3 wgan_gp.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/imp_wgan.gif" width="640"\>
</p>
