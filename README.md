# Point-PlaneNet: *Plane kernel based convolutional neural network for point clouds analysis*
Created by <a href="https://www.researchgate.net/profile/Moein_Peyghambarzadeh" target="_blank">S.M. Moein Peyghambarzadeh</a>, 
<a href="https://www.researchgate.net/profile/Fatemeh_Azizmalayeri" target="_blank">Fatemeh Azizmalayeri</a>, 
<a href="http://basu.ac.ir/en/~khotanlou" target="_blank">Hassan Khotanlou</a>,
<a href="http://www.salarpour.com" target="_blank"> AmirSalarpour</a>,</br>
Digital Signal Processing 2020 (DSP 2020), https://www.sciencedirect.com/science/article/abs/pii/S1051200419301873

![prediction example](https://raw.githubusercontent.com/moeinp70/Point-Planenet/master/fig1.png)

### Introduction
we propose PlaneConv; a new layer to extract features from
local structures of point clouds by learning a set of planes in the Euclidean
space. The kernel for PlaneConv is a new operation that learns the parameters of the planes by measuring the distance between these planes and local
points. We use this layer as a building block in a PointNet like structure and
call this network PlaneNet. In a series of experiments including point cloud
classification, part segmentation and semantic segmentation, our architecture achieves competitive results, obtaining good accuracy with low number
of parameters.



## Installation
Requirements:
- Python 3
- CUDA 8.0 or higher
- [tensorflow 1.4 or higher](https://www.tensorflow.org/get_started/os_setup)
- Windows 10 and Ubuntu 16.04.

This code has been tested with Python 3.5, Tensorflow 1.2 and CUDA 8.0 on Ubuntu 16.04. The code have also been tested on Windows 10 with python 3.5-3.7, tensorflow 1.4(or higher), cuda 9 (or higher)  and cudnn 6.0 (or higher).
 notice ; If u want compile with tensorflow 2.0 please change import to:
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```   
### Citation
If you find our work useful in your research, Please cite this paper:

    @article{peyghambarzadeh2020point,
      title={Point-PlaneNet: Plane kernel based convolutional neural network for point clouds analysis},
      author={Peyghambarzadeh, SM Moein and Azizmalayeri, Fatemeh and Khotanlou, Hassan and Salarpour, Amir},
      journal={Digital Signal Processing},
      volume={98},
      pages={102633},
      year={2020},
      publisher={Elsevier}
    }

## References
Our released code heavily based on each methods original repositories as cited below:
* <a href="https://github.com/charlesq34/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017).
* <a href="https://github.com/charlesq34/pointnet2" target="_black">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017).
