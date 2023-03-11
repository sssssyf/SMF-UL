# SMF-UL
Perceiving Spectral Variation: Unsupervised Spectrum Motion Feature Learning for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-17, 2022, Art no. 5543817, doi: 10.1109/TGRS.2022.3221534.


# Requirements
imageio
matplotlib
scikit-image
easydict
torch==1.1.0
torchvision==0.3.0
path.py
opencv-python>=3.0,<4.0
fast_slic
tensorboardX
getopt
math
numpy
sys
scipy
tqdm
sklearn
time
datetime

# Usage

We provide a demo of the Salinas hyperspectral data (TargetData). You can add the source data by run the file of HSImage.py.
The initial model can be trained by run the file of train.py with the corresponding file of configuration. Besides, we also provide three types of trained model pretrained on different SourceData.  Please note that due to the randomness of the parameter initialization, the experimental results might have slightly different from those reported in the paper. Please refer to the paper for more details.

# License
Copyright (C) 2022 Yifan Sun

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.

# Citation
If the code is helpful to you, please give a star or fork and cite the paper. Thanks!
[1] @ARTICLE{9945998,
  author={Sun, Yifan and Liu, Bing and Yu, Xuchu and Yu, Anzhu and Gao, Kuiliang and Ding, Lei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Perceiving Spectral Variation: Unsupervised Spectrum Motion Feature Learning for Hyperspectral Image Classification}, 
  year={2022},
  volume={60},
  pages={1-17},
  doi={10.1109/TGRS.2022.3221534}}

# References
[1]  @inproceedings{liu2020learning,
   title = {Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation},
   author = {Liu, Liang and Zhang, Jiangning and He, Ruifei and Liu, Yong and Wang, Yabiao and Tai, Ying and Luo, Donghao and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
   booktitle = {IEEE Conference on Computer Vision and Pattern Recognition(CVPR)},
   year = {2020}
}
