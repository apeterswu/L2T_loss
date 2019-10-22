# Learning to Teach with Dynamic Loss Functions
This repo contains the simple demo for the NeurIPS-18 paper: [Learning to Teach with Dynamic Loss Functions](https://papers.nips.cc/paper/7882-learning-to-teach-with-dynamic-loss-functions.pdf). 
```
@inproceedings{wu2018learning,
  title={Learning to teach with dynamic loss functions},
  author={Wu, Lijun and Tian, Fei and Xia, Yingce and Fan, Yang and Qin, Tao and Jian-Huang, Lai and Liu, Tie-Yan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6466--6477},
  year={2018}
}
```
# Description
* Please note this is only a simple demo for the Mnist experiments based on Lenet. 
* Please note the algorithm in the demo is little different to the paper, but the main spirit is same. 
* The code is based on the Theano framework, which is somehow too old to directly apply this code. 

### Detailed Critical Codes
Refer to `loss_lenet_light_dynamic.py` for the detailed demo codes. The general comments are here:
* [Define the loss parameters and loss computation graph](https://github.com/apeterswu/L2T_loss/blob/master/loss_lenet_light_dynamic.py#L484)
* [Teacher model paramters](https://github.com/apeterswu/L2T_loss/blob/master/loss_lenet_light_dynamic.py#L514)
* [Define the updates of model training according to the loss function (jointly train model update)](https://github.com/apeterswu/L2T_loss/blob/master/loss_lenet_light_dynamic.py#L543)
* [Define the reverse mode training](https://github.com/apeterswu/L2T_loss/blob/master/loss_lenet_light_dynamic.py#L583)
* [Train student model with fixed loss function](https://github.com/apeterswu/L2T_loss/blob/master/loss_lenet_light_dynamic.py#L646)
* [Detailed reverse model training](https://github.com/apeterswu/L2T_loss/blob/master/loss_lenet_light_dynamic.py#L754)
```
The 'reverse model training' defines the updates of the teacher model, and the last one is the detailed reverse model update chains. 
```
