# Binary Neural Networks with Feature Information Retention for Efficient Image Classification
This is the Pytorch implementation of FIRNet. 

Although binary neural networks (BNNs) enjoy extreme compression ratios, there are significant accuracy gap compared with full-precision models. Previous works propose various strategies to reduce the information loss induced by the binarization process, improving the performance of binary neural networks to some extent. However, in this letter, we argue that few studies try to alleviate this problem from the structure perspective, leading to inferior results. To this end, we propose a novel Feature Information Retention Network named FIRNet, which incorporates an extra path to propagate the untouched informative feature maps. Specifically, the FIRNet splits the input feature maps into two groups, one of which is fed into the normal layers and another kept untouched for information retention. Then we utilize the concatenation, shuffle and pooling operations to process these features efficiently. Finally, with minimal complexity increase, a FIR fusion layer is proposed to aggregate the features from two branches. Experimental results demonstrate that our proposed method achieves 1.0\% Top-1 accuracy improvement over the baseline model and outperforms other state-of-the-art BNNs on the ImageNet dataset.

## Dependencies
* Python 3.8.19
* Pytorch 1.9.0+cu111

## Training on CIFAR-10
```bash
cd cifar
bash run.sh
```
The training configs are as follows:
```bash
python -u main.py \
--gpus 1 \
--model resnet20_1w1a \
--results_dir ./result \
--data_path /data_path \
--dataset cifar10 \
--epochs 600 \
--lr 0.1 \
-b 128 \
-bt 128 \
--lr_type cos \
--weight_decay 1e-4 \
--tau_min 0.85 \
--tau_max 0.99 \
```

## Training on ImageNet
```bash
cd imagenet
bash run.sh
```
The training configs are as follows:
```bash
python -u main.py \
--gpus 2,3 \
--model resnet18_1w1a \
--results_dir ./result \
--data_path /data_path \
--dataset imagenet \
--epochs 200 \
--lr 0.1 \
-b 512 \
-bt 256 \
--lr_type cos \
--weight_decay 1e-4 \
--tau_min 0.85 \
--tau_max 0.99 \
```