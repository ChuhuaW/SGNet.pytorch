# Pytorch Implementation for Stepwise Goal-Driven Networks for Trajectory Prediction (RA-L/ICRA2022)



## Installation

# Cloning

We use part of the dataloader in Trajectron++, so we include [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) as a submodule. 
```
git clone --recurse-submodules git@github.com:ChuhuaW/SGNet.pytorch.git
```

# Environment

* Install conda environment from yml file

```
conda env create --file SGNet_env.yml
```

# Data

* JAAD and PIE
JAAD and PIE can be downloaded from https://github.com/ykotseruba/JAAD and https://github.com/aras62/PIE, respectively. Creating symlinks from the dataset path to ```./data```

```
ln -s path/to/dataset/ ./data/
```

* ETH/UCY
We follow [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) to preprocess data splits for the ETH and UCY datasets in this repository. Please refer to their repository for instruction. After the data is generated, please create symlinks from the dataset path to ```./data```

```
ln -s path/to/dataset/ ./data/
```


## Training

### Stochastic prediction

* Training on JAAD dataset:
```
cd SGDNet.Pytorch
python tools/jaad/train_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset JAAD --model SGNet_CVAE
```

* Training on PIE dataset:
```
cd SGDNet.Pytorch
python tools/pie/train_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset PIE --model SGNet_CVAE
```

* Training on ETH/UCY dataset:
```
cd SGDNet.Pytorch
python tools/ethucy/train_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset ETH --model SGNet_CVAE
python tools/ethucy/train_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset HOTEL --model SGNet_CVAE
python tools/ethucy/train_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset UNIV --model SGNet_CVAE
python tools/ethucy/train_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset ZARA1 --model SGNet_CVAE
python tools/ethucy/train_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset ZARA2 --model SGNet_CVAE
```

### Deterministic prediction

* Training on JAAD dataset:
```
cd SGDNet.Pytorch
python tools/jaad/train_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset JAAD --model SGNet
```

* Training on PIE dataset:
```
cd SGDNet.Pytorch
python tools/pie/train_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset PIE --model SGNet
```

* Training on ETH/UCY dataset:
```
cd SGDNet.Pytorch
python tools/ethucy/train_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset ETH --model SGNet
python tools/ethucy/train_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset HOTEL --model SGNet
python tools/ethucy/train_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset UNIV --model SGNet
python tools/ethucy/train_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset ZARA1 --model SGNet
python tools/ethucy/train_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset ZARA2 --model SGNet
```

## Evaluation

### Stochastic prediction

* Evaluating on JAAD dataset:
```
cd SGDNet.Pytorch
python tools/jaad/eval_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset JAAD --model SGNet_CVAE --checkpoint path/to/checkpoint
```

* Evaluating on PIE dataset:
```
cd SGDNet.Pytorch
python tools/pie/eval_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset PIE --model SGNet_CVAE --checkpoint path/to/checkpoint
```

* Evaluating on ETH/UCY dataset:
```
cd SGDNet.Pytorch
python tools/ethucy/eval_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset ETH --model SGNet_CVAE --checkpoint path/to/checkpoint
python tools/ethucy/eval_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset HOTEL --model SGNet_CVAE --checkpoint path/to/checkpoint
python tools/ethucy/eval_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset UNIV --model SGNet_CVAE --checkpoint path/to/checkpoint
python tools/ethucy/eval_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset ZARA1 --model SGNet_CVAE --checkpoint path/to/checkpoint
python tools/ethucy/eval_cvae.py --gpu $CUDA_VISIBLE_DEVICES --dataset ZARA2 --model SGNet_CVAE --checkpoint path/to/checkpoint
```

### Deterministic prediction

* Evaluating on ETH/UCY dataset:
[ETH/UCY checkpoints](https://drive.google.com/drive/folders/1FCudihx-dmns-lh61uOcOD5uIWaKdKh8?usp=sharing)

```
cd SGDNet.Pytorch
python tools/ethucy/eval_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset ETH --model SGNet --checkpoint path/to/checkpoint
python tools/ethucy/eval_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset HOTEL --model SGNet --checkpoint path/to/checkpoint
python tools/ethucy/eval_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset UNIV --model SGNet --checkpoint path/to/checkpoint
python tools/ethucy/eval_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset ZARA1 --model SGNet --checkpoint path/to/checkpoint
python tools/ethucy/eval_deterministic.py --gpu $CUDA_VISIBLE_DEVICES --dataset ZARA2 --model SGNet --checkpoint path/to/checkpoint
```

[JAAD/PIE checkpoints](https://drive.google.com/drive/folders/1SskmNtf9FMn4azAxIfKXcYUgAEuVKNgR?usp=sharing)

## Citation

```
@ARTICLE{9691856,
  author={Wang, Chuhua and Wang, Yuchen and Xu, Mingze and Crandall, David J.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Stepwise Goal-Driven Networks for Trajectory Prediction}, 
  year={2022}}
```
```diff
- Rank 3rd on nuScences prediction task at 6th AI Driving Olympics, ICRA 2021
```
The source code and pretrained models will be made availble. Stay tuned.
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-goal-driven-networks-for-trajectory/trajectory-prediction-on-ethucy)](https://paperswithcode.com/sota/trajectory-prediction-on-ethucy?p=stepwise-goal-driven-networks-for-trajectory)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-goal-driven-networks-for-trajectory/trajectory-prediction-on-jaad)](https://paperswithcode.com/sota/trajectory-prediction-on-jaad?p=stepwise-goal-driven-networks-for-trajectory)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stepwise-goal-driven-networks-for-trajectory/trajectory-prediction-on-pie)](https://paperswithcode.com/sota/trajectory-prediction-on-pie?p=stepwise-goal-driven-networks-for-trajectory)



