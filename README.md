# [Collaborative continual learner with Active Forgetting (CAF)]() 

[![DOI](https://zenodo.org/badge/684337404.svg)](https://zenodo.org/badge/latestdoi/684337404)
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/lywang3081/CAF/blob/master/LICENSE)

------
This code includes the official implementation of our papers ([1] is the main subject):

[1][Incorporating Neuro-Inspired Adaptability for Continual Learning in Artificial Intelligence]()

[2] [CoSCL: Cooperation of Small Continual Learners is Stronger than a Big One](https://arxiv.org/abs/2207.06543) (ECCV22)

[3] [AFEC: Active Forgetting of Negative Transfer in Continual Learning](https://papers.nips.cc/paper/2021/hash/bc6dc48b743dc5d013b1abaebd2faed2-Abstract.html) (NeurIPS21)

Please refer to the following procedures to install environments, prepare datasets and run experiments.
The instruction of reproducing the main results in each figure is described in corresponding script files.
We present demo commands for a direct comparison of our approach and the corresponding baseline. We have further recorded a [demo video](https://www.youtube.com/watch?v=iNJ88zkK-mM) of running these commands.


## **Visual Classification Tasks**

### Requirements

- Python 3.6
- GPU 1080Ti / Pytorch 1.10.1+cu11.1 / CUDA 11.1

To install necessary packages:
```
pip install -r requirements.txt
```

### Download Dataset
- Omniglot : https://drive.google.com/file/d/1WxFZQyt3v7QRHwxFbdb1KO02XWLT0R9z/view?usp=sharing
- CUB-200-2011 : https://github.com/visipedia/tf_classification/wiki/CUB-200-Image-Classification
- Tiny-ImageNet : ```cd download``` ```source download_tinyimgnet.sh```
- CORe50 : Download and unzip the [original data](https://vlomonaco.github.io/core50/) into the ``CORe50/data/core50_128x128`` folder. 

### Execution Command
The commands to run most of the experiments are included in **script_classification.sh**.
The innate diversity background can be selected by ```--mcl mcl-h / mcl-m / mcl-l```.
For a direct comparison, here we present demo commands for our approach (high diversity, narrowed width) with EWC as the default continual learner. 

For small-scale images:

```
# S-CIFAR-100
# EWC
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --experiment s_cifar100 --approach ewc --lamb 10000 --seed 0
# CAF
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --experiment s_cifar100 --approach ewc_caf --mcl mcl-h --lamb 10000 --lamb_kld 0.02 --lamb_af 1e-8 --use_sigmoid --seed 0

# R-CIFAR-100
# EWC
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --experiment r_cifar100 --approach ewc --lamb 10000 --seed 0
# CAF
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --experiment r_cifar100 --approach ewc_caf --mcl mcl-h --lamb 10000 --lamb_kld 0.02 --lamb_af 1e-8 --use_sigmoid --seed 0

# R-CIFAR-10/100
# EWC
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --experiment r_cifar10_100 --approach ewc --lamb 10000 --seed 0
# CAF
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --experiment r_cifar10_100 --approach ewc_caf --mcl mcl-h --lamb 10000 --lamb_kld 0.02 --lamb_af 1e-8 --use_sigmoid --seed 0

# Omniglot
cd dat
mkdir binary_omniglot
# EWC
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --experiment omniglot --approach ewc --lamb 500000 --seed 0
# CAF
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --experiment omniglot --approach ewc_caf --mcl mcl-h --lamb 500000 --lamb_kld 0.05 --lamb_af 1e-12 --use_sigmoid --seed 0
```

**Note:** Using the sigmoid function after the output of each learner brings further improvements to CAF, which can be implemented by adding ```--use_sigmoid``` to the above commands.


For large-scale images:

```
cd LargeScale_Image

# CUB-200-2011
# EWC
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --dataset CUB200 --trainer ewc --lamb 5 --tasknum 10 --seed 0
# CAF
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --dataset CUB200 --trainer ewc_caf --lamb 5 --lamb_kld 1e-3 --lamb_af 1e-7 --tasknum 10 --seed 0

# Tiny-ImageNet
# EWC
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --dataset tinyImageNet --trainer ewc --lamb 320 --tasknum 10 --seed 0
# CAF
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --dataset tinyImageNet --trainer ewc_caf --lamb 320 --lamb_kld 1e-3 --lamb_af 1e-6 --tasknum 10 --seed 0
```

For CORe50:

```
cd CORe50

# EWC
CUDA_VISIBLE_DEVICES=0 python3 core50.py --save_path results/ --data_path data/ --cuda yes --n_epochs 40 --use 1.0 --model ewc --temperature 5 --lamb 1e2 --n_val 0.2 --data_file core50 --inner_steps 1 --n_meta 2 --n_memories 0 --lr 0.003 --beta 0.03 --n_runs 5 --batch_size 32 --n_tasks 10 --temperature 5
# CAF
CUDA_VISIBLE_DEVICES=0 python3 core50.py --save_path results/ --data_path data/ --cuda yes --n_epochs 40 --use 1.0 --model ewc_caf --temperature 5 --lamb 1e2 --lamb1 1e-1 --lamb2 1e-5 --n_val 0.2 --data_file core50 --inner_steps 1 --n_meta 2 --n_memories 0 --lr 0.003 --beta 0.03 --n_runs 5 --batch_size 32 --n_tasks 10 --temperature 5
```


## **Atari Reinforcement Tasks**

### Requirements

- Python 3.6
- Pytorch 1.10.1+cu11.1 / CUDA 11.1
- ​[OpenAI Gym](https://github.com/openai/gym#installation), [Baselines](https://github.com/openai/baselines)

**Note:** The experimental environment for reinforcement learning is built based on [pytorch-a2c-ppo-acktr-gaail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). 
The implementation of Gym needs to further specify: ```pip install atari-py==0.2.5```

### Execution Command
The commands to run most of the experiments are included in **script_Atari.sh**.
For a direct comparison, here we present demo commands for our approach (high diversity, narrowed width) with EWC as the default continual learner. 


```
# Fine-tuning (for normalization)
CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --approach fine-tuning --seed 0 --date 220815  
# EWC
CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --approach ewc --reg_lambda 100000 --seed 0 --date 220815 
# CAF
CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --approach ewc_caf --reg_lambda 100000 --kld_lambda 0.01 --af_lambda 1e-5 --seed 0 --date 220815 
```


## **Citation**

If you find this code helpful to your work, please cite our paper(s):

```bibtex
@inproceedings{wang2022coscl,
  title={Coscl: Cooperation of small continual learners is stronger than a big one},
  author={Wang, Liyuan and Zhang, Xingxing and Li, Qian and Zhu, Jun and Zhong, Yi},
  booktitle={European Conference on Computer Vision},
  pages={254--271},
  year={2022},
  organization={Springer}
}

@article{wang2021afec,
  title={Afec: Active forgetting of negative transfer in continual learning},
  author={Wang, Liyuan and Zhang, Mingtian and Jia, Zhongfan and Li, Qian and Bao, Chenglong and Ma, Kaisheng and Zhu, Jun and Zhong, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={22379--22391},
  year={2021}
}
```
