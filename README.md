# Hot-Ham: High-order Tensor machine-learning Hamiltonian
## 1.Introduction
**Hot-Ham** is an open-source Python package developed by Prof. Jian Sun's group (https://sun.nju.edu.cn) at Nanjing University. It uses E(3)-equivariant message-passing neural networks to construct electronic Hamiltonians in both LCAO and orthogonal tight-binding representations. 

The models combine atom and distance embeddings with spherical harmonics to build and update equivariant node and edge features. For efficient high-order feature coupling, the LCAO model combines Gaunt tensor products with SO(2) convolution, while the orthogonal tight-binding model combines Fourier convolution with SO(2) convolution. 

## 2.Current Features
- Model training for LCAO Hamiltonian prediction using reference DFT Hamiltonian data.
- Model training for orthogonal tight-binding Hamiltonian prediction using reference band structures.
- Efficient coupling of high-order equivariant features through Gaunt tensor products or Fourier convolution, combined with SO(2) convolution.
- Support for electronic property calculations from the predicted LCAO Hamiltonians, including band structures, density of states, and optical conductivity.

## 3.Requirements
### Python
The python version is recommended to be newer than 3.10, with following packages:
- ase
- e3nn
- h5py
- tqdm
- json5
- numpy
- pyyaml
- matplotlib
- torch
- torch-scatter
- torch-geometric

## 4.Installation
You can use pip:
```shell
pip install git+https://github.com/liangzhixin-202169/Hot-Ham.git
```

After that, install `torch-scatter` according to your PyTorch and CUDA versions manually.

If you plan to use the additional analysis tools, we also recommend installing `scipy` and `spglib`:


## 5.Documentation
Documentation for Hot-Ham is available through the [Hot-Ham skill](hotham-skills/SKILL.md).

## 6.Reference
[1] Zhixin Liang, Yunlong Wang, Chi Ding, Junjie Wang, Hui-Tian Wang, Dingyu Xing and Jian Sun, Hot-Ham: an accurate and efficient E(3)-equivariant machine-learning electronic structures calculation framework, Chinese Physics Letters (2025). https://doi.org/10.1088/0256-307X/43/2/020704

[2] Yunlong Wang, Zhixin Liang, Chi Ding, Junjie Wang, Zheyong Fan, Hui-Tian Wang, Dingyu Xing and Jian Sun, GPUTB: Efficient machine learning tight-binding method for large-scale electronic properties calculations, Computational Materials Today, 8, 100039 (2025). https://doi.org/10.1016/j.commt.2025.100039

[3] Yunlong Wang, Zhixin Liang, Chi Ding, Junjie Wang, Zheyong Fan, Hui-Tian Wang, Dingyu Xing and Jian Sun, GPUTB-2: An efficient E(3) network method for learning high-precision orthogonal Hamiltonian, arXiv:2601.13656 (2026). https://doi.org/10.48550/arXiv.2601.13656
