# Hot-Ham: High-order Tensor machine-learning Hamiltonian
## 1. Introduction
**Hot-Ham** is an open-source Python package developed by Prof. Jian Sun's group (https://sun.nju.edu.cn) at Nanjing University. It uses E(3)-equivariant message-passing neural networks to construct electronic Hamiltonians in both LCAO and orthogonal tight-binding representations.

## 2. Current Features
- Model training for LCAO Hamiltonian prediction using reference DFT Hamiltonian data.
- Model training for orthogonal tight-binding Hamiltonian prediction using reference band structures.
- Efficient coupling of high-order equivariant features through optimized tensor product operations and convolution operations.
- Support for electronic property calculations from the predicted Hamiltonians.

## 3. Requirements
The python version is recommended to be newer than 3.10, with following packages:
- ase
- e3nn
- tqdm
- numpy
- scipy
- spglib
- pyyaml
- matplotlib
- torch
- torch-scatter
- torch-geometric

## 4. Installation
**1. Install `Hot-Ham` with pip**: `pip install git+https://github.com/liangzhixin-202169/Hot-Ham.git`

**2. Install `torch-scatter` manually according to your PyTorch and CUDA versions**: `pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`

## 5. Documentation and tests
Documentation and tests for Hot-Ham is available in [hotham-tests](hotham-tests).

## 6. Reference
[1] Zhixin Liang, Yunlong Wang, Chi Ding, Junjie Wang, Hui-Tian Wang, Dingyu Xing, and Jian Sun. Hot-Ham: an Accurate and Efficient E(3)-Equivariant Machine-Learning Electronic Structures Calculation Framework. *Chinese Physics Letters* **43**(2), 020704 (2026). https://doi.org/10.1088/0256-307X/43/2/020704

[2] Yunlong Wang, Zhixin Liang, Chi Ding, Junjie Wang, Zheyong Fan, Hui-Tian Wang, Dingyu Xing, and Jian Sun. GPUTB: Efficient machine learning tight-binding method for large-scale electronic properties calculations. *Computational Materials Today* **8**, 100039 (2025). https://doi.org/10.1016/j.commt.2025.100039

[3] Yunlong Wang, Zhixin Liang, Chi Ding, Junjie Wang, Zheyong Fan, Hui-Tian Wang, Dingyu Xing, and Jian Sun. GPUTB-2: An efficient E(3) network method for learning high-precision orthogonal Hamiltonian. *arXiv:2601.13656* (2026). https://doi.org/10.48550/arXiv.2601.13656
