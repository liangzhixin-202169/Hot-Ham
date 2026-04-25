# Hot-Ham: High-order Tensor machine-learning Hamiltonian
## 1.Introduction
**Hot-Ham** is developed by Prof. Jian Sun's group (https://sun.nju.edu.cn) at Nanjing Universityis. It a Python package designed for constructing E(3)-equivariant machine learning models to predict Density Functional Theory (DFT) Hamiltonians. The framework is based on message passing neural networks(MPNNs), using spherical tensors to represent E(3)-equivariant node and edge features. **Hot-Ham** utilizes local coordinate transformation and Gaunt tensor product to achieve efficient high-order spherical tensor products, which is critical to improve models' accuracy.

## 2.Current Features
- Building density functional theory Hamiltonian in the LCAO basis.
- Leveraging efficient Gaunt tensor products (with or without local coordinate transformation) to couple the node and edge equivariant features.
- Compensating for the lack of antisymmetric tensors in Gaunt tensor products via Clebsch-Gordan tensor product under local coordinate transformation(SO(2) convolution).

## 3.Requirements
### Python
The python version is recommended to be larger than 3.10, with following packages:
- numpy
- torch
- torch_geometric
- e3nn
- ase
- h5py
- json5
- tqdm
- pyyaml

### **Hot-Ham** installation
You can use pip:
```shell
pip install git+https://github.com/liangzhixin-202169/Hot-Ham.git
```

### Interface
Hot-Ham supports ABACUS and OpenMX.

#### ABACUS
#### OpenMX
`openmx_tools/patch` offers patch that directly outputs overlap matrix, and you need to place it in OpenMX's `source` directory when compiling.`openmx_tools/generate_dataset` provides script to extract Hamiltonian and overlap matrixes.

## 4.Usage
### Train
Hot-Ham can be trained with:
```shell
hotham train.json
```

## 5.Reference
[1] 1. Zhixin Liang, Yunlong Wang, Chi Ding, Junjie Wang, Hui-Tian Wang, Dingyu Xing and Jian Sun, Hot-Ham: an accurate and efficient E(3)-equivariant machine-learning electronic structures calculation framework, Chinese Physics Letters (2025). https://doi.org/10.1088/0256-307X/43/2/020704