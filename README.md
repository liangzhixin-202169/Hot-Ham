# Hot-Ham: High-order Tensor machine-learning Hamiltonian
## Introduction
**Hot-Ham** is a Python package designed for constructing E(3)-equivariant machine learning models to predict Density Functional Theory (DFT) Hamiltonians. The framework is based on message passing neural networks(MPNNs), using spherical tensors to represent E(3)-equivariant node and edge features. **Hot-Ham** utilizes local coordinate transformation and Gaunt tensor product to achieve efficient high-order spherical tensor products, which is critical to improve models' accuracy.

## Current Features
- Building density functional theory Hamiltonian in the LCAO basis.
- Leveraging efficient Gaunt tensor products (with or without local coordinate transformation) to couple the node and edge equivariant features.
- Compensating for the lack of antisymmetric tensors in Gaunt tensor products via Clebsch-Gordan tensor product under local coordinate transformation(SO(2) convolution).

## Requirements
### Python
The python version is recommended to be larger than 3.8.5, with following packages:
- Numpy
- Pytorch >= 1.10.2
- torch_geometric >= 2.4.0
- e3nn = 0.5.1
- ASE
- h5py

### OpenMX
The datasets are generated from OpenMX. `openmx_tools/patch` offers patch that directly outputs overlap matrix, and `openmx_tools/generate_dataset` provides script to extract Hamiltonian and overlap matrixes.

## Usage
### Train
Hot-Ham can be trained with:
```python
hotham/entrypoints/main.py train.json
```