from setuptools import setup, find_packages
from hotham import __version__


with open('README.md') as f:
    long_description = f.read()


setup(
    name="hotham",
    version=__version__,
    author="Liang Zhixin",
    author_email="171830553@smail.nju.edu.cn",
    url="https://github.com/liangzhixin-202169/Hot-Ham",
    packages=find_packages(),
    python_requires=">=3.8.5",
    install_requires=[
        "numpy",     
        "pyyaml",
        "torch>=2.4.0",
        "torch_geometric>=2.4.0",
        "e3nn==0.5.1",
        "ase",
        "h5py"
    ],
    license="MIT",
    description="Hot-Ham: High-order Tensor machine-learning Hamiltonian",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["hotham = hotham.entrypoints.main:main"]},
)
