from setuptools import setup, find_packages
from hotham import __version__


with open('README.md') as f:
    long_description = f.read()


setup(
    name="hotham",
    version=__version__,
    author="Liang Zhixin, Wang Yunlong",
    author_email="171830553@smail.nju.edu.cn, yunlongwang@smail.nju.edu.cn",
    url="https://github.com/liangzhixin-202169/Hot-Ham",
    packages=find_packages(),
    python_requires=">3.10",
    install_requires=[
        "ase==3.25.0",
        "e3nn==0.6.0",
        "tqdm",
        "numpy==2.4.4",
        "scipy",
        "spglib",
        "torch>=2.4.0",
        "pyyaml",
        "matplotlib",
        "torch-geometric>=2.4.0",
    ],
    license="MIT",
    description="Hot-Ham: High-order Tensor machine-learning Hamiltonian",
    long_description=long_description,
    long_description_content_type="text/markdown",

    include_package_data=True,
    package_data={
        "hotham": [
            "utilities/*.pt",
            "utilities/*.npy",
        ]
    },
)
