# Solving a Real-world optimization problem using Proximal Policy Optimization combined with Curriculum Learning and Reward Engineering ♻️

![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.7-306998)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
![dependencies status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)

## 🖊 Info
### Example render of the ContainerGym environment during evaluation
<p align="center">
<img src= "https://github.com/Pendu/ContainerGym_Prefinal/blob/2c3589ef8c90c77832ccc0808fc7aafa6eec1713/example.gif" width="80%" height="80%"/>
</p>

## 📚 Setup
### Pre-requisites
* Python >=3.8.0,<3.10
* Conda
### Install conda
```{bash}
bash Anaconda-latest-Linux-x86_64.sh
```

## 🤖 Installation

Clone the repository and run the following.

```{bash}
git clone https://github.com/Pendu/ppo_paper_draft.git
cd ppo_paper_draft
```

Create a conda virtual environment and run the following commands

```{bash}
conda create -n myenv python=3.8.8
conda activate myenv
pip install -r requirements.txt
```

## Reproduce results from the paper
```
python reproduce_results_paper.py
```
