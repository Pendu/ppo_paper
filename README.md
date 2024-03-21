# Solving a Real-world optimization problem using Proximal Policy Optimization combined with Curriculum Learning and Reward Engineering ♻️

![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.7-306998)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
![dependencies status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)

## 🖊 Info
### Example render of the environment during evaluation
<p align="center">
<img src= "https://github.com/Pendu/ContainerGym_Prefinal/blob/2c3589ef8c90c77832ccc0808fc7aafa6eec1713/example.gif" width="80%" height="80%"/>
</p>

## 📚 Setup
### Pre-requisites
* Python >=3.8.0,<3.10
* Conda
### Conda
* Follow instructions : [Installer link](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)

## 🤖 Installation

Create a conda virtual environment and run the following commands

```{bash}
conda create -n myenv python=3.8.8
conda activate myenv
pip install -r requirements.txt
```

##  📊 Reproduce results from the paper
```
python reproduce_results_paper.py
```

## Sample training for phase-1 (#TODO)
```
python train.py --train True --total-timesteps=30000 --bunkers 1 2 3 5 6 7 8 9 12 13 14 --seed 1 --track-wandb=False --max-episode-length 25 --log-dir ./results_training/setp1/ --gamma
 0.99 --wandb-project-name prefinal_mulbunk_5 --envlogger=False --envlogger-freq 2000 --track-local=True --number-of-presses=2 --use-min-rew True --batch-size 64 --filename-suffix baseline_origgaus_run5 --NA-Vf 512 512 --NA-Pf 512 512 --n-steps 6144 --CL-step 1
```

## 🎭 Support and Contributions

Feel free to ask questions. Posting in [Github Issues]( https://github.com/Pendu/ContainerGym_Prefinal/issues) and PRs are also welcome.
