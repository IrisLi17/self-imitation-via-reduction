# Self-Imitation via Reduction
Paper: Solving Compositional Reinforcement Learning Problems via Task Reduction

### Get Started
Prerequisite: 

* Ubuntu 16.04
* CUDA 10.0
* [MuJoCo](http://www.mujoco.org/) version 2.0. You can obtain a license and download the binaries from its website.
* [Conda](https://docs.conda.io/en/latest/miniconda.html)

Install:

Run ``conda env create -f environment.yml``. You may refer to [Troubleshooting](https://github.com/openai/mujoco-py/blob/master/README.md#troubleshooting) if you have problems installing ``mujoco-py``.

### How to Run
The scripts ``exp_push.sh``, ``exp_fetchstack.sh``, ``exp_masspoint.sh`` contain the commands for running different algorithms in *Push*, *Stack* and *Maze* scenarios respectively.

