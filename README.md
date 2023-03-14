#  Coordinate Ascent Policy Optimization (CAPO)
**Coordinate Ascent for Off-Policy RL with Global Convergence Guarantees**

Hsin-En Su, Yen-Ju Chen, [Ping-Chun Hsieh](https://pinghsieh.github.io/), Xi Liu

*26th Conference on Artificial Intelligence and Statistics (AISTATS 2023)*

[\[Paper\]](https://arxiv.org/abs/2212.05237) [Poster (TBD)] [Slide (TBD)]

## Read Me
Our CAPO code adapt the code of [stable-baseline3](https://github.com/DLR-RM/stable-baselines3) and [rl-baseline3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) as the foundation, all credit belongs to the owner of these two repos, however CAPO/offpac is implemented by us.

Note that although the code is working, it is out-of-sync from both repo, to reproduce the experiment, one should make sure that python is using the rl-baseline3-zoo and stable-baselines provided in this folder. (Either add `sys.path.insert(0, "/path/to/dependencies")` or modifiy the python env PATH.)

Note that hyperparams in rl-baseline3-zoo/hyperparams is not guaranteed to be up-to-date with the one in our paper, please follow the setting described in Appendix.

## MinAtar
Please install by following the instruction of [MinAtar](https://github.com/kenjyoung/MinAtar) before running.

## Running
An example to run CAPO on Minatar/Freeway-v0:
`python3 train.py --algo offpac -params KL:True --env MinAtar/Freeway-v0`
In this code base, CAPO share the fundamental with offpac, setting KL:True will run CAPO where setting KL:False will run offpac.

An example to run offpac on Minatar/Freeway-v0:
`python3 train.py --algo offpac --env MinAtar/Freeway-v0`

## Citation

If you find our repository helpful to your research, please cite our paper:

```
@article{su2022coordinate,
  title={Coordinate Ascent for Off-Policy RL with Global Convergence Guarantees},
  author={Su, Hsin-En and Chen, Yen-Ju and Hsieh, Ping-Chun and Liu, Xi},
  journal={arXiv preprint arXiv:2212.05237},
  year={2022}
}
```

## TODO
<!-- - modified Citation to AISTATS2023 -->
<!-- - modified the paper link to AISTATS2023 -->
<!-- - update poster -->
- update readme
- clean rl-baselines3-zoo, stable-baselines3
