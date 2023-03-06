# Read Me
Our CAPO code adapt the code of [stable-baseline3](https://github.com/DLR-RM/stable-baselines3) and [rl-baseline3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) as the foundation, all credit belongs to the owner of these two repos, however CAPO/offpac is implemented by us.

Note that although the code is working, it is out-of-sync from both repo, to reproduce the experiment, one should make sure that python is using the rl-baseline3-zoo and stable-baselines provided in this folder. (Either add `sys.path.insert(0, "/path/to/dependencies")` or modifiy the python env PATH.)

Note that hyperparams in rl-baseline3-zoo/hyperparams is not guaranteed to be up-to-date with the one in our paper, please follow the setting described in Appendix.

# MinAtar
Please install by following the instruction of [MinAtar](https://github.com/kenjyoung/MinAtar) before running.

# Running
An example to run CAPO on Minatar/Freeway-v0:
`python3 train.py --algo offpac -params KL:True --env MinAtar/Freeway-v0`
In this code base, CAPO share the fundamental with offpac, setting KL:True will run CAPO where setting KL:False will run offpac.

An example to run offpac on Minatar/Freeway-v0:
`python3 train.py --algo offpac --env MinAtar/Freeway-v0`