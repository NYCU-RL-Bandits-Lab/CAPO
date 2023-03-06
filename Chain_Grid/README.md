# READ ME
Our experiment chain is conducted on the foundation of [rlx](https://github.com/dasayan05/rlx)
In order to reproduce the experiment, the helper script for running experiment can be located in the folder `run_grid` and `run_chain`.
The usage for the scripts is:
`./run_{algo_name}.sh $seed $max_episode $max_horizon $env_size`

Horizon and max episode can be set to large numbers (we were using 1000 and 10000), where the `env_size` depends on the result you want to reproduce.