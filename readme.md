# Policy Gradient Bayesian Robust Optimization for Imitation Learning


<p align="center">
  <a href="https://arxiv.org/abs/2106.06499">View on ArXiv</a> |
  <a href="https://sites.google.com/view/pg-broil/home">Project Website</a>
</p>


If you find this repository is useful in your research, please cite the paper:
```
@inproceedings{javed2021pgbroil,
  title={Policy Gradient Bayesian Robust Optimization},
  author={Javed, Zaynah  and Brown, Daniel S. and Sharma, Satvik and Zhu,  Jerry and Balakrishna, Ashwin and Petrik, Marek and Dragan, Anca D. and Goldberg, Ken },
  booktitle = {Proceedings of the 38th International Conference on Machine Learning, {ICML}},
  year={2021}
}
```

## Installation

Dependencies for the CartPole environment can be installed by installing Open AI Gym. Dependencies for the Reacher environment can be installed by pip installing the modified version of the dm_control package and an Open AI gym wrapper for the package which are both included in the included source. 

This code repo builds on the [OpenAI Spinning Up gitrepo](https://spinningup.openai.com/en/latest/user/installation.html). First follow the instructions to install:

```
conda create -n pgbroil python=3.6
conda activate pgbroil
pip install -e .
```

Also install dm_control suite
```
pip install dm_control dmc2gym
```



## Running Policy Gradient BROIL:


To run BROIL VPG:
In the pg-broil/spinup/algos/pytorch/vpg directory run
```bash
python broil_vpg2.py --env (CartPole-v0, PointBot-v0)
```
Pass in BROIL arguments using
```bash
--broil_lambda
--broil_alpha
--risk_metric (cvar,erm)
```
(Can also pass in all of the arguments listed in the original spinningup vpg.py code, ie: seed, epochs, env_name, policy_lr, etc)

To run BROIL PPO:
In the pg-broil/spinup/algos/pytorch/ppo directory:

First create a folder (broil_dataX) and 3 subfolders (results, visualizations, PointBot_networks) within the PPO directory. Make sure to rename "broil_dataX" on line 557 so files are saved in the right folder. Then run the following command for a grid_search over PointBot-v0:
```bash
chmod +x broil_ppo_grid.sh
./broil_ppo_grid.sh
```
To run BROIL PPO on an environment:
```bash
python broil_ppo.py --env (CartPole-v0, PointBot-v0, reacher)
```
Pass in BROIL arguments using
```bash
--broil_lambda
--broil_alpha
--risk_metric (cvar,erm)
```



Once the above command finishes running, you can plot the cvar and expected value graphs by going into the broil_dataX folder and run:

```
python select_data_to_graph.py
```

Make sure to create the right folders (line 41) and metric (line 15: 'cvar', 'expected_return') to graph the wanted data.


## Evaluation of pretrained policy

To run evaluation of pretrained policy to get risk and return for plotting pareto frontier:

Pretrain a policy using spinningup then use evaluate_policy.py and give it the save path and the env name and it will run 100 policy evaluations and return the expected return under the posterior the cvar.

Note you need to pass in the max horizon for the MDP to initialize the buffer size. Max horizon should be the max number of steps possible in the environment.

```
python pg-broil/algos/pytorch/evaluation/evaluate_policy.py --save_path spinningup/data/installtest/installtest_s0 --env CartPole-v0 --num_rollouts 100 --max_horizon 200
```
## Bayesian REX

To generate the Bayesian REX posteriors used in the paper for TrashBot run:

```
python pg-broil/algos/pytorch/rex/brex/brex_basic.py --features_dir demonstrations/trashbot_demos --normalize
```

To generate the Bayesian REX posterior used in the appendix for the reacher environment run:
```
python pg-broil/algos/pytorch/rex/brex/brex_basic.py --features_dir demonstrations/reacher_easy_demos --env reacher --normalize
```

This will also print out the mean and MAP reward. To run a PBRL baseline, take the outputted MAP and put it into the spinup/examples/pytorch/broil_rtg_pg_v2/reacher_reward_utils.py or spinup/examples/pytorch/broil_rtg_pg_v2/pointbot_reward_utils.py (depending on the environment you want to run) and set the self.posterior to an array containing just 1.

## Pointbot Navigation

For the Pointbot Navigation environment:
Change the reward function posterior by going to spinningup/spinup/examples/pytorch/broil_rtg_pg_v2/pointbot_reward_utils.py and changing the self.penalties and self.posterior attributes. The default attributes are the ones used for the paper.

To recreate figures 2b and 2e go to spinningup/spinup/experiments/grapher.py and change the name_of_grid_search variable to maze_ppo_cvar**. To recreate figure 5 in the appendix change the name_of_grid_search variable to maze_ppo_erm**. Then run 
```
python grapher.py
```
and check in the respective folder for the trajectory visualization and in maze_ppo_erm**/visualizations or maze_ppo_cvar**/visualizations for the pareto frontier.


## Trashbot
### Demonstrator

To create demonstrations for the TrashBot environment first go to the spinningup/spinup/envs/pointbot_const.py and change the constants to create the trash environment in the paper which is given in the comments. Then run demonstrator.py

```
python demonstrator.py
```
Use mouse clicks to apply x and y force to the bot. If the bot is close enough to the trash another piece of trash will randomly spawn in the environment. The demonstrations will be created in pairs. The first demonstration is the good demo while the second is the bad demo. Then go to the demonstrations folder where there a folder will be created with visualizations of both demonstrations, .txt files with the states and actions, and the pickle files used for the algorithms.


### Behavioral Cloning 

To run BC for the TrashBot environment first go to the spinningup/spinup/envs/pointbot_const.py and change the constants to create the trash environment in the paper which is given in the comments. Then add the pkl files of the demos created by demonstrator.py (the demos used in the paper are already inside the demos folder). Then go to the spinningup/spinup/algos/pytorch/ppo directory and run the command:
```
python bc.py --env PointBot-v0
```
A folder will be created in the working directory which will have example rollouts and average statistics for trash collected and steps in the gray region.

### GAIL

To run GAIL for the TrashBot environment first go to the pg-broil/spinup/envs/pointbot_const.py and change the constants to create the trash enviornment in the paper which is given in the comments. Then add the pkl files of the demos created by demonstrator.py (the demos used in the paper are already inside the demos folder). Then go to the pg-broil/spinup/algos/pytorch/PyTorch-RL directory and run the command:
```
python gail/gail_gym.py --env PointBot-v0
```
A folder will be created in the working directory which will have example rollouts, rollouts for each epoch, the reward graph over epochs, and average statistics for trash collected and steps in the gray region.

### RAIL

To run RAIL (Risk Averse Imitation Learning) for the TrashBot environment, you can similarly run it as GAIL by passing in additional arguments.
```
python gail/gail_gym.py --env PointBot-v0 --cvar --alpha 0.9 --lamda [0, infinity)
```
