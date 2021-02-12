Running Policy Gradient BROIL:
==================================
To run BROIL VPG:
In the spinningup/spinup/algos/pytorch/vpg directory run
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
In the spinningup/spinup/algos/pytorch/ppo directory:

First create a folder (broil_dataX) and 3 subfolders (results, visualizations, PointBot_networks) within the PPO directory. Make sure to rename "broil_dataX" on line 476 so files are saved in the right folder. Then run the following command for a grid_search over PointBot-v0:
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

python3 select_data_to_graph.py

Make sure to create the right folders (line 41) and metric (line 15: 'cvar', 'expected_return') to graph the wanted data.




Default Arguments are defined in lines 507-522
File saving file directories are defined in lines 466-501
Alpha, Lambda, pi_lr, and vf_lr parameter arrays are defined in lines 539-546 to run multiple parameters
I also changed train_pi_iters from 80 to 40 on line 111.

===========================================

To run evaluation of pretrained policy to get risk and return for plotting pareto frontier

pretrain policy using spinninup then use evaluate_policy.py and give it the save path and the env name and it will run 100 policy evaluations and return the expected return under the posterior the cvar

Note you need to pass in the max horizon for the mdp to initialize the buffer size. max horizon should be the max number of steps possible in the environment.

```
python spinup/algos/pytorch/evaluation/evaluate_policy.py --save_path /home/dsbrown/code/spinningup/data/installtest/installtest_s0 --env CartPole-v0 --num_rollouts 100 --max_horizon 200
```
===========================================
For the Pointbot Navigation enviornment:
Change the reward function posterior by going to spinup/examples/pytorch/broil_rtg_pg_v2/pointbot_reward_utils.py and changing the self.penatilies and self.postierior attributes. The default attributes are the ones used for the paper.

To recreate figure 2b and 2e go to spinup/experiments/grapher.py and change the name_of_grid_search variable to maze_ppo_cvar**. To recreate figure 5 in the appendix change the name_of_grid_search variable to maze_ppo_erm**. Then run 
```
python grapher.py
```
and check in the respective folder for the trajecotry visuzalization and in maze_ppo_erm**/visualziaitons or maze_ppo_cvar**/visualziaitons for the pareto frontier.

To train some policies with differnt values of alpha and lambda go to spinup/algos/pytorch/ppo/broil_ppo_grid.sh and change values in the script. To run ERM instead of CVaR use the risk_metric flag. Then run:
```
./broil_ppo_grid.sh
```

Welcome to Spinning Up in Deep RL!
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```
