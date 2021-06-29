import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
import time
from tqdm import tqdm
import os, sys
from json import JSONEncoder
import json
from spinup.envs.pointbot_const import *
from spinup.algos.pytorch.ppo.visualize_pointbot import *
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from spinup.rewards.pointbot_reward_utils import PointBotReward
from spinup.rewards.pointbot_reward_brex import PointBotRewardBrex
from spinup.rewards.reacher_reward_utils import ReacherReward
from spinup.rewards.cartpole_reward_utils import CartPoleReward
from spinup.rewards.boxing_reward_utils import BoxingReward
from spinup.rewards.boxing_reward_brex import BoxingRewardBrex

from spinup.rewards.cvar_utils import cvar_enumerate_pg
import dmc2gym

torchify = lambda x: torch.FloatTensor(x).to(torch.device('cpu'))


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    #rew_dim is the dimensionality of the reward function posterior
    def __init__(self, obs_dim, act_dim, num_rew_fns, size, gamma=0.99, lam=0.95):
        self.num_rew_fns = num_rew_fns
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.rew_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.ret_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.val_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.posterior_returns = []
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=None):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        if last_val is None:
            last_val = np.zeros(self.num_rew_fns, dtype=np.float32)

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.vstack((self.rew_buf[path_slice], last_val))
        vals = np.vstack((self.val_buf[path_slice], last_val))

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        #TODO: see if there is a way to vectorize this
        for i in range(self.num_rew_fns):
            self.adv_buf[path_slice,i] = core.discount_cumsum(deltas[:,i], self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        # also store the cumulative returns for BROIL CVaR calculation
        self.posterior_returns.append(np.sum(rews, axis=0))

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        #TODO: see if we can vectorize this and figure out multithreading
        for i in range(self.num_rew_fns):
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[:,i])
            self.adv_buf[:,i] = (self.adv_buf[:,i] - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                        adv=self.adv_buf, logp=self.logp_buf, p_returns=self.posterior_returns)
        self.posterior_returns = []
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppo(env_fn, reward_dist, broil_risk_metric='cvar', actor_critic=core.BROILActorCritic, ac_kwargs=dict(), render=False, seed=0,
        steps_per_epoch=4000, epochs=50, broil_lambda=0.5, broil_alpha=0.95, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters =40, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl = .01,
        clip_ratio = .2, logger_kwargs=dict(), save_freq=10, clone=False, num_demos=0, train_pi_BC_iters=100,
        expert_fcounts = None):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        broil_lambda (float): amount to blend between maximizing expected return (1.0)
            and maximizing CVaR (0.0). Always between 0 and 1.
        broil_alpha (float): risk sensitivity in range [0,1) for computing alpha-CVaR
            higher alpha is more risk sensitive.
        gamma (float): Discount factor. (Always between 0 and 1.)
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.


    def get_reward_distribution(args, reward_dist, env, next_o, env_rew, action):
        if args.env == 'CartPole-v0':
            rew_dist = reward_dist.get_reward_distribution(next_o)
        elif args.env == 'PointBot-v0':
            rew_dist = reward_dist.get_reward_distribution(env, next_o)
        elif args.env == 'reacher':
            rew_dist = reward_dist.get_reward_distribution(env)
        elif args.env == 'Boxing-ram-v0':
            rew_dist = reward_dist.get_reward_distribution(env_rew)
        else:
            raise NotImplementedError("Unsupported Environment")

        return rew_dist

    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create BROIL actor-critic module
    num_rew_fns = reward_dist.posterior.size
    ac = actor_critic(env.observation_space, env.action_space, num_rew_fns, **ac_kwargs)
    #max_ep_len = env._max_episode_steps

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, num_rew_fns, local_steps_per_epoch, gamma, lam)

    mean_r = torch.zeros(num_rew_fns)
    std_r = torch.zeros(num_rew_fns)
    if clone:
        demo_obs, demo_acs = env.get_demos(num_demos)

#### compute BROIL policy gradient loss (robust version)
    def compute_broil_weights(batch_rets, weights):
        '''batch_returns: list of numpy arrays of size num_rollouts x num_reward_fns
           weights: list of weights, e.g. advantages, rewards to go, etc by reward function over all rollouts,
            size is num_rollouts*ave_rollout_length x num_reward_fns
        '''
        #inputs are lists of numpy arrays
        #need to compute BROIL weights for policy gradient and convert to pytorch

        #first find the expected on-policy return for current policy under each reward function in the posterior
        exp_batch_rets = np.mean(batch_rets.numpy(), axis=0)
        posterior_reward_weights = reward_dist.posterior
        
        if expert_fcounts is not None:
            reward_weights = reward_dist.get_posterior_weight_matrix()
            expert_returns = np.dot(reward_weights, expert_fcounts)
            exp_batch_rets -= expert_returns #baseline with what expert would have gotten under each reward function

        #calculate sigma and find either the conditional value at risk or entropic risk measure given the current policy
        if broil_risk_metric == "cvar":
            #Calculate policy gradient for conditional value at risk

            sigma, cvar = cvar_enumerate_pg(exp_batch_rets, posterior_reward_weights, broil_alpha)
           # print("sigma = {}, cvar = {}".format(sigma, cvar))

            #compute BROIL policy gradient weights
            total_rollout_steps = len(weights)
            broil_weights = np.zeros(total_rollout_steps, dtype=np.float64)
            for i, prob_r in enumerate(posterior_reward_weights):
                if sigma >= exp_batch_rets[i]:
                    w_r_i = broil_lambda + (1 - broil_lambda) / (1 - broil_alpha)
                else:
                    w_r_i = broil_lambda
                broil_weights += prob_r * w_r_i * np.array(weights)[:,i]


            return broil_weights,cvar

        elif broil_risk_metric == "erm":
            #calculate policy gradient for entropic risk measure
            erm = -1.0 / broil_alpha * np.log(np.dot(posterior_reward_weights, np.exp(-broil_alpha * exp_batch_rets)))

            #compute stable weighted soft-max
            exponents = -broil_alpha * exp_batch_rets
            z = exponents - max(exponents)
            numerators = np.exp(z)
            denominator = np.dot(numerators, posterior_reward_weights)
            softmax_probs = numerators / denominator

            #compute BROIL policy gradient weights for ERM
            total_rollout_steps = len(weights)
            erm_weights = broil_lambda * np.ones(len(posterior_reward_weights)) + (1-broil_lambda) * softmax_probs

            broil_weights = np.array(weights) * erm_weights
            broil_weights = np.dot(broil_weights, posterior_reward_weights)

            return broil_weights,erm


        else:
            print("Risk metric not implemented!")
            raise NotImplementedError


    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old, batch_returns = data['obs'], data['act'], data['adv'], data['logp'], data['p_returns']
        # Use advantage estimates to compute BROIL policy gradient weights
        broil_weights, cvar = compute_broil_weights(batch_returns, adv)
        weights = torch.as_tensor(broil_weights, dtype=torch.float32)
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * weights
        loss_pi = -(torch.min(ratio * weights, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info, cvar

    # Maximize log_pi for acs in demos
    def policy_BC_loss(demo_obs, demo_acs):
        # Policy loss
        loss_pi = 0
        for i in range(len(demo_acs)):
            obs_batch = torchify(np.array(demo_obs[i][:-1]))
            acs_batch = torchify(np.array(demo_acs[i]))
            pi, logp = ac.pi(obs_batch, acs_batch)
            loss_pi += (-(logp).mean())
        return loss_pi

    # Set up function for computing value loss for a particular reward function value estimator
    # def compute_loss_v(data, reward_index):
    #     obs, ret = data['obs'], data['ret'][:,i]
    #     return ((ac.v(obs) - ret)**2).mean()

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        new_mean_r = torch.mean(ret, dim=0)
        new_std_r = torch.std(ret, dim=0)
        
        ret = (ret - new_mean_r) / new_std_r
        mse = (ac.v(obs) - ret)**2

        mean_r[:] = new_mean_r
        std_r[:] = new_std_r
        return (mse).mean()


    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old, cvar = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info, cvar = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old), Risk=cvar, ExpectedRet=np.dot(np.mean(data['p_returns'].numpy(), axis=0), reward_dist.posterior))
        running_cvar.append(cvar)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    cvar_list = []
    wc_ret_list = []
    bc_ret_list = []
    ret_list = []
    obstacle_list = []
    trajectories_x = []
    trajectories_y = []
    trash_trajectories = []
    num_trashes = []
    obs_times = []
    # Behavior clone policy on some initial demos
    if clone and num_demos > 0:
        for i in range(train_pi_BC_iters):
            pi_optimizer.zero_grad()
            BC_loss = policy_BC_loss(demo_obs, demo_acs)
            BC_loss.backward()
            pi_optimizer.step()

    # Main loop: collect experience in env and update/log each epoch
    for epoch in tqdm(range(epochs)):
        first_rollout = True
        running_cvar = []
        total_reward_dist = np.zeros(num_rew_fns)
        running_ret = 0
        num_runs = 0
        obstacles = 0
        num_constraint_violations = 0
        num_episodes = 0
        constraint_violated = False
        for t in tqdm(range(local_steps_per_epoch)):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            #Unnormalize Values
            v = (v * std_r.numpy()) + mean_r.numpy()

            next_o, r, d, info = env.step(a)
            if args.env == 'Shelf-v0' or args.env == 'reacher': # Check if you ever violate a constraint in this episode
                if info['constraint']:
                    constraint_violated = True
            rew_dist = get_reward_distribution(args, reward_dist, env, next_o, r, a)
            total_reward_dist += rew_dist.flatten()
            running_ret += r
            ep_ret += r
            ep_len += 1
            if args.env == 'PointBot-v0':
                #print(env.obstacle(next_o))
                obstacles += int(env.obstacle(next_o))



            # save and log
            buf.store(o, a, rew_dist, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if render and first_rollout:
                env.render()
                time.sleep(0.01)

            if terminal or epoch_ended:
                if constraint_violated:
                    num_constraint_violations += 1
                num_episodes += 1

                constraint_violated = False
                first_rollout = False
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    buf.finish_path(v)
                else:
                    buf.finish_path()

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    # print(ep_ret)

                num_runs += 1

                if args.env == 'PointBot-v0':
                    last_trajectory = np.array(env.hist)
                    if TRASH:
                        num_trashes.append(len(env.current_trash_taken))
                        obs_times.append(obstacles)
                        obstacles = 0
                    if epoch == epochs - 1:
                        trajectories_x.append(last_trajectory[:, 0])
                        trajectories_y.append(last_trajectory[:, 2])
                        if TRASH:
                            env.current_trash_taken.append(env.next_trash)
                            trash_trajectories.append(env.current_trash_taken)

                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Store stuff for saving data
        ret_list.append(running_ret / float(num_runs))
        wc_ret_list.append(np.min(total_reward_dist) / float(num_runs))
        bc_ret_list.append(np.max(total_reward_dist / float(num_runs)))
        cvar_list.append(sum(running_cvar))
        obstacle_list.append(obstacles / float(num_runs))
        
        """print('True returns:', ret_list)
        print('Cvar: ', cvar_list)
        print('Worst case:', wc_ret_list)"""


        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Risk', average_only=True)
        logger.log_tabular('ExpectedRet', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        print("Frac Constraint Violations: %d/%d" % (num_constraint_violations, num_episodes))


    file_data = 'broil_data/'
    experiment_name = args.env + '_alpha_' + str(broil_alpha) + '_lambda_' + str(broil_lambda)

    metrics = {"conditional value at risk": ('_cvar', cvar_list),
               "true_return": ('_true_return', ret_list),
               "worst case return": ('_worst_case_return', wc_ret_list),
               "best case return": ('_best_case_return', bc_ret_list),
               "obstacle_collision": ('_obstacles', obstacle_list)}

    for metric, result in metrics.items():
        file_metric_description, results = result
        file_path = file_data + 'results/' + experiment_name + file_metric_description + '.txt'
        with open(file_path, 'w') as f:
            for item in results:
                f.write("%s\n" % item)

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)
    
    #Create PointBot Trajectory Visualization with Multiple Lambdas and Input for grapher.py
    if args.env == 'PointBot-v0':
        plt.ylim((env.grid[2], env.grid[3]))
        plt.xlim((env.grid[0], env.grid[1]))
        for i in range(4):
            x = trajectories_x[i]
            y = trajectories_y[i]
            if i == 0 and args.combiner:
                decodeddict = None
                if (os.path.exists(file_data + "states_combined.txt")):
                    f = open(file_data + "states_combined.txt", "r")
                    decodeddict = json.load(f)
                    f.close()
                a, b = str(broil_lambda) + "_x", str(broil_lambda) + "_y"
                temp = {a: trajectories_x, b: trajectories_y}
                if (os.path.exists(file_data + "states_combined.txt")):
                    os.remove(file_data + "states_combined.txt")
                f = open(file_data + "states_combined.txt", "w")
                if decodeddict == None:
                    json.dump(temp, f, cls=NumpyArrayEncoder)
                else:
                    decodeddict.update(temp)
                    json.dump(decodeddict, f, cls=NumpyArrayEncoder)
                f.close()

            plt.scatter([x[0]],[y[0]],  [6], '#00FF00', zorder=11)
            plt.scatter(x[1:], y[1:], len(x)*[6], zorder=9)
            if TRASH:
                for j in trash_trajectories[i]:
                    plt.scatter([j[0]],[j[1]], [25], zorder = 10, color = '#000000')
        
        #Create Single PointBot Visualization for lambda
        x_bounds = [obstacle.boundsx for obstacle in env.obstacle.obs]
        y_bounds = [obstacle.boundsy for obstacle in env.obstacle.obs]
        for i in range(len(x_bounds)):
            plt.gca().add_patch(patches.Rectangle((x_bounds[i][0], y_bounds[i][0]), width=x_bounds[i][1] - x_bounds[i][0], height=y_bounds[i][1] - y_bounds[i][0], fill=True, alpha=.5, linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3'))
        
        plt.savefig(file_data + 'visualizations/' + experiment_name + '.png')
        plt.clf()
        torch.save(ac.state_dict(), file_data + 'PointBot_networks/' + experiment_name + '.pt')
        env = gym.make(args.env)
        #visualize_policy(env, args.num_rollouts, ac, num_rew_fns, std_r, mean_r, reward_dist, local_steps_per_epoch, broil_alpha, file_data, args, max_ep_len, t, broil_lambda)
        if args.env == 'PointBot-v0':
            print(np.average(num_trashes[-100:]))
            print(np.std(num_trashes[-100:]))
            print(np.average(obs_times[-100:]))
            print(np.std(obs_times[-100:]))
    

    print(' Data from experiment: ', experiment_name, ' saved.')

if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--brex', type=bool, default=False)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--policy_lr', type=float, default=3e-4, help="learning rate for policy")
    parser.add_argument('--value_lr', type=float, default=1e-3)
    parser.add_argument('--risk_metric', type=str, default='cvar', help='choice of risk metric, options are "cvar" or "erm"' )
    parser.add_argument('--broil_lambda', type=float, default=1, help="blending between cvar and expret")
    parser.add_argument('--broil_alpha', type=float, default=0.95, help="risk sensitivity for cvar")
    parser.add_argument('--clone', action="store_true", help="do behavior cloning")
    parser.add_argument('--num_demos', type=int, default=0)
    parser.add_argument('--baseline_file', type=str, default='', help="file with feature counts of expert")
    parser.add_argument('--combiner', type=bool, default=False)
    parser.add_argument('--num_rollouts', type=int, default=0)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    if args.env == 'CartPole-v0':
        reward_dist = CartPoleReward()
    elif args.env == 'PointBot-v0':
        if args.brex:
            reward_dist = PointBotRewardBrex()
        else:
            reward_dist = PointBotReward()
    elif args.env == 'Shelf-v0':
        reward_dist = ShelfReward()
    elif args.env == 'reacher':
        reward_dist = ReacherReward()
    elif args.env == 'Boxing-ram-v0':
        if args.brex:
            reward_dist = BoxingRewardBrex()
        else:
            reward_dist = BoxingReward()
    else:
        raise NotImplementedError("Unsupported Environment")

    if args.env == 'reacher':
        env_fn = lambda: dmc2gym.make(domain_name='reacher', task_name='hard',seed=args.seed)
    else:
        env_fn = lambda : gym.make(args.env)


    #check if fcounts to load
    if args.baseline_file == '':
        fcount_baseline = None #business as usual
    else:
        #need to try and read in the file. I'll assume one line of comma separated feature counts
        freader = open(args.baseline_file, 'r')
        line = freader.read().strip()
        vals = []
        for val in line.split(','):
            vals.append(float(val))
        fcount_baseline = np.array(vals)

    ppo(env_fn, reward_dist=reward_dist, broil_risk_metric=args.risk_metric, broil_lambda=args.broil_lambda, broil_alpha=args.broil_alpha,
        actor_critic=core.BROILActorCritic, render=args.render,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        pi_lr=args.policy_lr, vf_lr = args.value_lr, logger_kwargs=logger_kwargs, clone=args.clone, num_demos=args.num_demos,
        expert_fcounts=fcount_baseline)
