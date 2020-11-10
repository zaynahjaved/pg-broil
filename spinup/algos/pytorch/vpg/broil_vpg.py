import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.vpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.examples.pytorch.broil_rtg_pg_v2.cartpole_reward_utils import CartPoleReward
from spinup.examples.pytorch.broil_rtg_pg_v2.cvar_utils import cvar_enumerate_pg


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        #self.adv_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = []
        #self.rew_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = []
        #self.ret_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = []
        #self.val_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = []
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
        #self.rew_buf[self.ptr] = rew
        self.rew_buf.append(rew)
        #self.val_buf[self.ptr] = val
        self.val_buf.append(val)
        self.logp_buf[self.ptr] = logp
        self.ptr += 1


    def finish_path(self, last_val=0):
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

        if isinstance(last_val, int):
            #path_slice = slice(self.path_start_idx, self.ptr)
            #rews = np.asarray([r+[last_val] for r in self.rew_buf]) #np.append(self.rew_buf[path_slice], last_val)
            #vals =  np.insert(np.asarray(self.rew_buf), len(self.rew_buf[0]), last_val*np.ones(len(self.rew_buf)), axis=1) #np.append(self.val_buf[path_slice], last_val)
            #vals = np.asarray([r+[last_val] for r in self.val_buf])
            rews = np.vstack((np.asarray(self.rew_buf), [last_val, last_val]))
            vals = np.vstack((np.asarray(self.val_buf), [last_val, last_val]))
        else:
            rews = np.vstack((np.asarray(self.rew_buf), last_val))
            vals = np.vstack((np.asarray(self.val_buf), last_val))
        
        advs = []
        # the next two lines implement GAE-Lambda advantage calculation
        for i in range(rews.shape[1]):
            deltas = rews[:-1,i] + self.gamma * vals[i, 1:] - vals[i, :-1]
            #print(deltas.shape)
            advs.append(core.discount_cumsum(deltas, self.gamma * self.lam))
            #print(np.asarray(advs).shape)
        #print(np.asarray(advs).shape)
        self.adv_buf = advs

        rets = []
        for i in range(len(rews)):
            rets.append(core.discount_cumsum(rews[i], self.gamma)[:-1])
        self.ret_buf = rets
        #self.ret_buf = np.sum(self.rew_buf, axis=0) #np.asarray(rets)
        # the next line computes rewards-to-go, to be targets for the value function
        #self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        #self.ret_buf.append(ep_ret_dist)
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
        #print(np.asarray(self.adv_buf).shape)
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def vpg(env_fn, reward_dist, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),  seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10, alpha=0.95):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

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
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

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

    # Create actor-critic module
    num_rew_fns = len(reward_dist.get_reward_distribution(np.zeros(obs_dim)))
    ac = actor_critic(env.observation_space, env.action_space, num_rew_fns, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing VPG policy loss
    def compute_loss_pi(data, wts):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        #print(wts)
        pi, logp = ac.pi(obs, act)
        wts = torch.from_numpy(wts)
        print(logp.size())
        print(adv.T.size())
        print(wts.size())

        loss_pi = 0
        for i in range(adv.size()[0]):
        # TODO: implement correct loss
            #loss += logp*adv[i]*wts[i]
            loss_pi += -1*(logp*wts[i]).mean()

        #loss_pi = -(loss).mean()
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def compute_broil_weights(batch_rets, batch_rtgs):
        #print(buf.ret_buf)
        #print(batch_rets)
        #batch_rets = buf.ret_buf #np.sum(buf.ret_buf, axis = 0)
        #batch_rtgs = buf.rtgs #buf.ret_buf
        #batch_rtgs =   
        #batch_rtgs = core.discount_cumsum(buf.rew_buf[slice(buf.path_start_idx, buf.ptr)], gamma)
        print("Rets {}".format(np.asarray(batch_rets).shape))

        batch_rtgs = buf.adv_buf
        print("Rtgs {}".format(np.asarray(batch_rtgs).shape))
        exp_batch_rets = np.mean(batch_rets, axis=0)
        posterior_reward_weights = reward_dist.posterior

        #print(posterior_reward_weights)

        sigma, cvar = cvar_enumerate_pg(exp_batch_rets, posterior_reward_weights, alpha)

        broil_wts = np.zeros(len(batch_rtgs))
        for i,prob_r in enumerate(posterior_reward_weights):
            if sigma > exp_batch_rets[i]:
                w_r_i = lam+(1-lam)/(1-alpha)
            else:
                w_r_i = lam

            broil_wts += prob_r*w_r_i*np.array(batch_rtgs)[:,i]

        return broil_wts

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        #TODO: implement correct loss
        loss = []
        v = np.asarray(ac.v(obs))
        #print(ret[:-1].size())
        for i in range(v.shape[0]):
            loss.append(((v[i]-ret[:-1])**2).mean())

        return loss

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizers = [Adam(ac.v.v_nets[i].parameters(), lr=vf_lr) for i in range(num_rew_fns)]

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # Get loss and info values before update
        wts_old = compute_broil_weights(batch_rets, batch_rtgs)
        pi_l_old, pi_info_old = compute_loss_pi(data, wts_old)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data)

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        wts = compute_broil_weights(batch_rets, batch_rtgs)
        loss_pi, pi_info = compute_loss_pi(data, wts)
        loss_pi.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            losses = compute_loss_v(data)
            for j in range(len(vf_optimizers)):
                vf_optimizers[j].zero_grad()
                loss_v = losses[j]
                loss_v.backward()
                mpi_avg_grads(ac.v)    # average grads across MPI processes
                vf_optimizers[j].step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old))


    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    #batch_rets = []
    #batch_rews = []
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        ep_rews = []
        batch_rets = []
        batch_rtgs = []

        buf.adv_buf = []
        buf.rew_buf = []
        buf.ret_buf = []
        buf.val_buf = []
        left_ac = 0
        right_ac = 0
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            if a == 0:
                left_ac += 1
            else:
                right_ac += 1
            next_o, r, d, _ = env.step(a)
            #ep_ret += r
            #ep_len += 1

            rew_dist = reward_dist.get_reward_distribution(o)
            ep_rews.append(rew_dist)

            # save and log
            buf.store(o, a, rew_dist, v, logp)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                ep_ret_dist, ep_len = np.sum(ep_rews, axis=0), len(ep_rews)
                batch_rets.append(ep_ret_dist)
                batch_rtgs.extend(reward_to_go(ep_rews))
                #print(ep_ret_dist)
                #if type(ep_ret_dist) != int:
                 #   buf.ret_buf.append(ep_ret_dist)
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    #buf.rew_buf = []
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=np.dot(ep_ret_dist,np.mean(batch_rets,axis=0)), EpLen=ep_len)
                    #print("Left {}".format(left_ac))
                    #print("Right {}".format(right_ac))
                o, ep_ret, ep_len = env.reset(), 0, 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        #logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--lam', type=float, default=0.9)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    reward_dist = CartPoleReward()

    vpg(lambda : gym.make(args.env), reward_dist=reward_dist, actor_critic=core.BROILActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, lam=args.lam,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, alpha=args.alpha)