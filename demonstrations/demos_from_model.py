import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import dmc2gym
import pickle

def load_policy_and_env(fpath, itr='last', deterministic=False, env_name=None):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        if env_name == 'reacher':
            env = dmc2gym.make(domain_name='reacher', task_name='easy', episode_length=200)
        else:
            env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, env_name=None):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    num_violations = 0
    num_target = 0
    violations = []
    target = []
    hit_by_opponent = 0
    score_hit = 0
    avoid_opponent = 0
    hit_feat_counts = []
    score_feat_counts = []
    avoid_feat_counts = []
    episode_feat_counts = []

    pellet_counts = 0
    power_pellet_counts = 0
    eat_ghost_counts = 0
    eat_cherry_counts = 0
    hit_ghost_counts = 0
    pellet_feat_counts = []
    power_feat_counts = []
    eat_ghost_feat_counts = []
    eat_cherry_feat_counts = []
    hit_ghost_feat_counts = []
    ep_scores = []

    demo_obs = []
    demo_acs = []

    prev_ale = 3
    curr_ale = 3
    while n < num_episodes:
        #hit_by_opponent = 0
        #score_hit = 0
        #avoid_opponent = 0
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, info = env.step(a)
        #print(r)
        if env_name == 'Boxing-ram-v0':
            if r == 0:
                avoid_opponent += 1
            elif r < 0:
                hit_by_opponent -= int(r)
            else:
                score_hit += int(r)
        if env_name == 'MsPacman-ram-v0':
            curr_ale = env.ale.lives()
            if r == 10:
                pellet_counts += 1
            if r == 50:
                power_pellet_counts += 1
            if r == 200 or r  == 400 or r == 800 or r == 1600:
                eat_ghost_counts += 1
            if r == 100:
                eat_cherry_counts += 1
            else:
                if curr_ale == prev_ale-1:
                    hit_ghost_counts += 1
                    prev_ale = curr_ale
        if env_name == 'reacher':
            if info['constraint']:
                num_violations += 1
            if env.get_features()[0]:
                num_target += 1
        ep_ret += r
        ep_len += 1
        demo_obs.append(o)
        demo_acs.append(a)
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            if env_name == 'reacher':
                print('Violations %d, Target %d'%(num_violations,num_target))
            if env_name == 'Boxing-ram-v0':
                print("damage %d"%hit_by_opponent)
                print("scores %d"%score_hit)
                print("avoid %d"%avoid_opponent)
            if env_name == 'MsPacman-ram-v0':
                print("pellet %d"%pellet_counts)
                print("power pellet %d"%power_pellet_counts)
                print("ghosts eaten %d"%eat_ghost_counts)
                print("cherry %d"%eat_cherry_counts)
                print("hit ghost %d"%hit_ghost_counts)
            if env_name == 'Boxing-ram-v0':
                episode_feat_counts.append([hit_by_opponent, score_hit, avoid_opponent, ep_ret])
            if env_name == 'MsPacman-ram-v0':
                episode_feat_counts.append([pellet_counts, power_pellet_counts, eat_ghost_counts, eat_cherry_counts, hit_ghost_counts])
                ep_scores.append(ep_ret)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            violations.append(num_violations)
            target.append(num_target)
            hit_feat_counts.append(hit_by_opponent)
            score_feat_counts.append(score_hit)
            avoid_feat_counts.append(avoid_opponent)
            num_violations = 0
            num_target = 0
            hit_by_opponent = 0
            avoid_opponent = 0
            score_hit = 0

            pellet_counts = 0
            power_pellet_counts = 0 
            eat_ghost_counts = 0
            eat_cherry_counts = 0
            hit_ghost_counts = 0
            prev_ale = 3
            n += 1
    if args.env_name == 'reacher':
        print(violations)
        print(target)
    if args.env_name == 'Boxing-ram-v0':
        features = {'Features': episode_feat_counts, "Obs": demo_obs, "Scores":ep_scores, "Acs": demo_acs}
        pickle.dump(features, open('boxing_demos.pkl', 'wb'))
    if args.env_name == 'MsPacman-ram-v0':
        features = {'Features': episode_feat_counts,"Scores": ep_scores, "Obs": demo_obs, "Acs":demo_acs}
        pickle.dump(features, open('pacman_demos.pkl', 'wb'))
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--env_name', type=str)
    #parser.add_argument('--save_feat_file', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic, args.env_name)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender), args.env_name)

