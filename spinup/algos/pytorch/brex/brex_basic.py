import argparse

# Use hand tuned feature count vectors for testing the MCMC convergence and distributions.
# Doesn't actually use any kind of real domain.


import numpy as np
import random
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import pickle

def calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts, confidence):
    '''use (i,j) indices and precomputed feature counts to do faster pairwise ranking loss'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():

        #do matrix multiply with last layer of network and the demo_cnts
        weights = torch.from_numpy(last_layer) #append bias and weights from last fc layer together
        demo_cnts_tensor = torch.from_numpy(demo_cnts)
        demo_returns = confidence * torch.mv(demo_cnts_tensor, weights)


        loss_criterion = nn.CrossEntropyLoss(reduction='sum') #sum up losses
        cum_log_likelihood = 0.0
        outputs = torch.zeros(len(pairwise_prefs),2) #each row is a new pair of returns
        for p, ppref in enumerate(pairwise_prefs):
            i,j = ppref
            outputs[p,:] = torch.tensor([demo_returns[i], demo_returns[j]])
        labels = torch.ones(len(pairwise_prefs)).long()
        cum_log_likelihood = = -loss_criterion(outputs, labels)
    return cum_log_likelihood.item()



def mcmc_map_search(pairwise_prefs, demo_cnts, num_steps, step_stdev, confidence, normalize):
    '''run metropolis hastings MCMC and record weights in chain'''

    last_layer = np.random.randn(len(demo_cnts[0]))

    #normalize the weight vector to have unit 3-norm
    if normalize:
        last_layer = last_layer / np.linalg.norm(last_layer)

    starting_loglik = calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts, confidence)
    map_loglik = starting_loglik
    map_reward = copy.deepcopy(last_layer)

    cur_reward = copy.deepcopy(last_layer)
    cur_loglik = starting_loglik


    reject_cnt = 0
    accept_cnt = 0
    chain = []
    log_liks = []
    for i in range(num_steps):
        #take a proposal step
        proposal_reward = cur_reward + np.random.normal(size=cur_reward.shape) * step_stdev #add random noise to weights of last layer

        #project
        if normalize:
            proposal_reward = proposal_reward / np.linalg.norm(proposal_reward)

        #calculate prob of proposal
        prop_loglik = calc_linearized_pairwise_ranking_loss(proposal_reward, pairwise_prefs, demo_cnts, confidence)
        if prop_loglik > cur_loglik:
            #accept always
            accept_cnt += 1
            cur_reward = copy.deepcopy(proposal_reward)
            cur_loglik = prop_loglik

            #check if this is best so far
            if prop_loglik > map_loglik:
                map_loglik = prop_loglik
                map_reward = copy.deepcopy(proposal_reward)
                print()
                print("step", i)

                print("updating map loglikelihood to ", prop_loglik)
                print("MAP reward so far", map_reward)
        else:
            #accept with prob exp(prop_loglik - cur_loglik)
            if np.random.rand() < np.exp(prop_loglik - cur_loglik):
                accept_cnt += 1
                cur_reward = copy.deepcopy(proposal_reward)
                cur_loglik = prop_loglik
            else:
                #reject and stick with cur_reward
                reject_cnt += 1
        chain.append(cur_reward)
        log_liks.append(cur_loglik)

    print("num rejects", reject_cnt)
    print("num accepts", accept_cnt)
    return map_reward, np.array(chain), np.array(log_liks)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_mcmc_steps', default=20000, type = int, help="number of proposals to generate for MCMC")
    parser.add_argument('--mcmc_step_size', default = 0.5, type=float, help="proposal step is gaussian with zero mean and mcmc_step_size stdev")
    parser.add_argument('--confidence', default=1, type=int, help='confidence in rankings, the beta parameter in the softmax')
    parser.add_argument('--normalize', default=False, action='store_true', help='turns on normalization so the reward function weight vectors have norm of 1')
    parser.add_argument('--features_dir', default='feature_files', help='directory containing pickle files of demonstration feature counts')
    parser.add_argument('--save_file', default='brex_reward_dist.pickle', help='where to save the outputted BREX reward distribution')
    parser.add_argument('--env', default=None, help='which environment this BREX reward is being generated for')

    args = parser.parse_args()
    
    pickle_files = []
    import os
    for filename in os.listdir(args.features_dir):
        if filename.endswith(".pkl" or ".p"):
            pickle_files.append(args.features_dir + "/" + filename)
    print(pickle_files)

    demo_fcnts = []
    pairwise_prefs = []
    counter = 0
    if args.env == 'reacher':
        for file in pickle_files:
            sum_feat = np.array([0.0,0.0,0.0])
            with open(file, 'rb') as handle:
                b = pickle.load(handle)
                for arr in b['features']:
                    sum_feat += arr

                sum_feat = [sum_feat[0], sum_feat[1]]
                demo_fcnts.append(sum_feat)
        pairwise_prefs =  [(5,4),(4,3),(3,2),(2,1),(1,0)]


    elif args.env == 'PointBot-v0':
        for file in pickle_files:
            with open(file, 'rb') as handle:
                b = pickle.load(handle)
                demo_fcnts += [b["Bad_feature"]]
                demo_fcnts += [b["Good_feature"]]
                pairwise_prefs += [(counter, counter+1)]
                counter += 2

    print("Hand crafted feature expectations")
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_features = 3 #inside obstacle, outside obstacle, number of trash
    if args.env == 'reacher':
        num_features = 2

    demo_fcnts = np.array(demo_fcnts).astype(float)

    #just need index tuples (i,j) denoting j is preferred to i. Assuming all pairwise prefs for now
    #check if really better, there might be ties!

    #Run mcmc
    best_reward_lastlayer,chain,log_liks = mcmc_map_search(pairwise_prefs, demo_fcnts, args.num_mcmc_steps, args.mcmc_step_size, args.confidence, args.normalize)

    print("best reward weights", best_reward_lastlayer)
    print("predictions over ranked demos vs. ground truth returns")
    pred_returns = np.dot(demo_fcnts, best_reward_lastlayer)

    print("average reward weights")
    print(np.mean(chain, axis=0))


    print("10 random chain samples")
    for _ in range(10):
        w = random.choice(chain)
        print(w)


    #Start saving reward distribution
    sampling_rate = 1000
    sampling_chain = chain[500:]
    
    weight_to_prob = {}
    brex_subsample_distr = [0, 0, 0]
    for i in range(sampling_chain.shape[0]):
        if (i%sampling_rate == 0):
            if args.env == 'reacher':
                weight_to_prob[(sampling_chain[i][0], sampling_chain[i][1])] = 1
                brex_subsample_distr[0] = brex_subsample_distr[0] + sampling_chain[i][0]
                brex_subsample_distr[1] = brex_subsample_distr[1] + sampling_chain[i][1]
            else:
                weight_to_prob[(sampling_chain[i][0], sampling_chain[i][1], sampling_chain[i][2])] = 1
                brex_subsample_distr[0] = brex_subsample_distr[0] + sampling_chain[i][0]
                brex_subsample_distr[1] = brex_subsample_distr[1] + sampling_chain[i][1]
                brex_subsample_distr[2] = brex_subsample_distr[2] + sampling_chain[i][2]

    for key in weight_to_prob:
        weight_to_prob[key] = 1/(len(weight_to_prob))

    brex_subsample_distr[0] = brex_subsample_distr[0]/(len(weight_to_prob))
    brex_subsample_distr[1] = brex_subsample_distr[1]/(len(weight_to_prob))
    if args.env != 'reacher':
        brex_subsample_distr[2] = brex_subsample_distr[2]/(len(weight_to_prob))

    print('WEIGHTS TO PROBABILITY')
    print(weight_to_prob)

    with open(args.save_file, 'wb') as handle:
        pickle.dump(weight_to_prob , handle, protocol=pickle.HIGHEST_PROTOCOL)
    #End saving reward distribution


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(log_liks)
    plt.title("log likelihoods")
    plt.figure()
    for i in range(num_features):
            plt.plot(chain[:,i],label='feature ' + str(i))
    plt.title("features")
    plt.legend()
    plt.show()
