import numpy as np

def relu(x):
    if x > 0:
        return x
    else:
        return 0.0


def cvar_fn_val(sigma, exp_ret_rs, prob_rs, alpha):
    fn_val_relu_part = 0.0
    for i,ret in enumerate(exp_ret_rs):
        fn_val_relu_part += prob_rs[i] * relu(sigma - ret)
    
    fn_val = sigma - 1.0 / (1.0 - alpha) * fn_val_relu_part
    return fn_val

def cvar_line_search_pg(exp_ret_rs, prob_rs, alpha, num_discretize=1000):
    '''use a line search to approximate sigma'''
    assert(len(exp_ret_rs) == len(prob_rs))
    assert(alpha >= 0 and alpha <= 1)
    assert(np.abs(np.sum(prob_rs) - 1.0) < 0.000001)
    #run simple discrete line search to approximate sigma for now

    max_val = -np.inf
    max_sigma = None    
    for x in np.linspace(min(exp_ret_rs), max(exp_ret_rs), num_discretize):
        cvar_val = cvar_fn_val(x, exp_ret_rs, prob_rs, alpha)
        #print(x, cvar_val)
        if cvar_val > max_val:
            max_val = cvar_val
            max_sigma = x
            #print("updating")
    
    return max_sigma, max_val





def cvar_enumerate_pg(exp_ret_rs, prob_rs, alpha):
    '''cvar is piecewise linear/concave so the max must be at one of the endpoints!
     we can just iterate over them until we find the smallest one'''

    sorted_exp_ret_rs, sorted_prob_rs = zip(*sorted(zip(exp_ret_rs, prob_rs)))
    #print("sorted rets", sorted_exp_ret_rs)
    #print("sorted probs", sorted_prob_rs)
    cum_prob = 0.0
    
        
    max_val = -np.inf
    max_sigma = None    
    for ret in sorted_exp_ret_rs:
        cvar_val = cvar_fn_val(ret, exp_ret_rs, prob_rs, alpha)
        #print(x, cvar_val)
        if cvar_val >= max_val:
            max_val = cvar_val
            max_sigma = ret
            #print("updating")
        elif cvar_val < max_val:
            #this function is concave so once it starts decreasing we can stop since we are only interested in maximum
            break
    
    return max_sigma, max_val



# if __name__ == "__main__":
#     #run test to make sure both give same answers.
#     #Note cvar_enumerate_pg is orders of magnitude faster and gives same answer as far as I can tell
#     for i in range(100):
#         seed = np.random.randint(1000)
#         print(seed)
#         np.random.seed(seed)
#         num_rewards = 50
#         exp_rets = 200*np.random.rand(num_rewards) - 100 #[10,40, 80]
#         probs = np.random.rand(num_rewards)#[0.3, 0.3, 0.4]
#         probs /= np.sum(probs)
#         #print(np.sum(probs))
#         alpha = 0.6
#         num_discretize = 10000
#         #print("exp rets", exp_rets)
#         #print("probs", probs)
#         sigma, cvar = cvar_line_search_pg(exp_rets, probs, alpha, num_discretize)
#         print("sigma = ", sigma)
#         print("cvar = ", cvar)

#         sigma_enumerate, cvar_enumerate = cvar_enumerate_pg(exp_rets, probs, alpha)
#         print("enum sigma", sigma_enumerate)
#         print("sort cvar", cvar_enumerate)

#         if abs(sigma_enumerate -  sigma) > 0.1 or abs(cvar - cvar_enumerate) > 0.001:
#             print("wrong")
#             print(abs(sigma_enumerate -  sigma))
#             input()


if __name__ == "__main__":
    #run test to make sure both give same answers.
    #Note cvar_enumerate_pg is orders of magnitude faster and gives same answer as far as I can tell
    num_rewards = 2
    exp_rets = [10, 90]
    probs = [0.05, 0.95]
    probs /= np.sum(probs)
    #print(np.sum(probs))
    alpha = 0.95
    num_discretize = 10000
    #print("exp rets", exp_rets)
    #print("probs", probs)
    sigma, cvar = cvar_line_search_pg(exp_rets, probs, alpha, num_discretize)
    print("sigma = ", sigma)
    print("cvar = ", cvar)

    sigma_enumerate, cvar_enumerate = cvar_enumerate_pg(exp_rets, probs, alpha)
    print("enum sigma", sigma_enumerate)
    print("sort cvar", cvar_enumerate)

    if abs(sigma_enumerate -  sigma) > 0.1 or abs(cvar - cvar_enumerate) > 0.001:
        print("wrong")
        print(abs(sigma_enumerate -  sigma))
        input()
