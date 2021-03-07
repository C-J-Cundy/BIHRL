import numpy as np
import itertools
import matplotlib.pyplot as plt
import random
#import seaborn as sns
import sys
np.set_printoptions(precision=3)
from itertools import chain, combinations
import multiprocessing
import os
import time
import scipy.stats
import matplotlib


tol = 1e-3
np.random.seed(2017)

# We will give reward 10 for getting to the correct destination, otherwise reward -1
# on every transition.
#---------------------------------------------------------------------------
def allin(list1, list2):
    """Naive quadratic-time"""
    for a in list1:
        if a not in list2:
            return False
    return True

def build_article_map(f='./wikispeedia_paths-and-graph/articles.tsv'):
    """Takes the file with all the articles and returns a dictionary mapping
    an article name to its alphabetical place in the articles"""
    out_dict = {}
    count = 0
    with open(f, 'r') as r:
        for _ in xrange(12):
            next(r)
        for line in r:
            out_dict[line.strip('\n')] = count
            count += 1
    return out_dict

def build_reverse_article_map(f='./wikispeedia_paths-and-graph/articles.tsv'):
    """Takes the file with all the articles and returns a dictionary mapping
    an article name to its alphabetical place in the articles"""
    out_dict = {}
    count = 0
    with open(f, 'r') as r:
        for _ in xrange(12):
            next(r)
        for line in r:
            out_dict[count] = line.strip('\n')
            count += 1
    return out_dict
def build_adjacency_matrix(art_dict, f='./wikispeedia_paths-and-graph/links.tsv'):
    """Taking the file where the graph of connections is defined, build a matrix with 
    adjacencies"""
    adj = np.zeros((4604, 4604), dtype='b')
    with open(f, 'r') as r:
        for _ in xrange(12):
            next(r)
        for line in r:
            adj[art_dict[line.split('\t')[0]], art_dict[line.split('\t')[0]]] = 1
    return adj

def build_adjacency_list(art_dict, f='./wikispeedia_paths-and-graph/links.tsv'):
    """Build an adjacency dictionary where di[rep] == [list of connected nodes].
    We trim the list of all nodes which are dead ends, and which can't be reached."""
    out_dict = {}
    with open(f, 'r') as r:
        for _ in xrange(12):
            next(r)
        for line in r:
            fr = art_dict[line.split('\t')[0].strip('\n')]
            to = art_dict[line.split('\t')[1].strip('\n')]
            if fr in out_dict.keys():
                out_dict[fr].append(to)
            else:
                out_dict[fr] = [to]
    for key in out_dict.keys(): #Remove all dead ends. Very messy
        to_remove = []
        for dest in out_dict[key]:
            if dest not in out_dict.keys():
                to_remove.append(dest)
        tmp = []
        tmp = [x for x in out_dict[key] if x not in to_remove]
        if tmp == []:
            del(out_dict[key])
        else:
            out_dict[key] = tmp
    return out_dict

def links_to(adj, number):
    """Returns the pages that link to the page indexed by number"""
    connected = []
    for i, v in adj.iteritems():
        if number in v:
            connected.append(i)
    return connected            
def trim_adjacency_list(adj):
    """Trim an adjacency list of all nodes which can't be reached"""
    old_list = adj.copy()
    for key in adj.keys():
        if links_to(old_list, key) == []:
            del(adj[key])
    return adj

def build_trajectories(art_dict, adj_list, f='./wikispeedia_paths-and-graph/paths_finished.tsv'):
    trajectories = [[]]
    with open(f, 'r') as r:
        for _ in xrange(16):
            next(r)
        for line in r:
            packed_nodes = line.split('\t')[3]
            if '<' in packed_nodes:
                pass
            else:                
                nodes = [art_dict[node] for node in packed_nodes.split(';')]
                trajectories.append(nodes)
    return [t for t in trajectories if allin(t, adj_list.keys())]
def test_train_split(trajectories):
    n = len(trajectories)
    test = list(np.random.choice(trajectories, size=n/2))[1:]
    train = [t for t in trajectories if t not in test][1:]    
    return test, train

def build_target_list(trajectories):
    target_set = set()
    for a in trajectories:
        target_set = target_set.union(set([a[-1]]))
    print len(target_set)
    return target_set
#---------------------------------------------------------------------------
#Set up some global embeddings of the graph structure
art_map = build_article_map()
art_map_r = build_reverse_article_map()
#adj_list = build_adjacency_list(art_map)
#adjusted_adj_list = trim_adjacency_list(build_adjacency_list(art_map))
#adjusted_adj_list_2 = trim_adjacency_list(trim_adjacency_list(build_adjacency_list(art_map)))
adjusted_adj_list_4 = trim_adjacency_list(trim_adjacency_list(
    trim_adjacency_list(trim_adjacency_list(build_adjacency_list(art_map)))))
paths = build_trajectories(art_map, adjusted_adj_list_4)
test, train = test_train_split(paths)
poss_targets = build_target_list(paths[1:])
#---------------------------------------------------------------------------

def precompute_options_values(adj, skills, g):
    """Precalculate the expected destinations and rewards for the skills"""
    rewards = np.zeros((4604, len(skills)))
    dests = np.zeros((4604, len(skills)), dtype='int')
    discounts = np.zeros((4604, len(skills))) #Containing prefactors g **(avg_len)
    skill_vals = []
    skill_Zs = []
    adj = adjusted_adj_list_4
    for skill in skills:
            out = value_iteration(adj, [], skill[1], g=g, goal=skill[0])
            skill_vals.append(out[0])
            skill_Zs.append(out[1])
    print skill_vals, skill_Zs
    for s_index, skill in enumerate(skills):
        for index in adj.keys():
            dests[index, s_index] = skill[0]
            out = option(index, skill[0], skill[1], g,
                         vals=skill_vals[s_index], Z=skill_Zs[s_index], n_repeat=30)
            if out[0] == None: #Skill is disconnected from destination
                dests[index, s_index] = index
                rewards[index, s_index] = -1
                discounts[index, s_index] = g
            else:
                rewards[index, s_index] = out[1]
                discounts[index, s_index] = np.power(g, out[2])
        print 'Done skill {}'.format(dests, rewards, discounts)
    skill_vals_2 = discounts * 0
    skill_Zs_2 = discounts * 0
    for index in adj.keys():
        for s_index in range(len(skills)):
            skill_vals_2[index, s_index] = skill_vals[s_index][index]
            skill_Zs_2[index, s_index] = skill_Zs[s_index][index]
    dests = np.array(dests, dtype='int')
    return rewards, dests, discounts, skill_vals_2, skill_Zs_2

def multi_pov(a):
    print a
    adj = a[0]
    skills = a[1]
    g = a[2]
    rewards = np.zeros((4604, len(skills)))
    dests = np.zeros((4604, len(skills)), dtype='int')
    discounts = np.zeros((4604, len(skills))) #Containing prefactors g **(avg_len)
    skill_vals = []
    skill_Zs = []
    adj = adjusted_adj_list_4
    for skill in skills:
            out = value_iteration(adj, [], skill[1], g=g, goal=skill[0])
            skill_vals.append(out[0])
            skill_Zs.append(out[1])
    for s_index, skill in enumerate(skills):
        for index in adj.keys():
            dests[index, s_index] = skill[0]
            out = option(index, skill[0], skill[1], g,
                         vals=skill_vals[s_index], Z=skill_Zs[s_index], n_repeat=15)
            if out[0] == None: #Skill is disconnected from destination
                dests[index, s_index] = index
                rewards[index, s_index] = -1
                discounts[index, s_index] = g
            else:
                rewards[index, s_index] = out[1]
                discounts[index, s_index] = np.power(g, out[2])
        print 'Done skill {}'.format(dests[0])
    skill_vals_2 = discounts * 0
    skill_Zs_2 = discounts * 0
    for index in adj.keys():
        for s_index in range(len(skills)):
            skill_vals_2[index, s_index] = skill_vals[s_index][index]
            skill_Zs_2[index, s_index] = skill_Zs[s_index][index]
    dests = np.array(dests, dtype='int')
    return rewards, dests, discounts, skill_vals_2, skill_Zs_2
    
def precompute_options_values_multipool(adj, skills, g):
    """Precalculate the expected destinations and rewards for the skills"""
    import os
    import multiprocessing
    processes=[]
    pool = multiprocessing.Pool(processes=len(skills))
    out = pool.map(multi_pov, [['fsdfa', [skill], g] for skill in skills])
    print out
    rewards = np.zeros((4604, len(skills)))
    dests = np.zeros((4604, len(skills)))
    discounts = np.zeros((4604, len(skills)))
    skill_vals_2 = np.zeros((4604, len(skills)))
    skill_Zs_2 = np.zeros((4604, len(skills)))
    if not os.path.exists('./povs/' + str(len(skills)) + '/' + str(skills[0][1])):
        os.makedirs('./povs/' + str(len(skills)) + '/' + str(skills[0][1]))
    names = ['rewards', 'dests', 'discounts', 'skill_vals_2', 'skill_Zs_2']
    for n in range(len(skills)):
        for index, array in enumerate([rewards, dests, discounts, skill_vals_2, skill_Zs_2]):
            array[:, n] = out[n][index][:, 0]
            if index == 1:
                dests = np.array(dests, dtype='int')
            np.save('./povs/' + str(len(skills)) + '/' + str(skills[0][1]) + '/' + str(names[index]), array)
    return [rewards, dests, discounts, skill_vals_2, skill_Zs_2]

def value_iteration(adj, skills, beta, g=0.99, goal=50,
                    v_init=-10*np.ones((4604)), skill_vals_precomputed=[],
                    early_quit=False):
    vs_0 = np.zeros((4604))
    vs_1 = v_init[:]
    vs_1[goal] = 40
    vs_1[links_to(adjusted_adj_list_4, goal)[0]] = 40
    vs_1[links_to(adjusted_adj_list_4, links_to(adjusted_adj_list_4, goal)[0])[0]] = 40
    count = 0  #Build list of singleton pages
    none_list = [x for x in range(4604) if x not in adj.keys()]
    skill_vals = []#precompute value iteration for the sub-MDPs that are skills
    skill_Zs = []#precompute value iteration for the sub-MDPs that are skills    
    if skills != []:
        rewards, dests, discounts, _, _ = skill_vals_precomputed
    indices = adj.keys()
    tmp = np.zeros(1000)
    skill_enum = enumerate(skills)
    while abs(np.amax(np.abs(vs_0 - vs_1))) > tol:
        np.copyto(vs_0, vs_1)
        vs_1 = vs_1 * 0
        for index in indices:
            t_count = 0
            for index1 in adj[index]:
                if index1 == goal:
                    tmp[t_count] = 20 #Transitioning to winning state
                else:
                    tmp[t_count] = -1 + vs_0[index1]*g
                t_count += 1
            for skill_index, skill  in skill_enum: #Skills are lists [destination, beta_2]
                tmp[t_count] = (rewards[tuple([index, skill_index])] +
                                vs_0[dests[tuple([index, skill_index])]]*discounts[tuple([index, skill_index])])
                t_count += 1
            z = np.exp(beta*tmp[:t_count])                           
            vs_1[index] = np.sum(tmp[:t_count] * z) / np.sum(z)
            vs_1[goal] = 0
        count += 1
        tmp = tmp * 0
   #     if count % 30 == 0:
#            print 'Doing value iteration, got to count {}'.format(count)
        if count > 1000 and early_quit:
            vs_1[0] = None
            return vs_1, None #To handle freezing
    Z = vs_0 * 0
    for index in adj.keys():
        if index in none_list:
            pass
        else:
            for index1 in adj[index]:
                if index1 == goal:
                    Z[index] += np.exp(20*beta)
                else:
                    Z[index] += np.exp(beta*(-1 + vs_0[index1]*g))
    return vs_0, Z
def gen_option_trajectories(O_a, skills, beta, g, theta=[], vals=-1*np.ones(4604), Zs=None, povs=None):
    """Given an action-trajectory O_a specified as a list [[a_0, s_0], [a_1, s_1],
    ... [a_2, s_2]], generate all sets of option-trajectories that are
    consistent with the action-trajectory. Do this following algorithm 1 in the
    literature provided. Also return a probability that the trajectory will be 
    generated given the chosen option."""
    adj = adjusted_adj_list_4
    T_o = [[[]] for i in range(len(O_a))] #Generate empty list-of-lists-of-lists
    Ps = [[] for i in range(len(O_a))]
    Ps[0] = [1]
    for i in range(len(O_a)):
        if T_o[i] != [] or i == 0:
            for k in range(1, len(O_a) - i):
                Os = adj[O_a[i]] + skills if skills != [] else adj[O_a[i]]
                for o in Os:
                    # print o, k
                    # print prob_consistent(O_a[i:i+k+1], o) != 0
                    p = prob_consistent(O_a[i:i+k+1], o, beta, g, theta, vals, Zs)
                    if p != 0:
                        to_add = [[]]
                        for index, t in enumerate(T_o[i]):
                            if to_add == [[]]:
                                to_add = [t + [o]]
                                p_add = [Ps[i][index] * p]
                            else:
                                to_add.append(t + [o])
                                p_add.append(p * Ps[i][index])
                        for index, t1 in enumerate(to_add):
                            if T_o[i+k] == [[]]:
                                T_o[i+k] = [t1]
                                Ps[i+k] = [p_add[0]]
                            else:
                                T_o[i+k].append(t1)
                                Ps[i+k].append(p_add[index])
                for index, o in enumerate(skills):
                    p = prob_consistent(O_a[i:i+k+1], o, beta, g, theta,
                                        povs[3][:, index], povs[4][:, index], option_flag=True)
                    if p != 0:
                        to_add = [[]]
                        for index, t in enumerate(T_o[i]):
                            if to_add == [[]]:
                                to_add = [t + [o]]
                                p_add = [Ps[i][index] * p]
                            else:
                                to_add.append(t + [o])
                                p_add.append(p * Ps[i][index])
                        for index, t1 in enumerate(to_add):
                            if T_o[i+k] == [[]]:
                                T_o[i+k] = [t1]
                                Ps[i+k] = [p_add[0]]
                            else:
                                T_o[i+k].append(t1)
                                Ps[i+k].append(p_add[index])                                
                                #                        T_o[i+k].extend(to_add) #Add to trajectories getting to i+k
    return T_o[-1], Ps[-1] #Return all trajectories that get to the final state.
def prob_consistent(O_a, o, beta, g, theta, vals, Zs=None, option_flag=False):
    if option_flag == False and o == O_a[-1]:
        return 1
    elif option_flag == False:
        return 0
    else:
        s = O_a[0]
        beta = o[1]
        if O_a[-1] == o[0]: #If actual destination is the destination of the skill
            for index, states in enumerate(O_a[:-1]):
                if index == 0:
                    if O_a[index + 1] == o[0]:
                        Q = 20
                        prob = np.exp(beta*Q) / Zs[states]
                    else:
                        Q = -1 + g*vals[int(O_a[index + 1])]                        
                        prob = np.exp(beta*Q) / Zs[states]
                else:
                    if O_a[index + 1] == o[0]:
                        Q = 20
                        prob *= np.exp(beta*Q) / Zs[states]
                    else: 
                        Q = -1 + g*vals[int(O_a[index + 1])] 
                        prob *= np.exp(beta*Q) / Zs[states]
            return prob
        else:
            return 0                
def compute_likelihood(O_a, skills, beta, g=0.99, theta=[],
                       init_val=np.random.random((4604)), povs=None):
    adj_list = adjusted_adj_list_4
    """Given an observed action-trajectory O_a, and a set of available actions
    Os, compute the likelihood of taking that trajectory given the reward function."""
    vals, Zs = value_iteration(adj_list, skills, beta, g, theta, init_val, povs) #Do value iteration
    O_os, P_os = gen_option_trajectories(O_a, skills, beta, g, theta, vals, Zs, povs) #Generate all option-trajectories    
    probs = []
    for index, O_o in enumerate(O_os): #For option-trajectory in option-trajectories
        state = O_a[0] #Initial state
        prob = 0
        for index, o in enumerate(O_o): #For option in option-trajectory (either state to go to or [goal, beta_k] pair
            #Now compute the prob of choosing the option o that was actually chosen
            if o == theta:
                to_add = np.exp(beta*20) / Zs[state]
            elif type(o) != list:
                to_add = np.exp(beta*(-1 + g*vals[o])) / Zs[state]
            else: #Compound option
                out = option(state, o[0], o[1], beta, g, vals, Zs, 10) #Gives samples, discounted rewards, lengths                
                to_add = P_os[index]*np.exp(beta*(out[0] + (np.pow(g,out[2]))*vals[o[0]])) / Zs[state]
            if prob == 0:
                prob = to_add
            else:
                prob *= to_add
                state = o
        probs.append(prob)
    return np.sum(probs), vals, Zs
def compute_likelihood_given(O_a, skills, beta, g, theta, vals, Zs, povs):
    """Given a precomputed value function (vals, Zs), along with a goal 
    theta, compute the likelihood of observing the observed trajectory."""
    O_os, P_os = gen_option_trajectories(O_a, skills, beta, g, theta, vals, Zs, povs) #Generate all option-trajectories
    if povs != []:
        rewards, dests, discounts, _, _ = povs
    probs = []
    for index, O_o in enumerate(O_os): #For option-trajectory in option-trajectories
        state = O_a[0] #Initial state
        prob = 0
        for o_index, o in enumerate(O_o): #For option in option-trajectory (either state to go to or [goal, beta_k] pair
            #Now compute the prob of choosing the option o that was actually chosen
            if type(o) != list and o == theta:
                to_add = np.exp(beta*20) / Zs[state]
                state = o
            elif type(o) != list:
                to_add = np.exp(beta*(-1 + g*vals[o])) / Zs[state]
                state = o
            else: #Compound option
                Q = (rewards[tuple([state, o_dict[o[0]]])] +
                     vals[dests[tuple([state, o_dict[o[0]]])]]*discounts[tuple([state, o_dict[o[0]]])])
                #                out = option(state, o[0], o[1], g, povs[3][:, o_dict[o[0]]], povs[4][:, o_dict[o[0]]], 10) #Gives samples, discounted rewards, lengths
                #                to_add = np.exp(beta*(out[1] + (pow(g,out[2]))*vals[o[0]])) / Zs[state]
                to_add = np.exp(beta*Q) / Zs[state]
                state = o[0]
            if prob == 0:
                prob = to_add
            else:
                prob *= to_add
        if prob == 0:
            print O_a
            print '??'
        probs.append(prob)
    return np.sum(probs)
def birl_agent(start, goal, skills, beta, g=0.99, theta=0, pov= None, vals=-1*np.ones((4604)), Z=None):
    """Given a value function defined on states vals, 
    return a sample from the trajectory of a Boltzmann-rational agent
    in the space"""
    adj_list = adjusted_adj_list_4
    if vals[0] == -1: #Can provide initial values or recompute them, but must provide pov or else is v slow
        vals, Z = value_iteration(adjusted_adj_list_4, skills, beta, g, theta, goal, skill_vals_precomputed=pov)
    else:
        vals = vals[:]
        Z = Z[:]
        s = start
        s_list = [s]
        count = 0
    while s != goal:
        probs = []
        for o in adj_list[s]:
            prob = 0
            if o == goal:
                prob = np.exp(beta*(20)) / Z[s]
                probs.append(prob)                
            else:
                prob = np.exp(beta*(-1 + g*vals[o])) / Z[s]
                probs.append(prob)
        for s_index, skill in enumerate(skills):
            prob = np.exp(beta*(pov[0][tuple([s, s_index])] + (g**pov[2][tuple([s, s_index])])*vals[skill[1]])) / Z[s]
            probs.append(prob)
        delta = 1 - np.cumsum(probs)[-1]
        probs[np.where(np.array(probs) == np.amax(np.array(probs)))[0][0]] += delta
        ff = np.random.choice(range(len(probs)), p=probs)
        o = adj_list[s][ff]
        s = o
        s_list.append(o)
        count += 1
        if count > 2000000: #Start and goal are disconnected
            return None                       
    return s_list

def sample_lengths(eta, num=30, seed=0, trajectories=test, pov=None):
    """Take a random sample of some of the actual paths and see how they compare
    to the predicted BIRL agent"""
    np.random.seed(seed)
    outs = [[]]
    path_indices = np.random.choice(range(len(trajectories)), size=num)
    count = 0
    for index in path_indices:
        vals, Zs = load_theta(paths[index][-1], eta)        
        start = paths[index][0]
        end   = paths[index][-1]
        sampled_path = birl_agent(start, end, eta[1], eta[0], eta[2], pov=pov, vals=vals, Z=Zs)
        count += 1
        if outs == [[]]:
            outs = [[paths[index], sampled_path]]
        else:
            outs.append([paths[index], sampled_path])
    return outs

def analyse_lengths(num, eta, seed=0, pov=None):
    samples = sample_lengths(eta, num, seed, paths, pov)
    #Get list of those paths that are
    human_lens = {}
    computer_lens = {}
    len_correct = 0
    for sample in samples:
        human_len = len(sample[0]) + 1
        computer_len = len(sample[1]) + 1
        if human_len in human_lens.keys():
            human_lens[human_len] += 1
        else:
            human_lens[human_len] = 1
        if computer_len in computer_lens.keys():
            computer_lens[computer_len] += 1
        else:
            computer_lens[computer_len] = 1
            print 'Lengths are: {} for human and {} for computer'.format(human_lens,
                                                                         computer_lens)
            human_keys = sorted(human_lens)
            plt.plot(human_keys, [human_lens[key] for key in human_keys],
                     markersize=8, linewidth=1, marker='o', linestyle='-', color='blue',
                     mfc='none', label='human')
            computer_keys = sorted(computer_lens)
            plt.plot(computer_keys, [computer_lens[key] for key in computer_keys],
                     markersize=8, linewidth=1, marker='o', linestyle='--', color='green',
                     mfc='none', label='computer')
            plt.legend()
            plt.show()
            #    return samples        
def count_popular(paths):
    totals  = []
    for path in paths:
        totals.extend(path)
    uniques, unique_indices, unique_counts = np.unique(totals, return_index=True,
                                                       return_counts=True)
    return uniques[np.argsort(unique_counts)[::-1]], np.sort(unique_counts)[::-1]

def option(start, destination, beta, g, vals=-1*np.ones((4604)), Z=-1*np.ones((4604)),
           n_repeat=10):
    """Given a destination, execute an option that goes to the destination with
    self-consistent Boltzmann policy with rationality constant beta and discount
    rate g. Optionally we can be provided with a converged value and partition
    function, which if we are using a lot we can pre-compute and reuse"""
    if vals[0] == -1: #Default value
        vals, Z  = value_iteration(adj_list, [], beta, g, 0, destination)
    else:
        vals = vals[:]
        Z = Z[:]
        states_chains = []
    for _ in range(n_repeat):
        states = birl_agent(start, destination, [], beta, g, 0, vals=vals, Z=Z)
        states_chains.append(states)
    if states_chains[0] == None: #States are disconnected
        return None, -1, g
    else:
        #        print states_chains
        return states_chains, np.mean([expsum(len(state), g) for state in states_chains]), np.mean([len(state) for state in states_chains])
    
def expsum(n=5, base=1):
    tot = 0
    for m in range(n):
        tot += -1*base**(m)
    return tot

# def compute_accuracy(path, k, n, eta):
#     """Given a path, take the first k nodes in the path and use them to form a 
#     probability distribution over the possible goals, using hyperparameters eta.
#     Then compute the accuracy of the sytem at choosing between the actual target
#     u_{k+1} and a target drawn at random from the other neighbours of u__k"""
#     beta, skills, g = eta
#     target = path[k]
#     uu = [x for x in adj_list[path[k-1]] if x != target] #To compare against
#     u_prime = np.random.choice(uu)
#     sum_1 = 0
#     sum_2 = 0
#     vals = -1*np.ones((4604))
#     for theta in adjusted_adj_list_4.keys(): #Summing over all possible goals
#         vals, Zs = load_theta(theta)
#         lik, vals, Zs = compute_likelihood(path[:k], skills, beta,
#                                            g=g, theta=theta, init_val=vals)
#         print 'Done theta == {}'.format(theta)
#         for act in [target, u_prime]:
#             if act == theta:
#                 prob = lik * np.exp(beta*(20)) / Zs[path[k-1]]
#             else:
#                 prob = lik * np.exp(beta*(-1 + g*vals[act])) / Zs[path[k-1]]
#             # else: #Compound option - currently ignored :/
#             #     to_add = np.exp(beta*(out[0] + (np.pow(g,out[2]))*vals[tuple(o[0])])) / Zs[tuple(state)]
#             if act == target:
#                 sum_1 += prob
#             else:
#                 sum_2 += prob
#         print "Sum_true == {} and Sum_wrong == {}, with prob {}".format(
#             sum_1, sum_2, (sum_1/sum_2) / (sum_1/sum_2 + 1))
#     return prob1/prob2
def compute_probs(path, k, eta, povs, use_cached=True):
    """Given a path, compute the probability of the true theta being chosen compared to a 
    node chosen uniformly at random amongst the nodes which have the same shortest path 
    length to u_k as theta"""
    theta = path[-1]
    sps = np.load('./short_paths/' + str(path[k-1]) + '.npy')    
    length_away = sps[theta]
    theta_f = np.random.choice(choose_k_away(length_away, path[k-1]))
    while theta_f == theta:
        theta_f = np.random.choice(choose_k_away(length_away, path[k-1]))
        print 'looping'
    u_1k = path[:k]
    vals, Zs = load_theta(theta, eta)
    vals_f, Zs_f = load_theta(theta_f, eta)    
    p1 = compute_likelihood_given(u_1k, eta[1], eta[0], eta[2], theta, vals, Zs, povs)
    p2 = compute_likelihood_given(u_1k, eta[1], eta[0], eta[2], theta_f, vals_f, Zs_f, povs)
    if p1 > p2:
        return 1
    else:
        return 0
    #    return p1 / (p1 + p2) #Probability of correct choice between theta and theta_f

def plot_bar():
    betas = np.array([0.4, 0.6, 0.8, 1.0, 1.2])
    results = np.array([[197.7, 189.3, 186.661, 187.439, 190.833], [145.366, 144.608, 148.250, 153.98, 161.03],
                        [124.73, 127.081, 135.5, 140.261, 148.44], [113, 117.460, 124.3, 132.417, 141.2, ],
                        [105.5, 109.882, 117.248, 125.753, 134.773],
                        [94.935, 100.400, 108.7, 117.917, 127.449]])
    fig, ax = plt.subplots()
    colors = ['1.0', '0.75', '0.55', '0.35', '0.2',  '0.1', '0.0']
    le = ['0', '25', '50', '75', '100', '150']
    width = 0.02
    for n in range(6):
        print results[n]
        ax.bar(betas + width*n, results[n], width, color=colors[n],
               label=le[n], linewidth=0.6, edgecolor='0.0')
    ax.legend(loc='upper right')
    ax.set_ylabel('NLML / 1000')
    ax.set_xlabel(r'$\beta$')
    ax.set_xticks(betas + 4*width / 2)
    ax.set_xticklabels(['0.4', '0.6', '0.8', '1.0', '1.2'])
    #ax.set_xticks([0.8, 1.0, 1.2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.show()

def do_max_likelihood(data, eta, povs):
    import math
    log_likelihood = 0
    for index, path in enumerate(data):
        if len(data) == 1:
            print data
        vals, Zs = load_theta(path[-1], eta)        
        if log_likelihood == 0:
            log_likelihood = np.log(compute_likelihood_given(path, eta[1], eta[0], eta[2], path[-1], vals, Zs, povs))
        else:
            ll = np.log(compute_likelihood_given(path, eta[1], eta[0], eta[2], path[-1], vals, Zs, povs))
            if not math.isinf(ll):
                log_likelihood += ll
            else:
                print 'infinite ll at index {} with theta {}'.format(index, path[-1])
        if index % 1000 == 0:
            print 'Computing likelihood, on datum {} out of {}'.format(index, len(data))
    return log_likelihood
def multi_log(a):
    return do_max_likelihood(a[0], a[1], a[2])
def distributed_log_likelihood(data, eta, povs, n_cores):
    total = 0
    pool = multiprocessing.Pool(processes=n_cores)
    data_lists = data
    data_lists = np.array_split(data_lists, n_cores)
    #    for x in range(n_cores):
    #        print 'dl is', data_lists[x]
    #        print 'eta is', eta
    #        print 'povs are', povs
    iters = [[list(data_lists[x]), eta, povs] for x in range(n_cores)]
    #    print len(iters)
    out = pool.map(multi_log, iters)
    return np.sum(out)
    
def choose_k_away(k, node, precomputed=True):
    """"Select all of the nodes that have a shortest path length of k away from node"""
    if precomputed:
        sps = np.load('./short_paths/' + str(node) + '.npy')
    else:
        sps = shortest_path(node)
    nodes = []
    for index, x in enumerate(sps):
        if x == k and index in poss_targets:
            nodes.append(index)
    return nodes

def shortest_path(theta, adj=adjusted_adj_list_4):
    """Returns an array with the shortest path from node theta to the node. 
    We use a simple implementation of Dijsktra's algorithm"""
    dists = 50*np.ones((4604))
    visited = np.zeros((4604), dtype='b')
    #
    current_node = theta
#    visited[theta] = True
    count = 0
    dists[theta] = 0
    while visited[current_node] == False:
        cur = dists[current_node]
        for other_node in adj[current_node]:
            if dists[other_node] > cur + 1:
                dists[other_node] = cur + 1
        visited[current_node] = True
        x1 = set(np.where(visited == False)[0])
        x2 = set(adj.keys())
        x  = x1.intersection(x2)
#        x = [x for x in np.where(visited == False)[0] if x in  adj.keys()]
        if len(x) == 0:
            np.save('./short_paths/' + str(theta) + '.npy', dists)            
            return dists
        else:
            current_node = x.pop()
        count += 1
        
def compute_accuracy(paths, k, n, eta, povs, samples=None):
    outs = []
    if samples == None:
        to_test = [x for x in paths if len(x) == n]
    else:
        to_test = np.random.choice([x for x in paths if len(x) == n], size=samples)
    for index, path in enumerate(to_test):
        outs.append(compute_probs(path, k, eta, povs))
        if index % 100 == 0:
            print "Evaluating at index {} out of {}".format(index, len(to_test))        
    return outs

def big_analysis(beta0s=[0.5, 0.8, 1.1, 1.4, 1.7], ks=range(6), betaps=[1.2, 1.5, 2, 3]):
    """Run a large-scale check over hyperparameters to see if we can get decent explanatory
    power with this model. """

def cache_thetas(eta, povs, start=0, end=None):
    """Run value iteration and save the value functions and normalisations corresponding to
    the relevant thetas for all thetas in the maximum likelihood set."""
    import os
    if end == None:
        end = len(adjusted_adj_list_4.keys())-1
    if len(eta[1]) > 0:
        if not os.path.isdir('./thetas/' + str(eta[0]) + '/' + str(len(eta[1])) + '/' + str(eta[1][0][1])): #To store values, Zs
            os.makedirs('./thetas/' + str(eta[0]) + '/' + str(len(eta[1])) + '/' + str(eta[1][0][1]))
        vals = np.random.random(4604)
        for index, theta in enumerate(adjusted_adj_list_4.keys()[start:end]):
            if (os.path.exists('./thetas/' + str(eta[0]) + '/' + str(len(eta[1])) + '/' + str(eta[1][0][1]) + '/' + str(theta) + '_vals.npy') and
                os.path.exists('./thetas/' + str(eta[0]) + '/' + str(len(eta[1])) + '/' + str(eta[1][0][1]) + '/' + str(theta) + '_Zs.npy')):
                pass
            else:
                vals, Zs = value_iteration(adjusted_adj_list_4, eta[1], eta[0], eta[2], goal=theta,
                                           v_init=vals, skill_vals_precomputed=povs, early_quit=True)
                if vals[0] == None:
                    vals[0] = -1
                    pass
                else:                    
                    np.save('./thetas/' + str(eta[0]) + '/' + str(len(eta[1])) + '/' + str(eta[1][0][1]) + '/' + str(theta) + '_vals', vals)
                    np.save('./thetas/' + str(eta[0]) + '/' + str(len(eta[1])) + '/' + str(eta[1][0][1]) + '/' + str(theta) + '_Zs', Zs)
            if index % 20 == 0:
                print 'Up to index, theta = {} {}'.format(index, theta)
            
    else:
        if not os.path.isdir('./thetas/' + str(eta[0]) + '/' + '0/'): #To store values, Zs
            os.makedirs('./thetas/' + str(eta[0]) + '/' + '0/' )
        vals = np.random.random(4604)
        for theta in adjusted_adj_list_4.keys()[start:end]:
            if (os.path.exists('./thetas/' + str(eta[0]) + '/' + '0' + '/' + str(theta) + '_vals.npy') and
                (os.path.exists('./thetas/' + str(eta[0]) + '/' + '0' + '/' + str(theta) + '_Zs.npy'))):
                pass
            else:
                vals, Zs = value_iteration(adjusted_adj_list_4, eta[1], eta[0], eta[2], goal=theta,
                                           v_init=vals, skill_vals_precomputed=povs, early_quit=True)
                if vals[0] == None:
                    vals[0] = -1                    
                    pass
                else:
                    np.save('./thetas/' + str(eta[0]) + '/' + '0' + '/' + str(theta) + '_vals', vals)
                    np.save('./thetas/' + str(eta[0]) + '/' + '0' + '/' + str(theta) + '_Zs', Zs)
#            print 'Up to theta = {}'.format(theta)

def cache_thetas_multicore(eta, povs, n_cores, start=0, stop=(len(adjusted_adj_list_4.keys())-1)):
    """Run value iteration and save the value functions and normalisations corresponding to
    the relevant thetas for all thetas in the maximum likelihood set."""
    import os
    import multiprocessing
    processes=[]
    for x in range(n_cores):
        theta_lists = adjusted_adj_list_4.keys()[start:stop]
        theta_lists = np.array_split(theta_lists, n_cores)
        p = multiprocessing.Process(target=cache_thetas,
                                    args = [eta, povs,theta_lists[x][0],
                                            theta_lists[x][-1]])
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
                                
def load_theta(theta, eta):
        """Given a theta, load and return value function and normalization factors corresponding
        to that theta."""
        
        if len(eta[1]) == 0: #i.e. there are hierarchical skills available
            vals = np.load('./thetas/' + str(eta[0]) + '/0/' + str(theta) + '_vals.npy')
            Zs = np.load('./thetas/' +  str(eta[0]) + '/0/' + str(theta) + '_Zs.npy')
        else:
            vals = np.load('./thetas/' + str(eta[0]) + '/' + str(len(eta[1])) + '/' + str(eta[1][0][1]) + '/' + str(theta) + '_vals.npy')
            Zs = np.load('./thetas/' + str(eta[0]) + '/' + str(len(eta[1])) + '/' + str(eta[1][0][1]) + '/' + str(theta) + '_Zs.npy')
        return vals, Zs

def compute_accuracies(paths, eta, povs, ks=[2,3,4], ns=[5,6,7], samples=None):
    results = np.zeros((max(ks), max(ns)))
    for n in ns:
        for k in ks:
            if k < n: #Don't try to predict theta when actually given theta
                print "Predicting theta with path length {} after seeing {} noedes".format(n, k) 
                results[k-1, n-1] = np.mean(compute_accuracy(paths, k, n, eta, povs))
    return results

def load_pov(eta):
    pov = [[] for x in range(5)]
    names = ['rewards', 'dests', 'discounts', 'skill_vals_2', 'skill_Zs_2']
    for index, name in enumerate(names):
        pov[index] = np.load('./povs/' + str(len(eta[1])) + '/' + str(eta[1][0][1]) + '/' + str(names[index] + '.npy'))
    return pov

def run_save(beta, k, beta1, cores, g=0.88):
    """Create Pov and cache thetas for a model with beta, g, k hierarchical
    top methods and beta1 inside them"""
    eta = (beta, [[x, beta1] for x in count_popular(train)[0][:k]], g)
    if len(eta[1]) > 0:
        if os.path.exists('./povs/' + str(len(eta[1])) + '/' + str(eta[1][0][1])):
            pov = load_pov(eta)
        else:
            pov = precompute_options_values_multipool(adjusted_adj_list_4, eta[1], g)
    else:
        pov = []
    cache_thetas_multicore(eta, pov, cores)
    cache_thetas((eta[0], eta[1], 0.85), pov) #Clean up any that got missed out
    return distributed_log_likelihood(train, eta, pov, cores)

def grid_search(betas, ks, beta1s, cores):
    outs = []
    for beta in betas:
        for k in ks:
            if k != 0:
                for beta1 in beta1s:
                    out = run_save(beta, k, beta1, cores)
                    outs.append([beta, k, beta1, out])
                    print outs
            else:
                out = run_save(beta, k, 0, cores)
                outs.append([beta, k, 0, out])
    return outs

def plot_accuracies(out, out1, plot_west=True):
    west_accs_2 = np.array([[0.628, 0.637, 0.590],
                            [0.81, 0.77, 0.75],
                            [0.805, 0.830, 0.815]])
    west_baseline = np.array([[0.60, 0.540, 0.530],
                            [0.74, 0.685, 0.66],
                            [0.77, 0.75, 0.73]])
    fig, axs = plt.subplots(1, 3)    
    xs = [5, 6, 7]
    colors = ['b', 'g', 'r']
    le = ['k = 2', 'k = 3', 'k = 4']
    for k in range(3):
        ax = axs[k]
        ax.plot(xs, west_accs_2[k, :], color='k', linewidth=4,
                markersize=10, marker='s', linestyle='--',
                label='West TF-IDF Baseline')
        ax.plot(xs, west_baseline[k, :], color='k', linewidth=4,
                 markersize=10, marker='o', linestyle='--',
                label='West Optimised Model')
        ax.plot(xs, out[k+1, 4:8], color='indianred', linewidth=4,
                 markersize=10, marker='s', label='Non-Hierarchical Model')
        ax.plot(xs, out1[k+1, 4:8], color='indianred', linewidth=4,
                 markersize=10, marker='o', label='Hierarchical Model')
        ax.set_xlim(xmin=5, xmax=7)
        ax.set_xlabel(r'$n$')
        ax.set_xticks(range(5,8))
        ax.set_ylim([0.5, 0.9])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)        
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_title('K = {}'.format(k))
    axs[0].spines['left'].set_visible(True)
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='upper left')
    axs[0].set_yticks(np.arange(0.5, 1.0, 0.1)) 
    plt.tight_layout(w_pad=0.2)
    plt.show()
    

    
#For testing
#sample_lengths(5, 0, paths, 1.2, 0.9, 0, [[4397, 3]], null_pov)
np1 = np.ones((4604, 1))
np1[:,0] = -1*np.ones((4604))
np2 = np.ones((4604, 1))
np2[:, 0] = np.array([x for x in range(4604)])
null_pov = (np1, np2, 0.9)
eta__1 = (1.0, [], 0.88)
eta__2 = (1.0, [[x, 3.0] for x in count_popular(train)[0][:20]], 0.88)
eta_0 = (1.2, [], 0.88)
eta_1 = (1.2, [[x, 2.0] for x in count_popular(train)[0][:5]], 0.88)
eta_4 = (1.0, [[x, 3.0] for x in count_popular(train)[0][:20]], 0.88)
eta_5 = (0.8, [[x, 3.0] for x in count_popular(train)[0][:5]], 0.88)
eta_6 = (1.0, [[x, 3.0] for x in count_popular(train)[0][:50]], 0.88)
eta_7 = (0.8, [[x, 3.0] for x in count_popular(train)[0][:75]], 0.88)
eta_10 = (0.6, [[x, 3.0] for x in count_popular(train)[0][:100]], 0.88)
eta_hh = (0.4, [[x, 3.0] for x in count_popular(train)[0][:150]], 0.88)


O_1 = [1649, 1439, 4293, 4040]
##############################################################################################################
###################################################################################################################
# if True:                                                                                                   #    #
#     povs_2 = [np.load(x) for x in ['./povs/2_rewards.npy', './povs/2_dests.npy', './povs/2_discounts.npy', #    #
#                                    './povs/2_vals.npy', './povs/2_Zs.npy']]                                #    #
#     povs_2[0] = np.array(povs_2[0], dtype='int')                                                                #
#                                                                                                                 #
#     povs_4 = [np.load(x) for x in ['./pov3.020rewards.npy', './pov3.020dests.npy', './pov3.020discounts.npy', # #
#                                    './pov3.020vals.npy', './pov3.020Zs.npy']]                                #  #
#     povs_4[1] = np.array(povs_4, dtype='int')                                                                   #
##     do_max_likelihood(train, eta_4, povs_4)                                                                     #
###################################################################################################################
##############################################################################################################
del(train[4322])
del(train[19545])
del(train[9156])
train = [x for x in train if len(x) < 20]
o_dict = {x: index for index, x in enumerate(count_popular(train)[0][:200])}
if __name__ == '__main__':
    import sys
    grid_search([0.8, 1.0, 1.2], [0, 25, 50, 75], [3.0], 36)    
    cores = 32
    tags = count_popular(train)[0][:50]
    gg = [[x, 3.2] for x in tags[:10]] + [[x, 2.8] for x in tags[10:20]] + [[x, 2.2] for x in tags[20:50]]
    eta = (1.0, gg, 0.88)
    pov = precompute_options_values_multipool(adjusted_adj_list_4, eta[1], 0.88)    
    cache_thetas_multicore(eta, pov, cores)
    cache_thetas((eta[0], eta[1], 0.85), pov) #Clean up any that got missed out
    print distributed_log_likelihood(train, eta, pov, cores)    
    grid_search([1.0], [30, 40, 50], [2.0, 3.0, 3.6], 36)
    run_save(float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]),
             int(sys.argv[4]))
    
    ###############################################################
    # pov_2n = precompute_options_values(adjusted_adj_list_4, [   #
    #     [x, 2] for x in count_popular(train)[0][:20]], 0.88)    #
    # np.save('./pov3rewards.npy', pov_n[0])                      #
    # np.save('./pov3dests.npy', pov_n[1])                        #
    # np.save('./pov3discounts.npy', pov_n[2])                    #
    # np.save('./pov3vals.npy', pov_n[3])                         #
    # np.save('./pov3Zs.npy', pov_n[4])                           #
    # cache_thetas_multicore(eta__2, pov_3n, 4)                   #
    # sample_lengths(10, 0, paths, 1.2, 0.9, 0, [[4397, 3]], pov) #
    ###############################################################
