# Very messy code to carry out an MCMC procedure for finding rewards
# for agents that choice between hierarchical options according to a
# Boltzmann-rational policy, specifically for the modified taxi driver
# example given in BHIRL paper.  We parameterise the state as [x, y,
# passenger_location, passenger_destination] and pass around
# trajectories as lists of [action, state] lists.

# The key functions are gen_option_trajectories, which generates all
# trajectories of options that are consistent with the observed
# trajectories of actions, and prob_consistent, which generates the
# probability that a given option would generate the observed
# trajectory of actions. We can precompute the results of
# gen_option_trajectories if we have fixed options.

# We also have some functions that take in a list of trajectories and
# return a joint probability by working out the product of the
# individual results.
# Chris Cundy 2017-11-26


import numpy as np
import itertools
import matplotlib.pyplot as plt
import random

# import seaborn as sns
import sys

np.set_printoptions(precision=3)
from itertools import chain, combinations
import multiprocessing
import os
import time
import scipy.stats
import matplotlib

# |R1|R|| | |G |
# |  | || | |  |
# |  | |  | |  |
# | || | || |  |
# |Y|| | ||B|B1|
# Represent the state as a list of 4 numbers with
# 1st and 2nd being the taxi position in X,Y coords, the 3rd being the passenger location (either R,G,B,Y, or taxi), and the 4th being the destination (either R,G,B,Y)

# Globals

tol = 1e-4
loc_dict = {  # Translate between location and state flag
    0: [1, 0],
    1: [4, 0],
    2: [3, 4],
    3: [0, 4],
}

# Data
atomic_actions = ["N", "E", "S", "W", "Pickup", "Putdown"]
O_a = [
    ["N", [2, 3, 0, 2]],
    ["W", [2, 2, 0, 2]],
    ["W", [1, 2, 0, 2]],
    ["N", [0, 2, 0, 2]],
    ["N", [0, 1, 0, 2]],
    ["E", [0, 0, 0, 2]],
    ["Pickup", [1, 0, 0, 2]],
    ["S", [1, 0, 4, 2]],
    ["S", [1, 1, 4, 2]],
    ["E", [1, 2, 4, 2]],
    ["E", [2, 2, 4, 2]],
    ["E", [3, 2, 4, 2]],
    ["S", [4, 2, 4, 2]],
    ["S", [4, 3, 4, 2]],
    ["W", [4, 4, 4, 2]],
    ["Putdown", [3, 4, 4, 2]],
    ["foo", None],
]
Os_1 = [
    "N",
    "E",
    "S",
    "W",
    "Pickup",
    "Putdown",
    ["Goto", [0, 0]],
    ["Goto", [4, 0]],
    ["Goto", [4, 4]],
    ["Goto", [0, 4]],
]

Os_3 = ["N", "E", "S", "W", "Pickup", "Putdown", ["Goto", [0, 0]], ["Goto", [4, 4]]]

O_a = [
    ["N", [2, 3, 0, 2]],
    ["W", [2, 2, 0, 2]],
    ["W", [1, 2, 0, 2]],
    ["N", [0, 2, 0, 2]],
    ["N", [0, 1, 0, 2]],
    ["E", [0, 0, 0, 2]],
    ["Pickup", [1, 0, 0, 2]],
    ["S", [1, 0, 4, 2]],
    ["S", [1, 1, 4, 2]],
    ["E", [1, 2, 4, 2]],
    ["E", [2, 2, 4, 2]],
    ["E", [3, 2, 4, 2]],
    ["S", [4, 2, 4, 2]],
    ["S", [4, 3, 4, 2]],
    ["W", [4, 4, 4, 2]],
    ["Putdown", [3, 4, 4, 2]],
    ["foo", None],
]
Os_2 = ["N", "E", "S", "W", "Pickup", "Putdown"]

O_a2 = [
    ["N", [2, 3, 0, 2]],
    ["W", [2, 2, 0, 2]],
    ["W", [1, 2, 0, 2]],
    ["N", [0, 2, 0, 2]],
    ["N", [0, 1, 0, 2]],
    ["E", [0, 0, 0, 2]],
    ["Pickup", [1, 0, 0, 2]],
    ["S", [1, 0, 4, 2]],
    ["S", [1, 1, 4, 2]],
    ["E", [1, 2, 4, 2]],
    ["E", [2, 2, 4, 2]],
    ["S", [3, 2, 4, 2]],
    ["S", [3, 3, 4, 2]],
    ["Putdown", [3, 4, 4, 2]],
    ["foo", None],
]

O_a3 = [
    ["N", [2, 3, 0, 1]],
    ["W", [2, 2, 0, 1]],
    ["W", [1, 2, 0, 1]],
    ["N", [0, 2, 0, 1]],
    ["N", [0, 1, 0, 1]],
    ["E", [0, 0, 0, 1]],
    ["Pickup", [1, 0, 0, 1]],
    ["S", [1, 0, 4, 1]],
    ["S", [1, 1, 4, 1]],
    ["E", [1, 2, 4, 1]],
    ["E", [2, 2, 4, 1]],
    ["E", [3, 2, 4, 1]],
    ["N", [4, 2, 4, 1]],
    ["N", [4, 1, 4, 1]],
    ["Putdown", [4, 0, 4, 1]],
    ["foo", None],
]

np.random.seed(2017)


def find_actions(O):
    """Given a trajectory of states, return the actions that must ahve been
    taken in order to get you that trajectory"""
    act_list = []
    for index, s in enumerate(O[:-1]):
        flag = True
        for aa in atomic_actions:
            if action(aa, s)[0] == O[index + 1] and flag:
                act_list.append(aa)
                flag = False
    return act_list


def action(a, s, theta=[-1] * 4):
    """Given a state s and an action a, execute the action a on the state and
    return the resulting state s1 and the awarded reward r. When the episode
    has terminated, the state is set to the special value of None. We 
    parameterise the reward function as a 6 element vector which can take on values
    n \in [0, 25], with n meaning a zero reward in the [n/ 5, n % 5] place. This represents
    some squares being 'free' to go into, maybe because the human enjoys taking those paths."""
    s1 = s[:]
    s_list = [
        s1[:]
    ]  # Might be overzealous copying going on, but got burnt by this before
    r = 0
    if s == None:  # In the special 'termination' state
        # Stay in the terminal state, don't award a reward
        s_list.append(s1)
    else:
        if a == "N":
            if not (s[1] == 0):
                s1[1] = s[1] - 1
                r = -1
                s_list.append(s1)
            else:
                r = -1
                s_list.append(s1)
        elif a == "S":
            if not (s[1] == 4):
                s1[1] = s1[1] + 1
                r = -1
                s_list.append(s1)
            else:
                r = -1
        elif a == "E":  # Have to deal with the walls here
            if not (
                s[0] == 4
                or (s[1] > 2 and (s[0] == 0 or s[0] == 2))
                or (s[1] < 2 and s[0] == 1)
            ):
                s1[0] = s[0] + 1
                r = -1
                s_list.append(s1)
            else:
                r = -1
                s_list.append(s1)
        elif a == "W":  # Have to deal with the walls here
            if not (
                s[0] == 0
                or (
                    ((s[1] > 2 and (s[0] == 1 or s[0] == 3)))
                    or (s[1] < 2 and s[0] == 2)
                )
            ):
                s1[0] = s[0] - 1
                r = -1
                s_list.append(s1)
            else:
                r = -1
                s_list.append(s1)
        elif a == "Pickup":
            if s[2] == 4:  # Can't pickup when the passenger's already in
                r = -10
                s_list.append(s1)
            elif loc_dict[s[2]] == [
                s[0],
                s[1],
            ]:  # Is the passenger at the taxi's location? * scaling
                s1[2] = 4  # Put the passenger in the taxi
                r = -1
                s_list.append(s1)
            else:
                r = -10  # Improper activation of the action
                s_list.append(s1)
        elif a == "Putdown":
            if (
                loc_dict[s[3]] == [s[0], s[1]] and s[2] == 4
            ):  # Is the taxi at the destination?
                r = 19  # Taxi has succeeded!
                s1 = None
                s_list.append(s1)
            else:
                r = -10
                s_list.append(s1)
        if s1 != None and (s1[0] + s1[1] * 5) in theta:  # The cell has a zero there
            r = r + 1  # Reward 0 instead of -1 for going into this state
    return s1, r, s_list


def what_zone(coords):
    """The grid is essentially partitioned into four zones where the R,G,B,Y
    points are located. This function returns the zone the coords are in"""
    x = coords[0]
    y = coords[1]
    if y < 2:
        if x < 2:
            return "R"
        if x >= 2:
            return "G"
    elif y > 2:
        if x < 1:
            return "Y"
        if x > 2:
            return "B"
    else:
        return None


def option(o, s, g=0.999, theta=[]):
    """Given an option o, execute the option on the state and return
    the resulting state s1 and the accumulated (discounted) reward r.
    The available options are each of the individual actions, as well
    as a compound option, ['Goto', [dest_x_coord, dest_y_coord]]"""
    if type(o) != list:  # Not a complex option
        return action(o, s, theta) + (o,)
    else:
        dest_x = o[1][0]
        dest_y = o[1][1]
        s_list = []
        a_list = []
        s1 = s[:]
        s_list.append(s1)
        r = 0
        count = 1

        while s1[0] != dest_x or s1[1] != dest_y:
            if what_zone([dest_x, dest_y]) == what_zone(s1):
                if s1[1] < dest_y:
                    s1, tr, _ = action("S", s1, theta)
                    a_list.append("S")
                elif s1[1] > dest_y:
                    s1, tr, _ = action("N", s1, theta)
                    a_list.append("N")
                elif s1[1] == dest_y:
                    if s1[0] < dest_x:
                        s1, tr, _ = action("E", s1, theta)
                        a_list.append("E")
                    elif s1[0] > dest_x:  # equality case should never happen
                        s1, tr, _ = action("W", s1, theta)
                        a_list.append("W")

            else:  # In different zone
                if s1[1] > 2:
                    s1, tr, _ = action("N", s1, theta)
                    a_list.append("N")
                elif s1[1] < 2:
                    s1, tr, _ = action("S", s1, theta)
                    a_list.append("S")
                elif s1[1] == 2:
                    if s1[0] < dest_x:
                        s1, tr, _ = action("E", s1, theta)
                        a_list.append("E")
                    elif s1[0] > dest_x:
                        s1, tr, _ = action("W", s1, theta)
                        a_list.append("W")
                    elif s1[0] == dest_x:
                        if s1[1] > dest_y:
                            s1, tr, _ = action("N", s1, theta)
                            a_list.append("N")
                        elif s1[1] < dest_y:
                            s1, tr, _ = action("S", s1, theta)
                            a_list.append("S")
            r += tr * (g ** (count - 1))  # As we multiply by g to get reward
            s_list.append(s1)
            count += 1
        return s1, r, s_list, a_list


def prob_consistent(O_a, o, theta=[]):
    """Given an action-trajectory O_a and an option o, return the probability
    that choosing the option in the first state O_a[0][1] will result in an
    action-trajectory that is given by O_a."""
    return 1 if option(o, O_a[0][1], 0.99, theta)[2] == list(zip(*O_a)[1]) else 0


def gen_option_trajectories(O_a, Os, theta=[]):
    """Given an action-trajectory O_a specified as a list [[a_0, s_0], [a_1, s_1],
    ... [a_2, s_2]], generate all sets of option-trajectories that are
    consistent with the action-trajectory. Do this following algorithm 1 in the
    literature provided."""
    T_o = [[[]] for i in range(len(O_a))]  # Generate empty list-of-lists-of-lists
    for i in range(len(O_a)):
        if T_o[i] != [] or i == 0:
            for k in range(1, len(O_a) - i):
                for o in Os:
                    # print o, k
                    # print prob_consistent(O_a[i:i+k+1], o) != 0
                    if prob_consistent(O_a[i : i + k + 1], o, theta) != 0:
                        to_add = [[]]
                        for t in T_o[i]:
                            if to_add == [[]]:
                                to_add = [t + [o]]
                            else:
                                to_add.append(t + [o])
                        for t1 in to_add:
                            if T_o[i + k] == [[]]:
                                T_o[i + k] = [t1]
                            else:
                                T_o[i + k].append(t1)
    #                        T_o[i+k].extend(to_add) #Add to trajectories getting to i+k
    return T_o[-1]  # Return all trajectories that get to the final state.


def value_iteration(
    beta, skills, g=0.99, theta=[], v_init=np.random.random((5, 5, 5, 4))
):
    """Computes the self-consistent Boltzmann value function for a gridworld
    with a given set of skills. At the moment this is slightly wrong as the
    extended options return cumulative rewards that are actually undiscounted.
    Shouldn't matter as long as the discount rate is high. We can give
    the value iteration an initialisation to speed up convergence."""
    Vs_0 = -1 * np.ones([6, 5, 5, 4])  # To hold values of grid states. The
    Vs_1 = np.concatenate(
        [v_init, np.zeros((1, 5, 5, 4))]
    )  # To deal with termination state
    count = 0
    tmp = np.array([0] * len(skills), dtype="float32")
    # First we build a matrix with all the rewards and destinations for each skill
    rewards = np.zeros((5, 5, 5, 4, len(skills)))
    dests = np.zeros(
        (6, 5, 5, 4, len(skills), 4), dtype="int"
    )  # Make it bigger so we can
    dests[5, :, :, :, :] = 0  # cache destinations and rewards
    strides = np.array(Vs_1.strides)
    itemsize = Vs_1.itemsize
    zo = np.zeros((6, 5, 5, 4, len(skills)), dtype="int")
    for index, _ in np.ndenumerate(Vs_1[:5, :, :, :]):
        for index1, skill in enumerate(skills):
            out = option(skill, list(index), g, theta)
            if out[0] == None:
                rewards[tuple(list(index) + [index1])] = 19
                dests[tuple(list(index) + [index1])] = [5, 0, 0, 0]
            else:
                rewards[tuple(list(index) + [index1])] = out[1]
                dests[tuple(list(index) + [index1])] = out[0]
    for index, _ in np.ndenumerate(Vs_1[:5, :, :, :]):
        dd = dests[tuple(index)]
        #       print np.dot(dd, strides)
        zo[index] = np.dot(dd, strides) / itemsize
    while abs(np.amax(Vs_0[:5, :, :, :] - Vs_1[:5, :, :, :])) > tol:
        np.copyto(Vs_0, Vs_1)
        Vs_1 = Vs_1 * 0
        for index, _ in np.ndenumerate(Vs_1[:5, :, :, :]):  # Enumerate through 2d array
            val = Vs_0.flat[zo[index]]
            tmp = rewards[tuple(index)] + g * val
            zz = np.exp(beta * tmp)
            Vs_1[index] = np.sum(tmp * zz) / np.sum(zz)
            tmp *= 0
        count += 1
    Zs = np.zeros(
        np.shape(Vs_1[:5, :, :, :])
    )  # Also return the computed normalization factors
    for index, _ in np.ndenumerate(Vs_1[:5, :, :, :]):
        Z = 0
        for skill in skills:
            out = option(skill, list(index), g, theta)
            if out[0] == None:  # In termination state,
                Z += np.exp(beta * (19 + 0))  # transition reward is 19, state is 0.
            else:
                Z += np.exp(beta * (out[1] + g * Vs_1[tuple(out[0])]))
        Zs[index] = Z
    return Vs_1[:5, :, :, :], Zs


def compute_likelihood(
    O_a, Os, beta, g=0.99, theta=[], init_val=np.random.random((5, 5, 5, 4))
):
    """Given an observed action-trajectory O_a, and a set of available actions
    Os, compute the likelihood of taking that trajectory given the reward function."""

    O_os = gen_option_trajectories(O_a, Os, theta)  # Generate all option-trajectories
    vals, Zs = value_iteration(beta, Os, g, theta, init_val)  # Do value iteration
    probs = []
    for O_o in O_os:  # For option-trajectory in option-trajectories
        state = O_a[0][1]  # Initial state
        prob = 0
        for o in O_o:  # For option in option-trajectory
            # Now compute the prob of choosing the option o that was actually chosen
            out = option(o, state, g, theta)
            if out[0] == None:
                to_add = np.exp(beta * (19 + 0)) / Zs[tuple(state)]
            else:
                to_add = (
                    np.exp(beta * (out[1] + g * vals[tuple(out[0])])) / Zs[tuple(state)]
                )
            if prob == 0:
                prob = to_add
            else:
                prob *= to_add
            state = out[0]
        probs.append(prob)
    return np.sum(probs), vals


def compute_likelihood_list(
    O_a_list,
    Os,
    beta,
    g=0.99,
    theta=[],
    init_vals=np.array([-1]),
    given_flag=False,
    gO_os=None,
    inits_given=False,
):
    """Given an observed action-trajectory O_a, and a set of available actions
    Os, compute the likelihood of taking that trajectory given the reward function."""
    num_nonzero = 0
    for x in theta:
        if x != -1:
            num_nonzero += 1
    prior = (1.0 / 5) * (1.0 / 25 ** (num_nonzero))
    if not inits_given:  # Not given
        init_vals = np.random.random((5, 5, 5, 4))
    vals, Zs = value_iteration(beta, Os, g, theta, init_vals)  # Do value iteration
    tmp_q = np.zeros([100])
    tmp_z = np.zeros([100])
    for index, o_a in enumerate(O_a_list):
        if given_flag:
            O_os = gO_os[index]
        else:
            O_os = gen_option_trajectories(
                o_a, Os, theta
            )  # Generate all option-trajectories
        probs = []  # Probability of each option-trajectory being chosen
        for O_o in O_os:  # For option-trajectory in option-trajectories
            state = o_a[0][1]  # Initial state
            prob = 0
            tmp_q = tmp_z * 0
            tmp_z = tmp_q * 0
            for index1, o in enumerate(O_o):  # For option in option-trajectory
                # Now compute the prob of choosing the option o that was actually chosen
                out = option(o, state, g, theta)
                if out[0] == None:
                    #                    to_add = np.exp(beta*(19  + 0)) / Zs[tuple(state)]
                    tmp_q[index1] = 19
                    tmp_z[index1] = Zs[tuple(state)]
                else:
                    #                    print(out)
                    #                   to_add = np.exp(beta*(out[1] + g*vals[tuple(out[0])])) / Zs[tuple(state)]
                    tmp_q[index1] = out[1] + g * vals[tuple(out[0])]
                    tmp_z[index1] = Zs[tuple(state)]
                # if prob == 0: #First option
                #     prob = to_add
                # else:
                #     prob *= to_add
                state = out[0]
            prob = np.prod(np.exp(beta * tmp_q[: index1 + 1]) / tmp_z[: index1 + 1])
            probs.append(prob)
        if index == 0:  # First observed trajectory
            p = np.sum(probs)
        else:
            p *= np.sum(probs)
    return p * prior, vals


def option_dict(p):
    if p < 10:
        return ["Goto", [p % 5, p / 5]]
    else:
        if p == 10:
            return ["Goto", [0, 3]]
        if p == 13:
            return ["Goto", [0, 4]]
        else:
            if p == 11 or p == 12:
                return ["Goto", [(p + 2) % 5, 3]]
            if p == 14 or p == 15:
                return ["Goto", [(p + 4) % 5, 4]]


def generate_option_sets(n=2):
    """Generate all sets of n options which go into the zones"""
    option_sets = []
    perms = list(powerset(range(16), n))
    for perm in perms:
        if len(perm) == 1:
            p = perm[0]
            option_sets.append(atomic_actions + [option_dict(p)])
        else:
            p = perm[0]
            q = perm[1]
            option_set = atomic_actions[:]
            option_sets.append(atomic_actions + [option_dict(p), option_dict(q)])
    return option_sets


def powerset(iterable, n):
    """Return the powerset of the given list, excluding the empty set"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, n + 1))


def gen_option_set(in_list):
    """Given a number representing the option set, return the option-set consisting
    of the atomic options and the compound options"""
    option_set = atomic_actions[:]
    for item in in_list:
        if item == 16:
            pass  # 16 represents the null option
        else:
            option_set.append(option_dict(item))
    return option_set


def MCMC(O_a_list, Os, beta, g, n, seed=0, n_burn=0, gO_s=None):
    np.random.seed(seed)
    R = np.ndarray.tolist(np.random.randint(low=-1, high=25, size=5))
    #    R = [-1, -1, -1, -1, -1]
    R1 = R[:]  # Proposed reward
    p_list = []
    p, vals_list = compute_likelihood_list(
        O_a_list, Os, beta, g, list(R), [-1], True, gO_s
    )
    Z = 0
    data = []
    accepted = 0
    marginals = np.zeros((5, n))
    print "Options are, {}".format(Os)
    for index, _ in enumerate(range(n)):
        if index % 100 == 0:
            print "At step {} out of {}".format(index, n)
        lim = 2  # Can be tuned to get good acceptance ratio ~ 0.2
        for i in np.random.choice(range(5), size=lim, replace=False):
            step = np.random.randint(5)
            if np.random.randint(2) == 0:
                R1[i] += step
            else:
                R1[i] -= step
            if R1[i] < -1:  # Wrap around if we get to the ends
                R1[i] = 24
            if R1[i] > 25:
                R1[i] = -1
        p1, vals1_list = compute_likelihood_list(
            O_a_list, Os, beta, g, list(R1), vals_list, True, gO_s, True
        )
        Z += p1
        change = min(1, p1 / p)  # M-H step
        print "chain {}, beta {}, p is {}, with reward {}".format(seed, beta, p, R)
        if np.random.uniform() < change:
            R = R1[:]
            vals_list = vals1_list[:]
            p = p1
            accepted += 1
        else:
            R1 = R[:]
        marginals[:5, index] = R1
        if index % 10 == 0:
            print "At step {} out of {}, with acceptance rate {}".format(
                index, n, float(accepted) / (index + 1)
            )
    print "Have {} acceptance rate".format(float(accepted) / n)
    if Os == gen_option_set([0, 15]):
        np.save(
            "./data/option_chain_" + str(seed) + "_" + str(len(O_a_list)),
            marginals[:, n_burn:],
        )
    else:
        np.save(
            "./data/irl_chain_" + str(seed) + "_" + str(len(O_a_list)),
            marginals[:, n_burn:],
        )
    return Z, marginals[:, n_burn:]


def options_MCMC(O_a_list, beta, g, n, seed=0, n_burn=0):
    """Given a list of trajectories, do MCMC in the space of latent theta
    and option-sets with up to two options in the set. We represent the
    option-sets as a tuple (a, b) representing the destinations of the
    option. 0 < a < 17, with 17 representing the null option."""
    np.random.seed(seed)
    #    R = np.ndarray.tolist(np.random.randint(low=-1, high=25, size=5))
    R = [-1, -1, -1, -1, -1]
    o_s = np.ndarray.tolist(np.random.randint(low=0, high=17, size=3))
    o_s1 = o_s[:]
    R1 = R[:]
    p_list = []
    vals_list = []
    p, vals_list = compute_likelihood_list(
        O_a_list, gen_option_set(o_s), beta, g, list(R)
    )
    Z = 0
    data = []
    accepted = 0
    marginals = np.zeros((5 + 3, n))
    for index, _ in enumerate(range(n)):
        if np.random.randint(2) == 0:
            lim = 2
            for i in np.random.choice(range(5), size=lim, replace=False):
                if np.random.randint(2) == 0:
                    R1[i] += 1
                else:
                    R1[i] -= 1
                if R1[i] == -2:  # Wrap around if we get to the ends
                    R1[i] = 24
                if R1[i] == 25:
                    R1[i] = -1
        else:
            if np.random.randint(2) == 0:
                j = np.random.randint(3)  # change a random coord in o_s
                if (
                    np.random.randint(2) == 0
                ):  # Randomly choose whether to increase or decrease
                    o_s1[j] += 1
                else:
                    o_s1[j] -= 1
                if o_s1[j] == -1:  # Wrap around if we get to the ends
                    o_s1[j] = 16
                elif o_s1[j] == 17:
                    o_s1[j] = 0
            else:
                k = np.random.randint(17)
                k1 = np.random.randint(17)
                k2 = np.random.randint(17)
                o_s1 = [k, k1, k2]
        p1, vals1_list = compute_likelihood_list(
            O_a_list,
            gen_option_set(o_s1),
            beta,
            g,
            list(R1),
            vals_list,
            inits_given=True,
        )
        change = min(1, p1 / p)
        if np.random.uniform() < change:
            R = R1[:]
            o_s = o_s1[:]
            vals_list = vals1_list[:]
            p = p1
            accepted += 1
        else:
            R1 = R[:]
            o_s1 = o_s[:]
        print "On chain {}, with beta {}, Current p is {}, at latent skills {}".format(
            seed, beta, p, o_s
        )
        marginals[:5, index] = R1
        marginals[5, index] = o_s1[0]
        marginals[6, index] = o_s1[1]
        marginals[7, index] = o_s1[2]
        Z += p1
        if index % 10 == 0:
            print "At step {} out of {}, with acceptance rate {}".format(
                index, n, float(accepted) / (index + 1)
            )
    print "Have {} acceptance rate".format(float(accepted) / n)
    np.save("./data/chain_" + str(seed) + "_" + str(beta), marginals[:, n_burn:])
    return Z / np.shape(marginals)[1], marginals[:, n_burn:]


def multimode_MCMC(O_a_list, beta, g, n, seed=0, n_burn=0, p=5):
    """We have a problem where we might have multiple modes. We solve that 
    problem by identifying the modes and working out Z"""
    Z_list = []
    marg_list = []
    mode_list = []
    np.random.seed(seed)
    for _ in range(p):
        a, b = options_MCMC(O_a_list, beta, g, n, np.random.randint(50000), n_burn=0)
        Z_list.append(a)
        marg_list.append(b)
        mode_list.append(
            np.ndarray.tolist(scipy.stats.mode(b[2:, :].T)[0])[0]
        )  # Stupid function has stupid return behaviour
    print "Z list is ", Z_list
    print "modes are ", mode_list
    non_dupes = set([frozenset(pairs) for pairs in mode_list])
    Zs = np.zeros(len(non_dupes))
    Z_counts = np.zeros(len(non_dupes))
    for index1, mode in enumerate(non_dupes):
        for index2, mode2 in enumerate(mode_list):
            if set(mode2) == set(mode):  # Pairs match, doesn't matter which is which
                Zs[index1] += Z_list[index2]
                Z_counts[index1] += 1
    Zs = Zs / Z_counts
    return np.sum(Zs)


def find_options_lik(O_a_list, theta, beta, g):
    results = np.zeros((17, 17))
    for n in range(17):
        for m in range(17):
            results[n, m] = compute_likelihood_list(
                O_a_list, gen_option_set([n, m]), beta, g, theta
            )[0]
        print "Got to {}".format(n)
    return results


def plot_options_lik(grid, norm=True, alpha=1.0, ax=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if norm:
        grid2 = np.log(grid[:] / np.amax(grid))
    if alpha != 1.0:
        cmap = "Blues"
    else:
        cmap = "Greys"
    plt.figure()
    ax = plt.gca()
    im = ax.matshow(grid2, cmap=cmap, alpha=alpha, vmin=-5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #    plt.xticks(np.arange(16))
    matplotlib.rc("xtick", labelsize=14)
    matplotlib.rc("ytick", labelsize=14)
    #    ax.grid(linestyle='--')
    plt.colorbar(im, cax=cax)
    plt.show()
    return ax


def plot_options_lik_2(grid1, grid2, norm=True):
    ax = plot_options_lik(grid1, norm=True, alpha=1.0)
    plot_options_lik(grid2, norm=True, alpha=0.4, ax=ax)


pool_dict = {0: 0.8, 1: 1.0, 2: 1.2, 3: 1.4, 4: 1.6, 5: 1.8, 6: 2.0, 7: 2.2, 8: 2.4}


def options_multipool_MCMC(O_a_list, beta, g, n, pool_num=4, big_machine=False):
    """Queries to see how many free cores are available on the machine 
    and then creates a set of n_cores processes, each of which evolve 
    separately"""
    processes = []
    if big_machine:
        for x in range(pool_num):
            p = multiprocessing.Process(
                target=options_MCMC,
                args=[O_a_list, pool_dict[x / 4], g, n, np.random.randint(4000), 200],
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        for x in range(pool_num):
            p = multiprocessing.Process(
                target=options_MCMC,
                args=[O_a_list, beta, g, n, np.random.randint(4000), 200],
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


def multipool_MCMC(O_a_list, beta, g, n, pool_num=4, big_machine=False):
    """Queries to see how many free cores are available on the machine
    and then creates a set of n_cores processes, each of which evolve
    separately, doing the standard BIRL (i.e. that we do BHIRL with no
    options)"""
    processes = []
    Os = Os_2
    gO_s = [
        gen_option_trajectories(O_a_list[x], Os, []) for x in range(0, len(O_a_list))
    ]
    if big_machine:
        for x in range(pool_num):
            p = multiprocessing.Process(
                target=MCMC,
                args=[
                    O_a_list,
                    Os_2,
                    pool_dict[x / 4],
                    g,
                    n,
                    np.random.randint(4000),
                    200,
                    gO_s,
                ],
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        for _ in range(pool_num):
            p = multiprocessing.Process(
                target=MCMC,
                args=[O_a_list, Os_2, beta, g, n, np.random.randint(4000), 200, gO_s],
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


def multipool_MCMC_spec_options(
    O_a_list, Os, beta, g, n, pool_num=4, big_machine=False
):
    """Queries to see how many free cores are available on the machine
    and then creates a set of n_cores processes, each of which evolve
    separately, doing the standard BIRL (i.e. that we do BHIRL with no
    options)"""
    processes = []
    gO_s = [
        gen_option_trajectories(O_a_list[x], Os, []) for x in range(0, len(O_a_list))
    ]
    if big_machine:
        for x in range(pool_num):
            p = multiprocessing.Process(
                target=MCMC,
                args=[
                    O_a_list,
                    Os,
                    pool_dict[x / 4],
                    g,
                    n,
                    np.random.randint(4000),
                    200,
                    gO_s,
                ],
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        for _ in range(pool_num):
            p = multiprocessing.Process(
                target=MCMC,
                args=[O_a_list, Os, beta, g, n, np.random.randint(4000), 200, gO_s],
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


def join_margs(directory):
    """Given a directory with many saved numpy arrays consisting of marginal 
    results from MCMC, concatenate the samples together into one chain"""
    chains = []
    files = os.listdir(directory)
    for f in files:
        chains.append(np.load(directory + "/" + f))
    out_array = chains[0]
    for out_arr in chains[1:]:
        out_array = np.concatenate([out_array, out_arr], axis=1)
    return out_array


def plot_theta_margs(marg_dirs1, marg_dirs2, betas=[0.1, 0.5, 1.2, 2]):
    thetas = [[-1] * 5]
    colors = ["r", "0.2"]
    for theta in thetas:
        probs1 = []
        errs1 = []
        probs2 = []
        errs2 = []
        for index, marg in enumerate(marg_dirs1):
            probs1.append(float(join_count_std(marg, theta)[0]))
            errs1.append(float(join_count_std(marg, theta)[1]))
            print "For BHIRL theta = {}, marg {} have {} with uncertainty {}".format(
                theta, betas[index], probs1[index], join_count_std(marg, theta)[1]
            )
        for index, marg in enumerate(marg_dirs2):
            probs2.append(float(join_count_std(marg, theta)[0]))
            errs2.append(float(join_count_std(marg, theta)[1]))
            print "For BIRL theta = {}, marg {} have {} with uncertainty{}".format(
                theta, betas[index], probs2[index], join_count_std(marg, theta)[1]
            )
        (plt_line, cap, barlin) = plt.errorbar(
            betas[: len(marg_dirs1)],
            probs1,
            yerr=errs1,
            linestyle="-",
            marker="o",
            linewidth=1,
            markersize=8,
            color=colors[0],
            label=r"$P_{BHIRL}(\theta|\beta, O_a)$",
            capsize=10,
        )
        plt_line2, cap2, barlin2 = plt.errorbar(
            betas[: len(marg_dirs2)],
            probs2,
            yerr=errs2,
            linestyle="-",
            marker="o",
            linewidth=1,
            markersize=8,
            color=colors[1],
            label=r"$P_{BIRL}(\theta|\beta, O_a)$",
            capsize=10,
        )

        cap[0].set_markeredgewidth(0.5)
        cap2[0].set_markeredgewidth(0.5)
        # plt.grid(True, which='both')
        # plt.minorticks_on()
        plt.xlabel(r"$\beta$", size=20)
        plt.ylabel(r"P(\theta)$", size=20)  #
        plt.legend()
        plt.show()


def brier_accuracy(n, m, beta, num_samples=50000):
    """Generates m trajectories from n agents with Boltzmann-rationality 
    constant beta, and does inference on these trajectories, working out 
    the Brier scores of the predictions from the vague options-prior, and
    the BIRL method."""
    birl_briers = []
    bhirl_briers = []
    for _ in range(n):
        theta = [np.random.randint(2) for _ in range(25)]
        skills = [np.random.randint(16) for _ in range(2)]
        skills = gen_option_set(skills)
        init_state = [
            np.random.randint(5),
            np.random.randint(5),
            np.random.randint(4),
            np.random.randint(4),
        ]
        trajectories = []
        for _ in range(m):
            trajectories.append(
                draw_trajectory(beta, init_state, skills, 0.99, theta)[0]
            )
        # Now work out the probability of the actual theta
        marginals_bhirl = options_MCMC(
            trajectories, beta, g, num_samples, seed=0, n_burn=50
        )
        marginals_irl = MCMC(trajectories, O_s, beta, g, num_samples)
        bhirl_probs = count_theta(marginals_bhirl, theta) / np.shape(marginals_bhirl)[1]
        irl_probs = count_theta(marginals_irl, theta) / np.shape(marginals_irl)[1]
        birl_briers.append((1 - irl_probs) ** 2)
        bhirl_briers.append((1 - bhirl_probs) ** 2)

    return birl_briers, bhirl_briers


def marginalise_options(O_a, n, beta, g=0.99, theta=[]):
    """Compute the probability of observing the trajectory, marginalising 
    over all possible sets of options """
    option_sets = generate_option_sets(n)
    print "Computing {} option-sets with up to {} options in each".format(
        len(option_sets), n
    )
    prior = 1
    prob = 0
    display_count = 8
    for index, option_set in enumerate(option_sets):
        prob += prior * compute_likelihood(O_a, option_set, beta, g, theta)[0]
        if index % display_count == 0:
            print "Done {} option-sets out of {}".format(index, len(option_sets))
    return prob


def plot_margs(marginals):
    if np.shape(marginals)[0] == 5:
        for n in range(4):
            plt.subplot(3, 2, n + 1)
            plt.axis("off")
            plt.plot(
                marginals[n, :],
                linestyle="-",
                marker="o",
                linewidth=1,
                markersize=2,
                color="0.2",
                alpha=0.3,
            )
            plt.plot(
                np.cumsum(marginals[n, :]) / np.arange(np.shape(marginals)[1]),
                linestyle="-",
                marker="o",
                linewidth=0.8,
                markersize=0,
                color="r",
                alpha=0.9,
            )
    else:
        for n in range(2):
            plt.subplot(2, 1, n + 1)
            plt.axis("off")
            plt.plot(
                marginals[n, :],
                linestyle="-",
                marker="o",
                linewidth=1,
                markersize=2,
                color="0.2",
                alpha=0.3,
            )
            plt.plot(
                np.cumsum(marginals[n, :]) / np.arange(np.shape(marginals)[1]),
                linestyle="-",
                marker="o",
                linewidth=0.8,
                markersize=0,
                color="r",
                alpha=0.9,
            )
    plt.show()


def plot_marg_cells(marginals):
    nums = np.zeros((5, 5))
    for n in range(np.shape(marginals)[1]):
        for x in marginals[:4, n]:
            if n != -1:

                nums[int(x) % 5, int(x) / 5] += 1
    return 4 * nums / np.shape(marginals)[1]


def count_theta(margs, theta, size=2):
    count = 0
    for n in range(np.shape(margs)[1]):
        if list(margs[:size, n]) == theta:
            count += 1
    print n
    return count


def count_partial_theta(margs, theta, size=5):
    count = 0
    for n in range(np.shape(margs)[1]):
        tmp = list(margs[:size, n])
        flag = True
        for p in theta:
            if p in tmp:
                del tmp[tmp.index(p)]
            else:
                flag = False
                break
        if flag:
            count += 1
    return count


def count_options(margs, options):
    count = 0
    for n in range(np.shape(margs)[1]):
        tmp = list(margs[2:, n])
        flag = True
        for p in options:
            if p in tmp:
                del tmp[tmp.index(p)]
            else:
                flag = False
                break
        if flag:
            count += 1
    return count


def join_count_std(directory, theta):
    files = os.listdir(directory)
    probs = []
    num = 0
    for f in files:
        probs.append(
            float(count_partial_theta(np.load(directory + "/" + f), theta))
            / np.shape(np.load(directory + "/" + f))[1]
        )
        num += 1
    print probs
    return np.mean(probs), np.std(probs) / np.sqrt(num)


def draw_trajectory(beta, init_state, skills, g, theta):
    """Given a taxi-driver problem specified with an initial state, carry out a
    draw from trajectories of the Boltzmann agent"""
    vals, Z = value_iteration(beta, skills, g, theta)
    s1_list = [init_state]
    a_list = []
    s1 = init_state
    o_list = []
    while s1 != None:
        probs = []
        for o in skills:
            out = option(o, s1, g, theta)
            prob = 0
            if out[0] == None:
                prob = np.exp(beta * (19 + 0)) / Z[tuple(s1)]
            else:
                prob = np.exp(beta * (out[1] + g * vals[tuple(out[0])])) / Z[tuple(s1)]
            probs.append(prob)
        delta = 1 - np.cumsum(probs)[-1]
        probs[-1] += delta
        ff = np.random.choice(range(len(skills)), p=probs)
        o = skills[ff]
        out = option(o, s1, g, theta)
        if len(out[2]) == 2 or (len(out[2]) == 1 and out[3] != []):
            s1_list.append(out[0])
        else:
            s1_list.extend(out[2][1:])
        o_list.append([o, s1, out[0]])
        if len(out[2]) == 2:
            a_list.append(out[3])
        else:
            a_list.extend(out[3])
        s1 = out[0]
        if len(s1_list) > 12:
            print ("--------")
            # print('len state list is', len(s1_list)) #
            # print('len option list is', len(o_list)) #
            # print('len action list is', len(a_list)) #

            ##################################
            print ("state list is", s1_list)  #
            print ("option list is", o_list)  #
            print ("action list is", a_list)  #
            print ("len state list is", len(s1_list))
            print ("len option list is", len(o_list))
            print ("len action list is", len(a_list))
            ##################################
            print ("--------")

    return s1_list + [None], o_list, a_list + ["foo"]


def plot_trajectory(state_list, o_list):
    plt.close()
    x_list = [t[0] for t in state_list[:-1]]
    y_list = [t[1] for t in state_list[:-1]]
    ox_list = []
    oy_list = []
    ox_list = []
    oy_list = []
    for o in o_list:
        if o[0][0] == "Goto":
            ox_list.append(o[1][0])
            oy_list.append(o[1][1])

    y_list = [-y for y in y_list]

    plt.plot(
        x_list,
        y_list,
        linestyle="-",
        marker="o",
        linewidth=1,
        markersize=8,
        color="orange",
    )
    plt.plot(
        ox_list, oy_list, linestyle="", marker="o", linewidth=0, markersize=8, color="b"
    )
    plt.xlim([-0.2, 4.2])
    plt.ylim([-4.2, 0.2])
    plt.grid(True, which="both")
    plt.minorticks_on()
    plt.show()


def find_ordered_thetas(directory):
    prob_max = 0
    theta_max = [-1, -1]
    thetas = []
    init_flag = False
    for theta1 in range(-1, 25):
        for theta2 in range(-1, 25):
            val = join_count_std(directory, [theta1, theta2])[0]
            if init_flag and [theta2, theta1] in zip(*thetas)[0]:
                pass
            else:
                thetas.append([[theta1, theta2], val])
                init_flag = True
        if theta1 % 5 == 0:
            print "done all theta {}".format(theta1)
    out = sorted(thetas, key=lambda tup: tup[1])
    out.reverse()
    return out


def print_exp_state(pos, goal):
    """Given a state of the experiment paramaterised by the position of the
    human, and the position of the goal (as an x, y list), return a string
    representing the state of the experiment"""
    world_string = """"""
    for y in range(15):
        if y % 2 == 0:
            world_string = world_string + 15 * "-"
        else:
            for x in range(15):
                if x % 2 == 0:
                    world_string = world_string + "|"
                else:
                    if (x - 1) / 2 == pos[0] and (y - 1) / 2 == pos[1]:
                        world_string = world_string + "U"
                    elif (x - 1) / 2 == goal[0] and (y - 1) / 2 == goal[1]:
                        world_string = world_string + "G"
                    else:
                        world_string = world_string + " "
        world_string = world_string + "\n"
    return world_string


exp_dict = {"W": 0, "A": 1, "S": 2, "D": 3}


def move_grid(pos, move):
    if move == 0:
        if pos[1] > 0:
            pos[1] -= 1
        else:
            print "Illegal move :/"
            time.sleep(1)
    if move == 1:
        if pos[0] < 6:
            pos[0] += 1
        else:
            print "Illegal move :/"
            time.sleep(1)
    if move == 2:
        if pos[1] < 6:
            pos[1] += 1
        else:
            print "Illegal move :/"
            time.sleep(1)
    if move == 3:
        if pos[0] > 0:
            pos[0] -= 1
        else:
            print "Illegal move :/"
            time.sleep(1)
    reward = -1
    return pos, reward


def run_experiment(dim=7):
    print """Hi there, you have been selected to play HUMAN NAVIGATION, THE GAME
    V 0.1. In a few moments, you will be presented with a grid. The object of
    the game is to navigate your avatar, represented with a U, to the Goal,
    represented with a G. You have four actions, which you can take by pressing
    the WASD keys. However, the direction that you will move when you press the
    key is different for each cell."""
    np.random.seed(2017)
    poss_goals = [[6, 4], [6, 6], [4, 4], [6, 3], [5, 5], [4, 6]]
    pos = [0, 0]
    move_key = np.zeros((dim, dim, 4))
    perms = [0, 1, 2, 3]
    tot_reward = 0
    for index, _ in np.ndenumerate(move_key[:, :, 0].reshape((dim, dim))):
        move_key[index, :] = np.random.permutation(perms)
    for game in range(len(poss_goals) - 1):
        reward = 0
        goal = poss_goals[np.random.randint(len(poss_goals) - 1)]
        while pos != goal:
            print """You have total reward {}, and the state of the world is
            as shown, what's your input?""".format(
                reward
            )
            print print_exp_state(pos, goal)
            move = ""
            while move not in exp_dict.keys():
                move = raw_input("Enter W, A, S or D:")
            move = move_key[tuple(pos + [exp_dict[move]])]
            pos, r = move_grid(pos, move)
            reward += r
            print print_exp_state(pos, goal)
            tot_reward += reward
        pos = [0, 0]

    print """Well done, you completed the game. You got a total reward of {}""".format(
        tot_reward, np.random.uniform(0, 0.0001)
    )


def plot_bar():
    """Plot a bar chart showing the probabilities assigned to the 
    various thetas."""
    ns = np.arange(5)
    # results = [[0.0985, 0.0525, 0.0108, 0.00551, 0.00522],
    #            [0.1029, 0.088, 0.095, 0.07118, 0.1007]]
    # errors = [[0.0162, 0.00985, 0.00360, 0.002446, 0.0],
    #           [0.018, 0.028, 0.022, 0.0189, 0.0246]]

    results = [
        [0.078, 0.0239, 0.0548, 0.047, 0.0779],
        [0.11604, 0.0799, 0.1917, 0.19709, 0.21134],
    ]
    errors = [
        [0.0215, 0.00608, 0.00717, 0.0072, 0.0103],
        [0.0404, 0.0414, 0.034, 0.05312, 0.036],
    ]

    fig, ax = plt.subplots()
    colors = ["0.2", "indianred"]
    le = ["BIRL", "BHIRL"]
    width = 0.4
    for n in range(2):
        print results[n]
        ax.bar(
            ns + width * n,
            results[n],
            width,
            color=colors[n],
            label=le[n],
            yerr=errors[n],
            capsize=10,
        )
    ax.legend(loc="upper right")
    ax.set_ylabel(r"$P(\theta|\mathcal{O}_{1:n}))$")
    ax.set_xlabel(r"$n$")
    ax.set_xticks(ns + 1 * width / 2)
    ax.set_xticklabels(["1", "2", "3", "4", "5"])
    plt.show()


# Some random starting points
inits = [[2, 3, 0, 2], [2, 3, 0, 2], [2, 3, 0, 1], [3, 0, 1, 3], [0, 3, 2, 0]]
O_as = [
    draw_trajectory(0.8, x, gen_option_set([0, 15]), 0.999, [-1, -1]) for x in inits
]
to_store = O_as[:]
# Wrap these into the right format with [action, state]
for i in range(len(O_as)):
    tmp = zip(O_as[i][2], O_as[i][0])
    O_as[i] = tmp

# print O_as[0]
# for i in range(len(O_as)):
#     tmp = zip(find_actions(O_as[i]), O_as[i])
#     O_as[i] = tmp


if __name__ == "__main__":
    import sys

    for n in range(1, 6):
        multipool_MCMC_spec_options(O_as[:n], Os_3, 0.8, 0.999, 3000, 8)
        multipool_MCMC(O_as[:n], 0.8, 0.999, 3000, 8)

    multipool_MCMC_spec_options(O_as[:4], Os_3, 0.8, 0.9, 4000, 4)
    multipool_MCMC(O_as[:4], 0.8, 0.9, 4000, 4)
    options_multipool_MCMC(O_as[:1], 0.8, 0.9, 2000, 1, True)

    for beta in [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2]:
        multipool_MCMC(O_as[:5], beta, 0.9, 3000, 16)

    for beta in [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2]:
        options_multipool_MCMC(O_as[:5], beta, 0.9, 3000, 16)

    if len(sys.argv) > 1:
        multipool_MCMC([O_a, O_a3], sys.argv[2], 0.9, 2000, 16, True)
        options_multipool_MCMC([O_a, O_a3], sys.argv[2], 0.9, 2000, 16, True)
    else:
        cpu_count = multiprocessing.cpu_count()
        print "Number of cores is {}".format(cpu_count)
        #    out = options_multipool_MCMC([O_a, O_a2], 0.8, 0.99, 100, cpu_count)
        for b in [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2]:
            multipool_MCMC(O_as, b, 0.999, 3000, cpu_count, False)

        for b in [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2]:
            options_multipool_MCMC(O_as, b, 0.999, 4000, 8, False)

        for b in [1.2, 1.6, 2]:
            options_multipool_MCMC(O_as, b, 0.999, 4000, 8, False)

        for b in [0.8, 1.2, 1.6, 2]:
            options_multipool_MCMC([O_a, O_a3], b, 0.9, 3000, 2)
            multipool_MCMC([O_a, O_a3], b, 0.9, 3000, 2)

        for b in [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2]:
            multimode_MCMC([O_a, O_a3], b, 0.9, 500, np.random.randint(10), 30, 5)

        plot_theta_margs(
            [
                "./data/new_b_o_0.1",
                "./data/new_b_o_0.3",
                "./data/new_b_o_0.5",
                "./data/new_b_o_0.8",
                "./data/new_b_o_1.2",
                "./data/new_b_o_1.6",
                "./data/new_b_o_2",
            ],
            [
                "./data/beta_mcmc_0.1",
                "./data/beta_mcmc_0.3",
                "./data/beta_mcmc_0.5",
                "./data/beta_mcmc_0.8",
                "./data/beta_mcmc_1.2",
                "./data/beta_mcmc_1.6",
                "./data/beta_mcmc_2",
            ],
            [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2, 1.6],
        )

        plot_theta_margs(
            [
                "./data/beta_o_0.1",
                "./data/beta_o_0.3",
                "./data/beta_o_0.5",
                "./data/beta_o_0.8",
                "./data/beta_o_1.2",
                "./data/beta_o_1.6",
                "./data/beta_o_2",
            ],
            [
                "./data/beta_mcmc_0.1",
                "./data/beta_mcmc_0.3",
                "./data/beta_mcmc_0.5",
                "./data/beta_mcmc_0.8",
                "./data/beta_mcmc_1.2",
                "./data/beta_mcmc_1.6",
                "./data/beta_mcmc_2",
            ],
            [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2, 1.6],
        )

        plot_theta_margs(
            [
                "./data/0.1",
                "./data/0.3",
                "./data/0.5",
                "./data/0.8",
                "./data/1.2",
                "./data/1.6",
                "./data/2",
            ],
            [
                "./data/irl_0.1",
                "./data/irl_0.3",
                "./data/irl_0.5",
                "./data/irl_0.8",
                "./data/irl_1.2",
                "./data/irl_1.6",
                "./data/irl_2",
            ],
            [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2],
        )

        plot_theta_margs(
            [
                "./data/0.1",
                "./data/0.3",
                "./data/0.5",
                "./data/0.8",
                "./data/1.2",
                "./data/1.6",
                "./data/2",
            ],
            [
                "./data/irl_0.1",
                "./data/irl_0.3",
                "./data/irl_0.5",
                "./data/irl_0.8",
                "./data/irl_1.2",
                "./data/irl_1.5",
                "./data/irl_2",
            ],
            [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2, 1.6],
        )

        plot_theta_margs(
            ["./data/0.1", "./data/0.3", "./data/0.5", "./data/0.8"],
            [
                "./data/irl_0.1",
                "./data/irl_0.3",
                "./data/irl_0.5",
                "./data/irl_0.8",
                "./data/irl_1.2",
                "./data/irl_1.6",
                "./data/irl_2",
            ],
            [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2],
        )

        # --------------------------------------------------
        # Plot focussing on the key area
        plot_theta_margs(
            ["./data/0.8", "./data/1.0", "./data/1.2", "./data/1.6", "./data/2"],
            ["./data/irl_0.8", "./data/irl_1.2", "./data/irl_1.6", "./data/irl_2"],
            [0.8, 1.0, 1.2, 1.6, 2],
        )
