import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from utils import generate_problem, visualize_value_function, simulate_MDP


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])
    Ts = tf.convert_to_tensor(Ts)
    terminal_idx = tf.argmax(terminal_mask)

    assert terminal_mask.ndim == 1 and reward.ndim == 2

    # perform value iteration
    for _ in range(1000):
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        # Ts is a 4 element python list of transition matrices for 4 actions

        # reward has shape [sdim, 4] - represents the reward for each state
        # action pair

        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition

        V_prev = V
        V_ind = tf.transpose(tf.linalg.matvec(Ts, V_prev)).numpy()
        V_ind[terminal_idx, :] = 0
        V_ind = tf.convert_to_tensor(V_ind)
        V_big = reward + gam * V_ind
        V = tf.reduce_max(V_big, axis=1)
        err = tf.linalg.norm(V - V_prev)

        ######### Your code ends here ###########

        if err < 1e-7:
            break

    return V


# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    plt.figure(213)
    opt_policy = visualize_value_function(np.array(V_opt).reshape((n, n)))
    opt_policy = np.flip(opt_policy.T, axis=0)
    plt.colorbar()
    plt.title("value iteration")
    plt.show()

    plt.figure(213)
    Ts = problem["Ts"]
    simulate_MDP(np.array(V_opt).reshape((n, n)), Ts)
    plt.colorbar()
    plt.title("MDP simulation - value iteration")
    plt.show()

    return opt_policy


if __name__ == "__main__":
    opt_policy_V = main()
