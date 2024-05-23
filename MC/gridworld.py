import io
import sys
from copy import deepcopy as dc

import numpy as np



UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

from gym import Env, spaces
from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):
    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """

    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob": p})



class GridworldEnv(DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4, 4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a: [] for a in range(nA)}

            def is_done(s):
                return s == 0 or s == (nS - 1)

            reward = 0.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        # Prepare state transition tensor and reward tensor
        self.P_tensor = np.zeros(shape=(nA, nS, nS))
        self.R_tensor = np.zeros(shape=(nS, nA))

        for s in self.P.keys():
            for a in self.P[s].keys():
                p_sa, s_prime, r, done = self.P[s][a][0]
                self.P_tensor[a, s, s_prime] = p_sa
                self.R_tensor[s, a] = r

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def observe(self):
        return dc(self.s)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        outfile.write('==' * self.shape[1] + '==\n')

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

        outfile.write('==' * self.shape[1] + '==\n')
        
        

import matplotlib.pyplot as plt

threshold = -1.0


def visualize_value_function(ax,  # matplotlib axes object
                             v_pi: np.array,
                             nx: int,
                             ny: int,
                             plot_cbar=True):
    hmap = ax.imshow(v_pi.reshape(nx, ny),
                     interpolation='nearest')
    if plot_cbar:
        cbar = ax.figure.colorbar(hmap, ax=ax)

    # disable x,y ticks for better visibility
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

    # annotate value of value functions on heat map
    for i in range(ny):
        for j in range(nx):
            cell_v = v_pi.reshape(nx, ny)[j, i]
            text_color = "w" if cell_v < threshold else "black"
            cell_v = "{:.2f}".format(cell_v)
            ax.text(i, j, cell_v, ha="center", va="center", color="w")


def visualize_policy(ax, pi: np.array, nx: int, ny: int):
    d_symbols = ['↑', '→', '↓', '←']
    pi = np.array(list(map(np.argmax, pi))).reshape(nx, ny)

    ax.imshow(pi, interpolation='nearest', cmap=plt.get_cmap('Paired'))

    ax.set_xticks(np.arange(pi.shape[1]))
    ax.set_yticks(np.arange(pi.shape[0]))
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(pi.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(pi.shape[0] + 1) - .5, minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(ny):
        for j in range(nx):
            direction = pi[j, i]
            direction = d_symbols[direction]
            ax.text(i, j, direction, ha="center", va="center", color="black", fontsize=20)
            
            
            



