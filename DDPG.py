


import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.animation as animation
# from IPython.display import HTML
# from IPython.display import clear_output

env = gym.make('CarRacing-v2', continuous=False)
print("Observation space: ", env.observation_space)
print("Action space: ", env.action_space)

"""## Check the Video"""

env.reset()
frames = []
# 50 frame은 도입부라서 학습에 사용되지 않음 > PASS
for i in range(50):
    s, r, terminated, truncated, info = env.step(0)  # 0-th action is no_op action
    frames.append(s)

# Create animation
fig = plt.figure(figsize=(5, 5))
plt.axis('off')
im = plt.imshow(frames[0])
def animate(i):
    im.set_array(frames[i])
    return im,
anim = animation.FuncAnimation(fig, animate, frames=len(frames))


"""# Preprocess the Image"""

# Every frme always contains a black area at the bottom of the frame, so we had better cut this black area.
# Also, Color imformation is not directly related to car racing. So we will use gray image for computation efficiency.
# 학습에 불필요한 부분은 CROP 후에 사용함 (Grayscale로 변환)
def preprocess(img):
    img = img[:84, 6:90] # CarRacing-v2-specific cropping

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

"""# Manual Environment"""

class ImageEnv(gym.Wrapper):

# skip_frame: 한번 action을 수행하면 4frame동안 진행함
# stack_frame: 4프레임을 모아서 한번의 입력으로 제공함
# initial_no_op: 최초 skip 프레임 수

    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

    # 처음으로 돌아감

    def reset(self):
        # Reset the original environment.
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                s, info = self.env.reset()

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # The initial observation is simply a copy of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info

    def step(self, action):
        # We take an action for self.skip_frames steps
        # terminated: 완료 / truncated: 실패
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info

"""
0: do nothing
1: steer left
2: steer right
3: steer gas
4: brake
"""




import numpy as np
import torch
import torch.nn as nn


# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, state, new_state, action, reward, done_bool):
		data = (state, new_state, action, reward, done_bool)
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind:
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class NaivePrioritizedBuffer(object):
	def __init__(self, capacity, prob_alpha=0.6):
		self.prob_alpha = prob_alpha
		self.capacity = capacity
		self.buffer = []
		self.pos = 0
		self.priorities = np.zeros((capacity,), dtype=np.float32)

	def add(self, state, next_state, action, reward, done):
		state = state.numpy()
		next_state = next_state.numpy()
		assert state.ndim == next_state.ndim
		state = np.expand_dims(state, 0)
		next_state = np.expand_dims(next_state, 0)

		max_prio = self.priorities.max() if self.buffer else 1.0

		if len(self.buffer) < self.capacity:
			self.buffer.append((state, next_state, action, reward, done))
		else:
			self.buffer[self.pos] = (state, next_state, action, reward, done)

		self.priorities[self.pos] = max_prio
		self.pos = (self.pos + 1) % self.capacity

	def sample(self, batch_size, beta=0.4):
		if len(self.buffer) == self.capacity:
			prios = self.priorities
		else:
			prios = self.priorities[:self.pos]

		probs = prios ** self.prob_alpha
		probs /= probs.sum()

		indices = np.random.choice(len(self.buffer), batch_size, p=probs)
		samples = [self.buffer[idx] for idx in indices]

		total = len(self.buffer)
		weights = (total * probs[indices]) ** (-beta)
		weights /= weights.max()
		weights = np.array(weights, dtype=np.float32)

		batch = list(zip(*samples))
		states = np.concatenate(batch[0])
		actions = batch[2]
		rewards = batch[3]
		next_states = np.concatenate(batch[1])
		dones = batch[4]

		return np.array(states), np.array(next_states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1), indices, weights

	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in list(zip(batch_indices, batch_priorities)):
			self.priorities[idx] = prio

	def __len__(self):
		return len(self.buffer)

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


feat_size = 1
latent_dim = 512

''' Utilities '''


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Actor(nn.Module):
    def __init__(self, action_dim, img_stack):
        super(Actor, self).__init__()

        self.encoder = torch.nn.Sequential(  ## input size:[96, 96]
            torch.nn.Conv2d(img_stack, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),  ## output size: [128, 3, 3]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            Flatten(),  ## output: 512
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, action_dim),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.linear(x)

        return x

class Critic(nn.Module):
    def __init__(self, action_dim, img_stack):
        super(Critic, self).__init__()

        self.encoder = torch.nn.Sequential(  ## input size:[96, 96]
            torch.nn.Conv2d(img_stack, 16, 5, 2, padding=2),  ## output size: [16, 48, 48]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 5, 2, padding=2),  ## output size: [32, 24, 24]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 5, 2, padding=2),  ## output size: [64, 12, 12]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 5, 4, padding=2),  ## output size: [128, 3, 3]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),  ## output size: [256, 2, 2]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, 5, 2, padding=2),  ## output size: [512, 1, 1]
            Flatten(),  ## output: 512
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + action_dim, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
        )

    def forward(self, x, u):

        x = self.encoder(x)
        x = torch.cat([x, u], 1)
        x = self.linear(x)

        return x



class DDPG(object):
	def __init__(self, action_dim, img_stack):
		self.action_dim = action_dim
		self.actor = Actor(action_dim, img_stack).to(device)
		self.actor_target = Actor(action_dim, img_stack).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
		self.actor_loss = []

		self.critic = Critic(action_dim, img_stack).to(device)
		self.critic_target = Critic(action_dim, img_stack).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
		self.critic_loss = []


	def select_action(self, state):
		state = state.float().to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, beta_PER, batch_size=100, discount=0.99, tau=0.005):

		for it in range(iterations):

			# print("training")

			# Sample replay buffer
			x, y, u, r, d, indices, w = replay_buffer.sample(batch_size, beta=beta_PER)
			state = torch.FloatTensor(x).squeeze(1).to(device)
			#             print('state size: ' +str(state.size()))
			u = u.reshape((batch_size, self.action_dim))
			action = torch.FloatTensor(u).to(device)
			#             print('action size: ' +str(action.size()))
			next_state = torch.FloatTensor(y).squeeze(1).to(device)
			#             print('next state size: ' +str(next_state.size()))
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)
			w = w.reshape((batch_size, -1))
			weights = torch.FloatTensor(w).to(device)

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = weights * ((current_Q - target_Q).pow(2))
			prios = critic_loss + 1e-5
			critic_loss = critic_loss.mean()
			self.critic_loss.append(critic_loss)
			# print("critic_loss"+str(critic_loss))

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()
			self.actor_loss.append(actor_loss)
			# print("actor_loss"+ str(actor_loss))


			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

	def save(self, directory, name):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
		torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))

		torch.save(self.critic.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
		torch.save(self.critic_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))

	def load(self, directory, name):
		self.actor.load_state_dict(
			torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
		self.actor_target.load_state_dict(
			torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

		self.critic.load_state_dict(
			torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
		self.critic_target.load_state_dict(
			torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

	def load_actor(self, directory, name):
		self.actor.load_state_dict(
			torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
		self.actor_target.load_state_dict(
			torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

import torch
import gym
import numpy as np
import os
import gym
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import argparse

from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, env_name, random_seed, img_stack, action_repeat):
        self.env = gym.make(env_name)
        self.env.seed(random_seed)
        self.action_space = self.env.action_space
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = img_stack
        self.action_repeat = action_repeat

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        #         print(img_rgb)
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [np.expand_dims(img_gray, axis=0)] * self.img_stack  # four frames for decision
        return torch.FloatTensor(self.stack).permute(1, 0, 2, 3)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(np.expand_dims(img_gray, axis=0))
        assert len(self.stack) == self.img_stack
        return torch.FloatTensor(self.stack).permute(1, 0, 2, 3), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


def train(env):
    ######### Hyperparameters #########
    env_name = env
    log_interval = 10  # print avg reward after interval
    random_seed = 0
    gamma = 0.99  # discount for future rewards
    batch_size = 100  # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.5
    polyak = 0.995  # target policy update parameter (1-tau)
    policy_noise = 0.2  # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2  # delayed policy updates parameter
    max_episodes = int(1e8)  # max num of episodes
    max_timesteps = 500  # max timesteps in one episode
    save_every = 100  # model saving interal
    img_stack = 4  # number of image stacks together
    action_repeat = 8  # repeat action in N frames
    max_size = 1e6
    vis = True

    """ parameters for epsilon declay """
    epsilon_start = 1
    epsilon_final = 0.01
    decay_rate = max_episodes / 50

    """ beta Prioritized Experience Replay"""
    beta_start = 0.4
    beta_frames = 25000

    # if not os.path.exists('./TD3tested'):
    #     os.mkdir('./TD3tested')
    directory = "./{}".format(env_name)  # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)

    ###################################

    env = Env(env_name, random_seed, img_stack, action_repeat)
    # print("env")
    action_dim = env.action_space.shape[0]
    # if vis:
    #     draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")
    #if args.policy == 'TD3':
        # policy = TD3(action_dim, img_stack)
    # if args.policy == 'DDPG':
    policy = DDPG(action_dim, img_stack)
    replay_buffer = NaivePrioritizedBuffer(int(max_size))

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)

    # logging variables:

    log_f = open("log.txt", "w+")
    ## for plot
    Reward = []
    total_timesteps = 0
    episode_timesteps = 0
    running_score = 0

    # training procedure:
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        # print("here")
        episode_timesteps = 0
        score = 0

        for t in range(max_timesteps):
            # select action and add exploration noise:
            #             print("state: " + str(state))
            action = policy.select_action(state)
            # print("action: " + str(action))
            exploration_noise = (epsilon_start - epsilon_final) * math.exp(-1. * total_timesteps / decay_rate)
            action = action + np.random.normal(0, exploration_noise, size=action_dim)
            action = action.clip(env.action_space.low, env.action_space.high)
            #             print("action clipped: " + str(action))

            # take action in env:
            next_state, reward, done, die = env.step( action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]) )
            #             print("state: " +str(next_state))
            env.render()
            replay_buffer.add(state, next_state, action, reward, float(done))
            state = next_state

            score += reward
            total_timesteps += 1
            episode_timesteps += 1

            # if episode is done then update policy:
            if done or t == (max_timesteps - 1):
                beta = min(1.0, beta_start + total_timesteps * (1.0 - beta_start) / beta_frames)
                policy.train(replay_buffer, episode_timesteps, beta)
                break

        wandb.log({'Episode Reward': score})
        running_score = running_score * 0.99 + score * 0.01



        if episode % log_interval == 0:
            # if vis:
            #     draw_reward(xdata = episode, ydata = running_score)
            log_f.write('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}\n'.format(episode, score, running_score))
            log_f.flush()
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(episode, score, running_score))


        # if avg reward > 300 then save and stop traning:
        if running_score >= 900:
            #         if episode % save_every == 0:
            print("########## Model received ###########")
            name = filename
            policy.save(directory, name)
            log_f.close()
            break

        if episode % 100 == 0:
            if not os.path.exists(directory):
                os.mkdir(directory)
            policy.save(directory, filename)


wandb.init(project='RL_A2C')
wandb.run.name = 'RL_A2C_DDPG'
wandb.run.save()

train('CarRacing-v2')
