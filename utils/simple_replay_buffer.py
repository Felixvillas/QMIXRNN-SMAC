import numpy as np
import random

class EpReplayBuffers:
    '''
    A numpy replay buffer for episode's implementation of QMIX
    '''
    def __init__(self, obs_dim, num_agents, action_dim, ep_limits, ep_size=5000, multi_steps=1, batch_size=32) -> None:
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.ep_limits = ep_limits
        self.ep_size = ep_size
        self.multi_step = multi_steps
        self.batch_size = batch_size
        self.obs = np.zeros((self.ep_size, self.ep_limits, self.num_agents, self.obs_dim), dtype=np.float32)
        self.action = np.zeros((self.ep_size, self.ep_limits, self.num_agents), dtype=np.int64)
        self.avail_action = np.zeros((self.ep_size, self.ep_limits, self.num_agents, self.action_dim), dtype=np.float32)

    def store(self, ep_dict, idx):
        # store 1 episode experience of smac into buffer
        self.obs[idx] = ep_dict['obs']
        self.action[idx] = ep_dict['action']
        self.avail_action[idx] = ep_dict['avail_action']

    def sample(self, idxes, max_ep_len):
        # get experience   
        obs_batch = self.obs[idxes][:, :max_ep_len]
        act_batch = self.action[idxes][:, :max_ep_len]
        avail_act_batch = self.avail_action[idxes][:, :max_ep_len]

        return obs_batch, act_batch, avail_act_batch

class TotalEpReplayBuffer(EpReplayBuffers):
    def __init__(self, obs_dim, action_dim, ep_limits, ep_size=5000, multi_steps=1, batch_size=32) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ep_limits = ep_limits
        self.ep_size = ep_size
        self.multi_step = multi_steps
        self.batch_size = batch_size
        self.obs = np.zeros((self.ep_size, self.ep_limits, self.obs_dim), dtype=np.float32)
        self.reward = np.zeros((self.ep_size, self.ep_limits), dtype=np.float32)
        self.done = np.zeros((self.ep_size, self.ep_limits), dtype=np.float32)
        self.ep_length = np.zeros((self.ep_size,), dtype=np.int64)
    
    def store(self, ep_dict, ep_len, idx):
        # store 1 episode experience of smac into buffer
        self.obs[idx] = ep_dict['obs']
        self.reward[idx] = ep_dict['reward']
        self.done[idx][ep_len:] = 1.
        self.ep_length[idx] = ep_len + 1

    def sample(self, idxes):
        # sample batch_size episode experience in uniform distribution
        max_ep_len = max(self.ep_length[idxes])
        # get experience
        obs_batch = self.obs[idxes][:, :max_ep_len]
        rew_batch = self.reward[idxes][:, :max_ep_len]
        done_batch = self.done[idxes][:, :max_ep_len]

        return obs_batch, rew_batch, done_batch, max_ep_len

class ReplayBuffer:
    '''
    A numpy replay buffer for episode's implementation of QMIX
    '''
    def __init__(self, obs_dim, state_dim, num_agents, action_dim, ep_limits, ep_size=5000, multi_steps=1, batch_size=32) -> None:
        self.ep_size = ep_size
        self.multi_step = multi_steps
        self.batch_size = batch_size
        self.num_in_buffer = 0
        self.next_idx = 0
        assert self.multi_step == 1, \
            f"now EpReplayBuffer is not available for multi step TD, which now multi_step is {self.multi_step}"

        self.buffers = EpReplayBuffers(
            obs_dim=obs_dim, num_agents=num_agents, action_dim=action_dim,
            ep_limits=ep_limits, ep_size=ep_size, multi_steps=multi_steps,
            batch_size=batch_size
        )
        self.total_buffer = TotalEpReplayBuffer(
            obs_dim=state_dim, action_dim=action_dim, ep_limits=ep_limits, 
            ep_size=ep_size, multi_steps=multi_steps, batch_size=batch_size
        )
    
    def store(self, ep_dict, total_ep_dict, ep_len):
        self.buffers.store(ep_dict=ep_dict, idx=self.next_idx)
        self.total_buffer.store(ep_dict=total_ep_dict, ep_len=ep_len, idx=self.next_idx)

        self.next_idx = (self.next_idx + 1) % self.ep_size
        self.num_in_buffer = min(self.num_in_buffer + 1, self.ep_size)
    
    def next_timestep(self, current_timestep_np):
        next_timestep_np = np.zeros_like(current_timestep_np)
        next_timestep_np[:, :-1] = current_timestep_np[:, 1:]
        return next_timestep_np

    def sample(self):
        idxes = random.sample(range(self.num_in_buffer), self.batch_size)
        total_obs_batch, total_rew_batch, total_done_batch, max_ep_len = self.total_buffer.sample(idxes)
        obs_batchs, act_batchs, avail_act_batchs = self.buffers.sample(idxes, max_ep_len)
        next_obs_batchs = self.next_timestep(obs_batchs)
        next_avail_act_batchs = self.next_timestep(avail_act_batchs)
        next_total_obs_batch = self.next_timestep(total_obs_batch)
        return obs_batchs, act_batchs, avail_act_batchs, \
                    total_obs_batch, total_rew_batch, total_done_batch, \
                        next_obs_batchs, next_avail_act_batchs, next_total_obs_batch
