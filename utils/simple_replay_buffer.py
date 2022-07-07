from .segment_tree import MinSegmentTree, SumSegmentTree
import numpy as np
import random

def sample_n_unique(sampling_f, n):
    '''
    Helper function. Given a function 'sampling_f'
    that return comparable objects, sample n such unique objects
    '''
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)

    return res

class EpReplayBuffer:
    '''
    A numpy replay buffer for episode's implementation of QMIX
    '''
    def __init__(self, obs_dim, action_dim, ep_limits, ep_size=5000, multi_steps=1, batch_size=32) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ep_limits = ep_limits
        self.ep_size = ep_size
        self.multi_step = multi_steps
        assert self.multi_step == 1, \
            f"now EpReplayBuffer is not available for multi step TD, which now multi_step is {self.multi_step}"
        self.batch_size = batch_size
        self.num_in_buffer = 0
        self.next_idx = 0
        self.obs = np.zeros((self.ep_size, self.ep_limits, self.obs_dim), dtype=np.float32)
        self.action = np.zeros((self.ep_size, self.ep_limits), dtype=np.int64)
        self.reward = np.zeros((self.ep_size, self.ep_limits), dtype=np.float32)
        self.done = np.zeros((self.ep_size, self.ep_limits), dtype=np.float32)
        self.avail_action = np.zeros((self.ep_size, self.ep_limits, self.action_dim), dtype=np.float32)
        self.ep_length = np.zeros((self.ep_size,), dtype=np.int64)
        # self.pad = np.zeros((self.ep_size, self.ep_limits), dtype=np.float32)

    def store(self, ep_dict, ep_len):
        # store 1 episode experience of smac into buffer
        self.obs[self.next_idx] = ep_dict['obs']
        self.action[self.next_idx] = ep_dict['action']
        self.reward[self.next_idx] = ep_dict['reward']
        self.done[self.next_idx][ep_len:] = 1.
        self.avail_action[self.next_idx] = ep_dict['avail_action']
        self.ep_length[self.next_idx] = ep_len + 1
        self.next_idx = (self.next_idx + 1) % self.ep_size
        self.num_in_buffer = min(self.num_in_buffer + 1, self.ep_size)

    def sample(self, idxes=None, max_ep_len=None):
        # sample batch_size episode experience in uniform distribution
        assert self.batch_size <= self.num_in_buffer
        # get experience
        if max_ep_len is None:
            max_ep_len = max(self.ep_length[idxes])
            
        obs_batch = self.obs[idxes][:, :max_ep_len]
        act_batch = self.action[idxes][:, :max_ep_len]
        rew_batch = self.reward[idxes][:, :max_ep_len]
        done_batch = self.done[idxes][:, :max_ep_len]
        avail_act_batch = self.avail_action[idxes][:, :max_ep_len]

        return obs_batch, act_batch, rew_batch, done_batch, avail_act_batch, max_ep_len
        

        
class ReplayBuffer:
    '''
    A numpy replay buffer
    '''
    def __init__(self, obs_dim, action_dim, size=1000000, multi_steps=1, batch_size=32) -> None:
        self.obs_dim = obs_dim
        self.size = size
        self.multi_steps = multi_steps
        self.batch_size = batch_size
        self.num_in_buffer = 0
        self.next_idx = 0
        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.size,), dtype=np.int64)
        self.reward = np.zeros((self.size), dtype=np.float32)
        self.done = np.zeros((self.size), dtype=np.bool_)
        self.avail_action = np.zeros((self.size, action_dim), dtype=np.float32)

    def store(self, obs, action, reward, done, avail_action):
        self.obs[self.next_idx] = obs
        self.action[self.next_idx] = action
        self.reward[self.next_idx] = reward
        self.done[self.next_idx] = done
        self.avail_action[self.next_idx] = avail_action
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.num_in_buffer + 1, self.size)

    def sample(self, idxes=None):
        assert self.batch_size <= self.num_in_buffer
        # random.sample is much more faster than np.random.choice
        if idxes == None:
            idxes = random.sample(range(self.num_in_buffer), self.batch_size)
        # obs_batch = np.concatenate([np.expand_dims(self.obs[idx], axis=0) for idx in idxes], axis=0)
        obs_batch = self.obs[idxes]
        act_batch = self.action[idxes]
        next_obs_batch = self.obs[((np.array(idxes) + self.multi_steps) % self.num_in_buffer).tolist()]
        avail_act_batch = self.avail_action[idxes]
        next_avail_act_batch = self.avail_action[((np.array(idxes) + self.multi_steps) % self.num_in_buffer).tolist()]
        if self.multi_steps > 1:
            # multi step for td-loss
            rew_batch = np.concatenate(
                [np.expand_dims(
                    [self.reward[(idx + idx_off) % self.num_in_buffer] 
                        for idx_off in range(self.multi_steps)], axis=0)
                                for idx in idxes], axis=0)
            done_mask = np.concatenate(
                [self._encode_done(
                        [self.done[(idx + idx_off) % self.num_in_buffer]
                            for idx_off in range(self.multi_steps)])
                                for idx in idxes], axis=0)
        else:
            rew_batch = self.reward[idxes]
            done_mask = np.array([1. if self.done[idx] else 0. for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, avail_act_batch, next_avail_act_batch


    def _encode_done(self, list_done):
        done_mask = np.zeros([self.multi_steps])
        for i in range(self.multi_steps):
            if list_done[i] == True:
                done_mask[i:] = 1.
                return np.expand_dims(done_mask, axis=0)

        return np.expand_dims(done_mask, axis=0)


class PEReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, action_dim, size=1000000, multi_steps=1, alpha=0.6, batch_size=32) -> None:
        super().__init__(obs_dim, action_dim, size, multi_steps, batch_size)
        assert alpha > 0,\
            f'alpha:{alpha} for PER should be an positive decimal that located in (0, 1)'
        # assert multi_steps <= 1,\
        #     f'multi_step:{multi_steps} should be 1 because PER cannot match with multi step TD learning'
        self.alpha = alpha
        self.max_priority = 1.
        self.tree_idx = 0
        # capacity should be positive and a power of 2
        tree_capacity = 1
        while tree_capacity < self.size:
            tree_capacity *= 2
        # Tree for PER
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, obs, action, reward, done, avail_action):
        super().store(obs, action, reward, done, avail_action)
        # initialize priority for new transition
        self.sum_tree[self.tree_idx] = self.max_priority ** self.alpha
        self.min_tree[self.tree_idx] = self.max_priority ** self.alpha
        self.tree_idx = (self.tree_idx + 1) % self.size
    
    def sample(self, beta=0.4, idxes=None):
        assert self.batch_size <= self.num_in_buffer
        assert beta > 0
        if idxes == None:
            idxes = self._sample_proportional()
        obs_batch = self.obs[idxes]
        act_batch = self.action[idxes]
        next_obs_batch = self.obs[((np.array(idxes) + self.multi_steps) % self.num_in_buffer).tolist()]
        avail_act_batch = self.avail_action[idxes]
        next_avail_act_batch = self.avail_action[((np.array(idxes) + self.multi_steps) % self.num_in_buffer).tolist()]
        if self.multi_steps > 1:
            # multi step for td-loss
            reward_batch = np.concatenate(
                [np.expand_dims(
                    [self.reward[(idx + idx_off) % self.num_in_buffer] 
                        for idx_off in range(self.multi_steps)], axis=0)
                                for idx in idxes], axis=0)
            done_mask = np.concatenate(
                [self._encode_done(
                        [self.done[(idx + idx_off) % self.num_in_buffer]
                            for idx_off in range(self.multi_steps)])
                                for idx in idxes], axis=0)
        else:
            reward_batch = self.reward[idxes]
            done_mask = np.array([1. if self.done[idx] else 0. for idx in idxes], dtype=np.float32)
        weight = np.array([self._calculate_weight(idx, beta) for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, reward_batch, next_obs_batch, done_mask, avail_act_batch, next_avail_act_batch, weight, idxes

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.num_in_buffer
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        # sample based in proportional variant
        indices = []
        p_total = self.sum_tree.sum(0, self.num_in_buffer-1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    
    def _calculate_weight(self, idx, beta):
        # get max_weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.num_in_buffer) ** (-beta)

        # calculate weight
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.num_in_buffer) ** (-beta)
        weight = weight / max_weight

        return weight