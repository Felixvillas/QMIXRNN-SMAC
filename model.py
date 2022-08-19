import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from utils.simple_replay_buffer import EpReplayBuffer
import random

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
USE_CUDA = torch.cuda.is_available()
if USE_CUDA: 
    device = torch.device('cuda:0') 
    # torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class QMIX(nn.Module):
    def __init__(self, obs_size=16, state_size=32, num_agents=2, num_actions=5) -> None:
        super(QMIX, self).__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.net_embed_dim = 64

        self.fc1 = nn.Linear(obs_size, self.net_embed_dim)
        self.rnn = nn.GRUCell(self.net_embed_dim, self.net_embed_dim)
        self.fc2 = nn.Linear(self.net_embed_dim, num_actions)

        self.mix_embed_dim = 32
        self.hyper_embed_dim = 64
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_size, self.hyper_embed_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_embed_dim, num_agents * self.mix_embed_dim)
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_size, self.mix_embed_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_size, self.hyper_embed_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_embed_dim, self.mix_embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_size, self.mix_embed_dim),
            nn.ReLU(),
            nn.Linear(self.mix_embed_dim, 1)
        )

    def orth_init(self):
        # orthogonal initialization
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

    
    def get_value(self, obs):
        '''get q_value accroding to current obs, avail_actions'''
        # input : obs:     (num_agents, obs_size), avail_actions:(num_agents, num_actions)
        # output: q_value: (num_agents, num_actions)
        q_1 = F.relu(self.fc1(obs))
        h = self.rnn(q_1, self.eval_rnn_hidden)
        q_value = self.fc2(h)
        self.eval_rnn_hidden = h
        return q_value

    def get_batch_value(self, obs):
        '''get batch q_value accroding to current obs, avail_actions'''
        # input : obs:          (batch_size, episode_limits, num_agents, obs_size), 
        #         avail_actions:(batch_size, episode_limits, num_agents, num_actions)
        # output: q_value:      (batch_size, episode_limits, num_agents, num_actions)
        [batch_size, timesteps] = obs.shape[:2]  # batch_size here is actually episode_number
        self.init_train_rnn_hidden(episode_num=batch_size)
        q_1 = F.relu(self.fc1(obs))
        rnn_value = []
        for t in range(timesteps):
            batch_value = []
            for a_id in range(self.num_agents):
                batch_value.append(
                    self.rnn(q_1[:, t, a_id, :], self.train_rnn_hidden[:, a_id, :])
                )
            self.train_rnn_hidden = torch.stack(batch_value, dim=1)
            rnn_value.append(self.train_rnn_hidden)
        rnn_value = torch.stack(rnn_value, dim=1)
        q_value = self.fc2(rnn_value)
        return q_value
        
    def get_batch_total(self, max_q_value, state):
        '''get batch q_total according to all q_value and state'''
        # input: max_q_value: (batch_size, episode_limits, num_agents)
        #        state      : (batch_size, episode_limits, state_size)
        # output:q_total    : (batch_size, episode_limits)
        w1, b1, w2, b2 = self.get_mix_weight(state)
        # First layer
        q_total_1 = F.elu(torch.matmul(max_q_value.unsqueeze(-2), w1) + b1)
        # Second layer
        q_total_2 = torch.matmul(q_total_1, w2) + b2

        q_total = q_total_2.squeeze()
        return q_total


    def forward(self, obs, avail_actions=None, is_train=True):
        raise NotImplementedError

    def get_mix_weight(self, state):
        # q_total weight
        w1 = self.hyper_w1(state)
        b1 = self.hyper_b1(state)
        w2 = self.hyper_w2(state)
        b2 = self.hyper_b2(state)
        return F.softmax(w1.view(state.shape[0], state.shape[1], self.num_agents, self.mix_embed_dim), dim=-2), \
               b1.unsqueeze(-2), \
               F.softmax(w2.unsqueeze(-1), dim=-2), b2.unsqueeze(-1)

    def init_train_rnn_hidden(self, episode_num):
        # init a gru_hidden for every agent of every episode during training
        self.train_rnn_hidden = torch.zeros((episode_num, self.num_agents, self.net_embed_dim)).to(device)

    def init_eval_rnn_hidden(self):
        # init a gru_hidden for every agent of every episode during evaluating
        self.eval_rnn_hidden = torch.zeros((self.num_agents, self.net_embed_dim)).to(device)


class QMIX_agent(nn.Module):
    def __init__(
            self,
            obs_size=16, 
            state_size=32, 
            num_agents=2, 
            num_actions=5, 
            is_share_para=False, 
            is_ddqn=True, 
            multi_steps=8,
            is_per=False, 
            alpha=0.6, 
            beta=0.2, 
            prior_eps=1e-6, 
            gamma=0.99, 
            replay_buffer_size=5000, 
            episode_limits=60,
            batch_size=32, 
            optimizer=torch.optim.RMSprop, 
            learning_rate=3e-4,
            grad_norm_clip=10,
        ) -> None:
        super(QMIX_agent, self).__init__()
        assert multi_steps == 1 and is_per == False and is_share_para == True, \
            f"Now QMIX with rnn is not compatible with multi_steps and per, \
                as well as only compatible with share net para"
        self.obs_size = obs_size
        self.state_size = state_size
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.is_ddqn = is_ddqn
        self.multi_steps = multi_steps
        self.is_per = is_per
        self.alpha = alpha
        self.beta = beta
        self.prior_eps = prior_eps
        self.gamma = gamma
        self.episode_limits = episode_limits
        self.batch_size = batch_size
        
        # Construct Q_net and target_Q_net
        self.Q = QMIX(obs_size, state_size, num_agents, num_actions).to(device)
        self.target_Q = QMIX(obs_size, state_size, num_agents, num_actions).to(device)
        self.Q.orth_init() # orthogonal initialization
        self.target_Q.load_state_dict(self.Q.state_dict())
        
        self.params = list(self.Q.parameters())
        self.grad_norm_clip = grad_norm_clip
        # RMSProp alpha:0.99, RMSProp epsilon:0.00001
        self.optimizer = optimizer(self.params, learning_rate, alpha=0.99, eps=1e-5)
        self.MseLoss = nn.MSELoss(reduction='sum')

        # Consturct buffer
        self.replay_buffer = [
            EpReplayBuffer(obs_size, num_actions, episode_limits, replay_buffer_size, multi_steps, batch_size) for _ in range(num_agents)
        ]
        self.replay_buffer.append(
            EpReplayBuffer(state_size, num_actions, episode_limits, replay_buffer_size, multi_steps, batch_size)
        )

    def select_actions(self, obs, avail_actions, random_selection):
        '''greedily select actions according to current obs'''
        # input : obs:(num_agents, obs_shape), avail_actions:(num_agents, num_actions), random_selection:(num_agents,)
        # output: actions: a list that length is num_agents
        obs = torch.from_numpy(obs).type(torch.FloatTensor).to(device)
        avail_actions = torch.BoolTensor(avail_actions).to(device)
        q_values = self.Q.get_value(obs)
        q_values[avail_actions == 0.0] = -float('inf')
        max_actions = q_values.max(-1)[1].cpu()
        random_actions = Categorical(avail_actions).sample().cpu()
        actions = [random_actions[i].item() if random_selection[i] else max_actions[i].item() for i in range(self.num_agents)]
        return actions

    def update(self):
        '''update Q: 1 step of gradient descent'''
        idxes = random.sample(range(self.replay_buffer[-1].num_in_buffer), self.batch_size)
        obs_batch_total, _, rew_batch_total, done_total, _, max_ep_length = \
            self.replay_buffer[-1].sample(idxes)
        # every agent's experience
        obs_batch = np.zeros((self.batch_size, max_ep_length, self.num_agents, self.obs_size))
        act_batch = np.zeros((self.batch_size, max_ep_length, self.num_agents))
        next_obs_batch = np.zeros((self.batch_size, max_ep_length, self.num_agents, self.obs_size))
        avail_act_batch = np.zeros((self.batch_size, max_ep_length, self.num_agents, self.num_actions))
        next_avail_act_batch = np.zeros((self.batch_size, max_ep_length, self.num_agents, self.num_actions))
        for i in range(self.num_agents):
            obs_batch_i, act_batch_i, _, _, avail_act_batch_i, _ = \
                self.replay_buffer[i].sample(idxes, max_ep_length)
            obs_batch[:, :, i, :] = obs_batch_i
            act_batch[:, :, i] = act_batch_i
            next_obs_batch[:, :-1, i, :] = obs_batch_i[:, 1:, :]  # the last trajectory's next obs is None
            avail_act_batch[:, :, i, :] = avail_act_batch_i
            next_avail_act_batch[:, :-1, i, :] = avail_act_batch_i[:, 1:, :] # the last trajectory's next avail act is None
        
        # Convert numpy nd_array to torch tensor for calculation
        # every agent's experience
        obs_batch = torch.from_numpy(obs_batch).type(torch.FloatTensor).to(device)
        act_batch = torch.from_numpy(act_batch).type(torch.int64).to(device)
        next_obs_batch = torch.from_numpy(next_obs_batch).type(torch.FloatTensor).to(device)
        avail_act_batch = torch.from_numpy(avail_act_batch).type(torch.BoolTensor).to(device)
        next_avail_act_batch = torch.from_numpy(next_avail_act_batch).type(torch.BoolTensor).to(device)
        
        # Q_total's experience
        obs_batch_total = torch.from_numpy(obs_batch_total).type(torch.FloatTensor).to(device)
        rew_batch_total = torch.from_numpy(rew_batch_total).type(torch.FloatTensor).to(device)
        next_obs_batch_total = torch.zeros_like(obs_batch_total).type(torch.FloatTensor).to(device)
        next_obs_batch_total[:, :-1, :] = obs_batch_total[:, 1:, :]
        not_done_total = torch.from_numpy(1 - done_total).type(torch.FloatTensor).to(device)

        # We choose Q based on action taken.
        all_current_Q_values = self.Q.get_batch_value(obs_batch)
        current_Q_values = all_current_Q_values.gather(-1, act_batch.unsqueeze(-1)).squeeze(-1)
        total_current_Q_values = self.Q.get_batch_total(current_Q_values, obs_batch_total)
        # mask valueless current Q values: In every episode, the first step is always have value 
        total_current_Q_values *= torch.cat((torch.ones(done_total.shape[0], 1).to(device), not_done_total[:, :-1]), dim=1)
        
        # compute target
        target_Q_output = self.target_Q.get_batch_value(next_obs_batch)
        # Mask out unavailable actions: refer to pymarl
        target_Q_output[next_avail_act_batch == 0.0] = -9999999
        if self.is_ddqn:
            # target_current_Q_values: get target values from current values
            target_current_Q_values = torch.zeros_like(target_Q_output).to(device)
            target_current_Q_values[:, :-1] = all_current_Q_values.clone().detach()[:, 1:]
            # target_current_Q_values = self.Q.get_batch_value(next_obs_batch).detach()
            target_current_Q_values[next_avail_act_batch == 0.0] = -9999999
            target_act_batch = target_current_Q_values.max(-1)[1]
            target_Q_values = target_Q_output.gather(-1, target_act_batch.unsqueeze(-1)).squeeze(-1)
        else:
            target_Q_values = target_Q_output.max(-1)[0]

        total_target_Q_values = self.target_Q.get_batch_total(target_Q_values, next_obs_batch_total)
        # mask valueless target Q values and compute the target of the current Q values
        total_target_Q_values = rew_batch_total + self.gamma * not_done_total * total_target_Q_values
        
        # take gradient step
        # compute loss: Detach variable from the current graph since we don't want gradients for next Q to propagated
        loss = self.MseLoss(total_current_Q_values, total_target_Q_values.detach())
        # Clear previous gradients before backward pass
        self.optimizer.zero_grad()
        # run backward pass
        loss = loss / not_done_total.sum()
        loss.backward()
        # grad_norm_clip: Reduce magnitude of gradients above this L2 norm
        nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
        # Perform the update
        self.optimizer.step()

        return loss.item()

    def increase_bate(self, current_t, max_training_steps):
        '''annealing increase bate when use per'''
        if self.is_per:
            fraction = min((current_t+1) / max_training_steps, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)
        else:
            return

    def update_targets(self):
        '''load para from Q to target_Q'''
        self.target_Q.load_state_dict(self.Q.state_dict())

    def evaluate(self, env, episode_num=32):
        '''evaluate Q model'''
        eval_data = []
        for _ in range(episode_num):
            eval_ep_rewards = []
            done = False
            env.reset()
            self.Q.init_eval_rnn_hidden()
            while not done:
                last_obs = env.get_obs()
                avail_actions = env.get_avail_actions()
                recent_observations = np.concatenate([np.expand_dims(ob, axis=0) for ob in last_obs], axis=0)
                random_selection = np.zeros(self.num_agents).astype(np.bool_)
                action = self.select_actions(recent_observations, avail_actions, random_selection)
                reward, done, info = env.step(action)
                eval_ep_rewards.append(reward)

                if done:
                    eval_data.append(
                        [sum(eval_ep_rewards), len(eval_ep_rewards), 1. if 'battle_won' in info and info['battle_won'] else 0.]
                    )

        start = episode_num // 4
        end = episode_num * 3 // 4
        sort_eval_data = sorted(eval_data, key=lambda x: x[-1])[start: end] # sorted by win or nor
        return np.mean(sort_eval_data, axis=0)
