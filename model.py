import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from utils.simple_replay_buffer import ReplayBuffer

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
        self.obs_size = obs_size
        self.state_size = state_size
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
        batch_size, timesteps, num_agents, obs_dim = obs.shape  # batch_size here is actually episode_number
        self.init_train_rnn_hidden(episode_num=batch_size)
        _, _, hidden_dim = self.train_rnn_hidden.shape

        q_value = []
        for t in range(timesteps):
            '''
              note: (batch_size, num_agents, dim) --> (batch_size*num_agents, dim) [###By tensor.reshape]
              As there is no temporal relationship among agents and nn.GRUCell can only accept 2-D data as inputs, 
              so we can concatenate batch_size experiences of different agents for faster cuda parallel computing
            '''
            q_1 = F.relu(self.fc1(obs[:, t].reshape(-1, obs_dim)))
            rnn_value = self.rnn(q_1, self.train_rnn_hidden.reshape(-1, hidden_dim))
            q_2 = self.fc2(rnn_value)
            self.train_rnn_hidden = rnn_value.reshape(batch_size, num_agents, hidden_dim)
            q_value.append(q_2.reshape(batch_size, num_agents, -1))
        q_value = torch.stack(q_value, dim=1)
        return q_value
        
    def get_batch_total(self, max_q_value, state):
        '''get batch q_total according to all q_value and state'''
        # input: max_q_value: (batch_size, episode_limits, num_agents)
        #        state      : (batch_size, episode_limits, state_size)
        # output:q_total    : (batch_size, episode_limits)
        '''
          note: (batch_size, episode_limits, dim) --> (batch_size*episode_limits, dim) [###By tensor.reshape]
          like get_batch_value, for faster(actually uncertain) cuda parallel computing
        '''
        batch_size, timesteps, _ = state.shape
        state = state.reshape(-1, self.state_size)
        max_q_value = max_q_value.reshape(-1, 1, self.num_agents)
        w1, b1, w2, b2 = self.get_mix_weight(state)
        # First layer
        q_total_1 = F.elu(torch.bmm(max_q_value, w1) + b1)
        # Second layer
        q_total_2 = torch.bmm(q_total_1, w2) + b2

        q_total = q_total_2.squeeze().reshape(batch_size, timesteps)
        return q_total


    def forward(self, obs, *args, **kwargs):
        raise NotImplementedError

    def get_mix_weight(self, state):
        # q_total weight
        w1 = self.hyper_w1(state).reshape(-1, self.num_agents, self.mix_embed_dim)
        b1 = self.hyper_b1(state).unsqueeze(-2)
        w2 = self.hyper_w2(state).unsqueeze(-1)
        b2 = self.hyper_b2(state).unsqueeze(-1)
        return torch.abs(w1), b1, torch.abs(w2), b2
        # return F.softmax(w1, dim=-2), b1, F.softmax(w2, -2), b2

    def init_train_rnn_hidden(self, episode_num):
        # init a gru_hidden for every agent of every episode during training
        self.train_rnn_hidden = torch.zeros((episode_num, self.num_agents, self.net_embed_dim), dtype=torch.float32, device=device)

    def init_eval_rnn_hidden(self):
        # init a gru_hidden for every agent of every episode during evaluating
        self.eval_rnn_hidden = torch.zeros((self.num_agents, self.net_embed_dim), dtype=torch.float32, device=device)


class QMIX_agent(nn.Module):
    def __init__(
            self,
            env_info=None,
            args=None
        ) -> None:
        super(QMIX_agent, self).__init__()
        assert args.multi_steps == 1 and args.is_per == False and args.share_para == True, \
            f"Now QMIX with rnn is not compatible with multi_steps and per, \
                as well as only compatible with share net para"
        self.env_info = env_info
        self.obs_size = self.env_info['obs_shape']
        self.state_size = self.env_info['state_shape']
        self.num_agents = self.env_info['n_agents']
        self.num_actions = self.env_info['n_actions']
        self.episode_limits = self.env_info['episode_limit']

        self.is_ddqn = args.is_ddqn
        self.multi_steps = args.multi_steps
        self.is_per = args.is_per
        self.alpha = args.alpha
        self.beta = args.beta
        self.prior_eps = args.prior_eps
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        
        # Construct Q_net and target_Q_net
        self.Q = QMIX(self.obs_size, self.state_size, self.num_agents, self.num_actions).to(device)
        self.target_Q = QMIX(self.obs_size, self.state_size, self.num_agents, self.num_actions).to(device)
        # self.Q.orth_init() # orthogonal initialization
        self.target_Q.load_state_dict(self.Q.state_dict())
        
        self.params = list(self.Q.parameters())
        self.grad_norm_clip = args.grad_norm_clip
        if args.optimizer == 0:
            # Adam: 3m, 2s_vs_1sc
            self.optimizer = torch.optim.Adam(self.params, args.learning_rate)
        elif args.optimizer == 1:
            # RMSProp alpha:0.99, RMSProp epsilon:0.00001
            self.optimizer = torch.optim.RMSprop(self.params, args.learning_rate, alpha=0.99, eps=1e-5)

        # Consturct buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_size, state_dim=self.state_size, num_agents=self.num_agents, action_dim=self.num_actions,
            ep_limits=self.episode_limits, ep_size=args.replay_buffer_size, multi_steps=self.multi_steps,
            batch_size=self.batch_size
        )

    def get_env_info(self):
        return self.obs_size, self.state_size, self.num_actions, self.num_agents, self.episode_limits

    def can_sample(self):
        return self.replay_buffer.num_in_buffer >= self.batch_size

    def select_actions(self, obs, avail_actions, random_selection):
        '''epsilon greedily select actions according to current obs'''
        # input : obs:(num_agents, obs_shape), avail_actions:(num_agents, num_actions), random_selection:(num_agents,)
        # output: actions: a list that length is num_agents
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        avail_actions = torch.as_tensor(avail_actions, dtype=torch.bool, device=device)
        q_values = self.Q.get_value(obs)
        q_values[avail_actions == 0.0] = -float('inf')
        max_actions = q_values.max(-1)[1].cpu()
        random_actions = Categorical(avail_actions).sample().cpu()
        actions = [random_actions[i].item() if random_selection[i] else max_actions[i].item() for i in range(self.num_agents)]
        return actions

    def update(self):
        '''update Q: 1 step of gradient descent'''
        obs_batchs, act_batchs, avail_act_batchs, \
                    total_obs_batch, total_rew_batch, total_done_batch, = self.replay_buffer.sample()
        
        # Convert numpy nd_array to torch tensor for calculation
        # every agent's experience
        obs_batchs = torch.as_tensor(obs_batchs, dtype=torch.float32, device=device)
        act_batchs = torch.as_tensor(act_batchs, dtype=torch.int64, device=device)
        avail_act_batchs = torch.as_tensor(avail_act_batchs, dtype=torch.float32, device=device)
        total_obs_batch = torch.as_tensor(total_obs_batch, dtype=torch.float32, device=device)
        total_rew_batch = torch.as_tensor(total_rew_batch, dtype=torch.float32, device=device)
        not_done_total = torch.as_tensor(1 - total_done_batch, dtype=torch.float32, device=device)

        # We choose Q based on action taken.
        all_current_Q_values = self.Q.get_batch_value(obs_batchs)
        current_Q_values = all_current_Q_values[:, :-1].gather(-1, act_batchs.unsqueeze(-1)).squeeze(-1)
        total_current_Q_values = self.Q.get_batch_total(current_Q_values, total_obs_batch[:, :-1])
        
        # compute target
        target_Q_output = self.target_Q.get_batch_value(obs_batchs)[:, 1:]
        # Mask out unavailable actions: refer to pymarl
        target_Q_output[avail_act_batchs[:, 1:] == 0.0] = -9999999
        if self.is_ddqn:
            # target_current_Q_values: get target values from current values
            target_current_Q_values = all_current_Q_values.clone().detach()
            target_current_Q_values[avail_act_batchs == 0.0] = -9999999
            target_act_batch = target_current_Q_values[:, 1:].max(-1)[1]
            target_Q_values = target_Q_output.gather(-1, target_act_batch.unsqueeze(-1)).squeeze(-1)
        else:
            target_Q_values = target_Q_output.max(-1)[0]

        total_target_Q_values = self.target_Q.get_batch_total(target_Q_values, total_obs_batch[:, 1:])
        # mask valueless target Q values and compute the target of the current Q values
        total_target_Q_values = total_rew_batch + self.gamma * not_done_total * total_target_Q_values
        
        # take gradient step
        # mask valueless current Q values: In every episode, the first step is always have value
        mask = torch.cat(
            (torch.ones(size=(total_done_batch.shape[0], 1), dtype=torch.float32, device=device), not_done_total[:, :-1]),
            dim=1
        )
        # compute loss: Detach variable from the current graph since we don't want gradients for next Q to propagated
        mask_td_error = (total_current_Q_values - total_target_Q_values.detach()) * mask
        loss = (mask_td_error ** 2).sum() / mask.sum()
        # Clear previous gradients before backward pass
        self.optimizer.zero_grad()
        # run backward pass
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

    def save(self, checkpoint_path):
        '''save model'''
        torch.save(self.Q.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path):
        '''load model'''
        self.Q.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.target_Q.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def evaluate(self, env, episode_num=32):
        '''evaluate Q model'''
        eval_data = []
        for _ in range(episode_num):
            eval_ep_rewards = []
            done = False
            action = None
            env.reset()
            self.Q.init_eval_rnn_hidden()
            while not done:
                last_obs = env.get_obs(action)
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

        return np.mean(eval_data, axis=0)
