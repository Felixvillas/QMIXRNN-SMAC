import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import mean
from tqdm import *
import torch
import datetime

from tensorboardX import SummaryWriter

def init_episode_temp(ep_limits, state_shape, num_agents, obs_dim, action_dim):
    episode_obs = np.zeros((ep_limits+1, num_agents, obs_dim), dtype=np.float32)
    episode_state = np.zeros((ep_limits+1, state_shape), dtype=np.float32)
    episode_action = np.zeros((ep_limits+1, num_agents), dtype=np.int64)
    episode_reward = np.zeros((ep_limits+1), dtype=np.float32)
    episode_avail_action = np.zeros((ep_limits+1, num_agents, action_dim), dtype=np.float32)
    return episode_obs, episode_state, episode_action, episode_reward, episode_avail_action

def store_hyper_para(args, store_path):
    argsDict = args.__dict__
    f = open(os.path.join(store_path, 'hyper_para.txt'), 'w')
    f.writelines('======================starts========================' + '\n')
    for key, value in argsDict.items():
        f.writelines(key + ':' + str(value) + '\n')
    f.writelines('======================ends========================' + '\n')
    f.close()
    print('==================hyper parameters store done!==================')
    

def qmix_learning(
    env_class,
    q_func,
    exploration,
    args=None
):
    '''
    Parameters:
    '''
    assert args.save_model_freq % args.target_update_freq == 0
    last_test_t, num_test = -args.test_freq - 1, 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Initialize Env
    env = env_class(map_name=args.map_name, seed=args.seed)
    env_info = env.get_env_info()
    # Initialize QMIX_agent
    QMIX_agent = q_func(
            env_info=env_info,
            args=args
    )
    obs_size, state_size, num_actions, num_agents, episode_limit = QMIX_agent.get_env_info()

    # Construct tensor log writer
    env_name = args.map_name
    log_dir = f'./results/StarCraft/{env_name}/'
    log_dir = log_dir + env_name
    if args.is_ddqn:
        log_dir = log_dir + '_ddqn'
    if args.multi_steps > 1:
        log_dir = log_dir + f'_{args.multi_steps}multisteps'
    if args.is_per:
        log_dir = log_dir + '_per'
    if args.share_para:
        log_dir = log_dir + '_sharepara'
    log_dir = log_dir + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = log_dir + f'seed_{args.seed}_{datetime.datetime.now().strftime("%m%d_%H-%M-%S")}/'
    writer = SummaryWriter(log_dir=log_dir)
    
    # store hyper parameters
    if args.store_hyper_para:
        store_hyper_para(args, log_dir)

    #############
    #  RUN ENV  #
    #############
    num_param_update = 0
    env.reset()
    QMIX_agent.Q.init_eval_rnn_hidden()
    episode_obs, episode_state, episode_action, episode_reward, episode_avail_action = \
        init_episode_temp(episode_limit, state_size, num_agents, obs_size, num_actions)

    last_obs = env.get_obs()
    last_state = env.get_state()
    # for episode experience
    ep_rewards = []
    episode_len = 0
    
    # log paramaters
    log_rewards = []
    log_steps = []
    log_win = []
    queue_maxsize = 32
    queue_cursor = 0
    rewards_queue = []
    steps_queue = []
    win_queue = []

    # refer pymarl: in every episode, t in exploration.value(t) is consistent
    t_exploration = 0

    for t in tqdm(range(args.training_steps)):

        # get avail action for every agent
        avail_actions = env.get_avail_actions()

        # eps-greedily select actions
        random_selection = np.random.random(num_agents) < exploration.value(t_exploration)
        # last_obs is a list of array that shape is (obs_shape,) --> numpy.array:(num_agents, obs_shape)
        recent_observations = np.concatenate([np.expand_dims(ob, axis=0) for ob in last_obs], axis=0)
        action = QMIX_agent.select_actions(recent_observations, avail_actions, random_selection)
        
        # Advance one step
        reward, done, info = env.step(action)

        # experience
        episode_obs[episode_len] = np.concatenate([np.expand_dims(ob, axis=0) for ob in last_obs], axis=0)
        episode_state[episode_len] = last_state
        episode_action[episode_len] = np.array(action)
        episode_reward[episode_len] = reward
        episode_avail_action[episode_len] = np.array(avail_actions)

        ep_rewards.append(reward)
        obs = env.get_obs(action)
        state = env.get_state()

        # Resets the environment when reaching an episode boundary
        if done:
            '''for last experience in every episode'''
            # get avail action for every agent
            avail_actions = env.get_avail_actions()
            # eps-greedily select actions
            random_selection = np.random.random(num_agents) < exploration.value(t_exploration)
            # last_obs is a list of array that shape is (obs_shape,) --> numpy.array:(num_agents, obs_shape)
            recent_observations = np.concatenate([np.expand_dims(ob, axis=0) for ob in obs], axis=0)
            action = QMIX_agent.select_actions(recent_observations, avail_actions, random_selection)
            episode_obs[episode_len+1] = np.concatenate([np.expand_dims(ob, axis=0) for ob in obs], axis=0)
            episode_state[episode_len+1] = state
            episode_action[episode_len+1] = np.array(action)
            episode_reward[episode_len+1] = 0
            episode_avail_action[episode_len+1] = np.array(avail_actions)

            # store one episode experience into buffer
            episode_dict = {
                'obs': episode_obs, 
                'action': episode_action, 
                'avail_action': episode_avail_action
            }
            total_episode_dict = {
                'obs': episode_state, 
                'reward': episode_reward, 
            }
            QMIX_agent.replay_buffer.store(episode_dict, total_episode_dict, episode_len)

            # tensorboard log
            rewards_queue.append(sum(ep_rewards))
            steps_queue.append(len(ep_rewards))
            win_queue.append(1. if 'battle_won' in info and info['battle_won'] else 0.)
            queue_cursor = min(queue_cursor + 1, queue_maxsize)
            if queue_cursor == queue_maxsize:
                log_rewards.append(mean(rewards_queue[-queue_maxsize:]))
                log_steps.append(mean(steps_queue[-queue_maxsize:]))
                log_win.append(mean(win_queue[-queue_maxsize:]))
                # tensorboard log
                writer.add_scalar(tag=f'starcraft{env_name}_train/reward', scalar_value=log_rewards[-1], global_step=t+1)
                writer.add_scalar(tag=f'starcraft{env_name}_train/length', scalar_value=log_steps[-1], global_step=t+1)
                writer.add_scalar(tag=f'starcraft{env_name}_train/wintag', scalar_value=log_win[-1], global_step=t+1)
            
            ep_rewards = []
            episode_len = 0
            
            env.reset()
            QMIX_agent.Q.init_eval_rnn_hidden()
            obs = env.get_obs()
            state = env.get_state()
            # init para for new episide
            episode_obs, episode_state, episode_action, episode_reward, episode_avail_action = \
                init_episode_temp(episode_limit, state_size, num_agents, obs_size, num_actions)
            # update t_exploration
            t_exploration = t
        else:
            episode_len += 1

        last_obs = obs
        last_state = state
        
        if args.is_per:
            # PER: increase beta
            QMIX_agent.increase_bate(t, args.training_steps)

        # train and evaluate
        if (done and QMIX_agent.can_sample()):
            # gradient descent: train
            loss = QMIX_agent.update()
            num_param_update += 1

            # tensorboard log
            writer.add_scalar(tag=f'starcraft{env_name}_train/loss', scalar_value=loss, global_step=t+1)

            # Periodically update the target network by Q network to target Q network
            if num_param_update % args.target_update_freq == 0:
                QMIX_agent.update_targets()
            # evaluate the Q-net in greedy mode
            if (t - last_test_t) / args.test_freq >= 1.0:
                eval_data = QMIX_agent.evaluate(env, args.evaluate_num)                               
                writer.add_scalar(tag=f'starcraft{env_name}_eval/reward', scalar_value=eval_data[0], global_step=num_test * args.test_freq)
                writer.add_scalar(tag=f'starcraft{env_name}_eval/length', scalar_value=eval_data[1], global_step=num_test * args.test_freq)
                writer.add_scalar(tag=f'starcraft{env_name}_eval/wintag', scalar_value=eval_data[2], global_step=num_test * args.test_freq)
                last_test_t = t
                num_test += 1
            # model save
            if num_param_update % args.save_model_freq == 0:
                QMIX_agent.save(checkpoint_path=os.path.join(log_dir, 'agent.pth'))

    ### log train results
    df = pd.DataFrame({})
    df.insert(loc=0, column='rewards', value=log_rewards)
    df.insert(loc=1, column='steps', value=log_steps)
    df.insert(loc=2, column='wintag', value=log_win)
    df_avg = pd.DataFrame({})
    df_avg.insert(loc=0, column='rewards', 
        value=df['rewards'].rolling(window=20, win_type='triang', min_periods=1).mean())
    df_avg.insert(loc=0, column='steps', 
        value=df['steps'].rolling(window=20, win_type='triang', min_periods=1).mean())
    df_avg.insert(loc=2, column='wintag',
        value=df['wintag'].rolling(window=20, win_type='triang', min_periods=1).mean())
    _, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(df_avg['rewards'], label='rewards')
    ax1.set_ylabel('rewards')
    ax2.plot(df_avg['steps'], label='steps')
    ax2.set_ylabel('steps')
    ax3.plot(df_avg['wintag'], label='wintag')
    ax3.set_ylabel('wintag')

    ax1.set_title(f'{env_name}-{num_agents}agents')
    ax2.set_xlabel('∝episode')
    plt.legend()
    plt.savefig(log_dir + env_name)

    writer.close()
    env.close()
