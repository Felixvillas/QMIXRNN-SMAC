import numpy as np
import os
import datetime

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

def construct_results_dir(args):
    log_dir = f'./results/StarCraft/{args.map_name}/'
    log_dir = log_dir + args.map_name
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
    return log_dir