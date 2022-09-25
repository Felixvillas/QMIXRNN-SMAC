import os
import numpy as np
from statistics import mean
from tqdm import *
import torch
from tensorboardX import SummaryWriter

from utils.tools import store_hyper_para, construct_results_dir
from utils.schedule import LinearSchedule
from utils.sc_wrapper import single_net_sc2env
from smac.env import StarCraft2Env
from model import QMIX_agent
import time, datetime

def qmix_learning(
    args=None
):
    '''
    Parameters:
    '''
    assert args.save_model_freq % args.target_update_freq == 0
    last_test_t, num_test = -args.test_freq - 1, 0
    if args.seed == None:
        time.sleep(3)
        args.seed = int(''.join(reversed(datetime.datetime.now().strftime("%m%d%H%M%S"))))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Initialize Env
    if args.share_para:
        env = single_net_sc2env(map_name=args.map_name, seed=args.seed)
    else:
        env = StarCraft2Env(map_name=args.map_name, seed=args.seed)
    env_info = env.get_env_info()

    # Initialize qmix_agent
    qmix_agent = QMIX_agent(
            env_info=env_info,
            args=args
    )

    # Construct tensor log writer
    log_dir = construct_results_dir(args)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Construct linear schedule
    exploration_schedule = LinearSchedule(args.anneal_steps, args.anneal_end, args.anneal_start)

    # store hyper parameters
    if args.store_hyper_para:
        store_hyper_para(args, log_dir)

    #############
    #  RUN ENV  #
    #############
    num_param_update = 0
    
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
    t = 0
    pbar = tqdm(total=args.training_steps)

    while t < args.training_steps:
        # run episode
        epsilon = exploration_schedule.value(t)
        ep_rewards, win_flag, episode_len = qmix_agent.run_episode(env, epsilon)

        rewards_queue.append(sum(ep_rewards))
        steps_queue.append(len(ep_rewards))
        win_queue.append(win_flag)
        queue_cursor = min(queue_cursor + 1, queue_maxsize)
        if queue_cursor == queue_maxsize:
            log_rewards.append(mean(rewards_queue[-queue_maxsize:]))
            log_steps.append(mean(steps_queue[-queue_maxsize:]))
            log_win.append(mean(win_queue[-queue_maxsize:]))
            # tensorboard log
            writer.add_scalar(tag=f'starcraft{args.map_name}_train/reward', scalar_value=log_rewards[-1], global_step=t+1)
            writer.add_scalar(tag=f'starcraft{args.map_name}_train/length', scalar_value=log_steps[-1], global_step=t+1)
            writer.add_scalar(tag=f'starcraft{args.map_name}_train/wintag', scalar_value=log_win[-1], global_step=t+1)
            
        t += episode_len
        pbar.update(episode_len)
            
        if args.is_per:
            # PER: increase beta
            qmix_agent.increase_bate(t, args.training_steps)

        # train and evaluate
        if qmix_agent.can_sample():
            # gradient descent: train
            loss = qmix_agent.update()
            num_param_update += 1

            # tensorboard log
            writer.add_scalar(tag=f'starcraft{args.map_name}_train/loss', scalar_value=loss, global_step=t+1)

            # Periodically update the target network by Q network to target Q network
            if num_param_update % args.target_update_freq == 0:
                qmix_agent.update_targets()
            # evaluate the Q-net in greedy mode
            if (t - last_test_t) / args.test_freq >= 1.0:
                eval_data = qmix_agent.evaluate(env, args.evaluate_num)
                # env reset after evaluate             
                writer.add_scalar(tag=f'starcraft{args.map_name}_eval/reward', scalar_value=eval_data[0], global_step=num_test * args.test_freq)
                writer.add_scalar(tag=f'starcraft{args.map_name}_eval/length', scalar_value=eval_data[1], global_step=num_test * args.test_freq)
                writer.add_scalar(tag=f'starcraft{args.map_name}_eval/wintag', scalar_value=eval_data[2], global_step=num_test * args.test_freq)
                last_test_t = t
                num_test += 1
            # model save
            if num_param_update % args.save_model_freq == 0:
                qmix_agent.save(checkpoint_path=os.path.join(log_dir, 'agent.pth'))

    writer.close()
    env.close()

    # last model save
    qmix_agent.save(checkpoint_path=os.path.join(log_dir, 'agent.pth'))