from smac.env import StarCraft2Env
import numpy as np

class single_net_sc2env(StarCraft2Env):
    def __init__(self, map_name="8m", step_mul=8, move_amount=2, difficulty="7", game_version=None, seed=None, continuing_episode=False, obs_all_health=True, obs_own_health=True, obs_last_action=False, obs_pathing_grid=False, obs_terrain_height=False, obs_instead_of_state=False, obs_timestep_number=False, state_last_action=True, state_timestep_number=False, reward_sparse=False, reward_only_positive=True, reward_death_value=10, reward_win=200, reward_defeat=0, reward_negative_scale=0.5, reward_scale=True, reward_scale_rate=20, replay_dir="", replay_prefix="", window_size_x=1920, window_size_y=1200, heuristic_ai=False, heuristic_rest=False, debug=False):
        super().__init__(map_name, step_mul, move_amount, difficulty, game_version, seed, continuing_episode, obs_all_health, obs_own_health, obs_last_action, obs_pathing_grid, obs_terrain_height, obs_instead_of_state, obs_timestep_number, state_last_action, state_timestep_number, reward_sparse, reward_only_positive, reward_death_value, reward_win, reward_defeat, reward_negative_scale, reward_scale, reward_scale_rate, replay_dir, replay_prefix, window_size_x, window_size_y, heuristic_ai, heuristic_rest, debug)

        self.action_one_hot = np.eye(self.n_actions)
        self.action_one_hot = np.concatenate([self.action_one_hot, np.zeros((1, self.n_actions))], axis=0)
        self.agent_one_hot_id = np.eye(self.n_agents)
        self.new_last_action = (np.zeros(self.n_agents).astype(np.int64) - 1).tolist()

    def get_env_info(self):
        env_info = super().get_env_info()
        # obs --> (obs, one-hot of agent_idx, avail_actions)
        env_info['obs_shape'] += env_info['n_agents'] + env_info['n_actions']
        return env_info

    def get_obs(self, last_action=None):
        if last_action is None:
            # the first step in a episode
            last_action = self.new_last_action
        obs = super().get_obs()
        obs_copy = []
        for idx in range(self.n_agents):
            obs_copy.append(
                np.concatenate(
                    [obs[idx], self.action_one_hot[last_action[idx]], self.agent_one_hot_id[idx]], axis=0
                )
            )
        return obs_copy

