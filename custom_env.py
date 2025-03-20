import numpy as np
import gymnasium as gym
# import time
from gymnasium import spaces
from cuPSS_server import cuPSS_server
from utils import Renderer, DataHandler, find_interface


class cuPSS_env(gym.Env):
    def __init__(self, env_config ) -> None:
        super().__init__()

        self._s_name = env_config['sname']
        self.nx, self.ny = env_config['size']
        self.rmode = env_config['rmode']
        self.activity_mode = "all" # "all" or "interface"
        assert self.activity_mode in ["all", "interface"]


        # cuPSS related initializations
        self.cupss_channel_order = {0: "Qxx", 1: "Qxy", 2: "vx", 3: "vy", 4: "phi"}
        self.n_channels = len(self.cupss_channel_order.keys())

        # environment options and configurations
        self.bins = self.ny
        self.action_space = spaces.Box(low = 0, high = 5, shape = (self.bins,), dtype = np.float64)
        self.observation_space = spaces.Box(low = -100, high = 100, shape = (3*self.bins, ), dtype = np.float64)
        
        self.steps_per_episode = 10000
        self.trajectory_reward = 0
        self._steps = 0

        # Handles render configs and functions 
        render_conf = {'size': (self.nx, self.ny), 'channel_order': self.cupss_channel_order, 'fps': 50, 'bin_size': self.bins}
        self.renderer = Renderer(self.rmode, render_conf)

        # Handles data processing and analysis functions 
        data_handler_config = {'size': (self.nx, self.ny), 'channel_order': self.cupss_channel_order, 'bin_size': self.bins}
        self.dhandle = DataHandler(data_handler_config)

        # Handles interaction with cupss 
        self.server = cuPSS_server(server_name = self._s_name) 



    def reset(self, seed = 42, options = None): 
        self._steps = 0
        self.trajectory_reward = 0
        self.server.reset()
        data = self._initialize_env_state(seed = seed)
        observation = self.dhandle.get_obs(data) 
        self._last_observation = observation
        return observation, {'trajectory_reward': self.trajectory_reward}


    def step(self, action):
        self._curr_action = action
        action_to_write = self.dhandle.action_preprocess(action, self._last_observation, mode = self.activity_mode)
        self.server.write(action_to_write)
        self._steps += 1

        # read the response from cuPSS server and render the observed data
        obs_df = self.dhandle.extract(self.server.read())
        obs_df['action'] = action_to_write
        obs_df['action_out'] = self._curr_action 

        observation = self.dhandle.get_obs(obs_df)
        obs_df['observation'] = observation
        self._last_observation = observation

        self.renderer.render(obs_df)
        # calculate reward based on the observation and the target  
        reward = 0 #self.dhandle._calc_reward(obs_df)
        self.trajectory_reward += reward

        # conditions for trunctions and termination.
        # ampl = max(observation[:self.bins])-min(observation[:self.bins])
        terminated: bool = False  # add a constraint that if interface fluctuations increase beyond a point terminate
        truncated: bool  = True if self._steps > self.steps_per_episode else False  
        info = {'trajectory_reward': self.trajectory_reward}
        info.update(obs_df)

        return observation, reward, terminated, truncated, info


    def _initialize_env_state(self, seed):
        data = self.dhandle.generate_init_state(seed = seed)
        for i in range(self.n_channels):
            field = data[self.cupss_channel_order[i]]
            self.server.write(field)

        return data
        

    def close(self):
        self.server.close()



# class cuPSS_env(gym.Env):
#     def __init__(self, env_config ) -> None:
#         self.bins = 16
#         self.action_space = spaces.Box(low = 0, high = 1, shape = (self.bins,), dtype = np.float64)
#         self.observation_space = spaces.Box(low = 0.2, high = 0.8, shape = (self.bins, ), dtype = np.float64)
        


#     def reset(self, seed = 42, options = None): 
#         observation = self.observation_space.sample() 
#         return observation, {'trajectory_reward': 0}


#     def step(self, action):
#         observation = self.observation_space.sample()
#         reward = 0
#         terminated: bool = False  # add a constraint that if interface fluctuatioons increase beyond a point terminate
#         truncated: bool  = False  
#         info = {'trajectory_reward': 0}
 
#         return observation, reward, terminated, truncated, info


