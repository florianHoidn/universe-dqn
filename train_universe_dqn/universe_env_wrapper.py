import baselines.common.atari_wrappers as baselines
import gym
import numpy as np
from universe import spaces
from reward_approximator import RewardFuncApproximator

class OpenAiEnv(gym.Wrapper):
    def __init__(self, env=None, 
                 mouse_area = (30, 95, 666, 575),
                 mouse_grid_stride = 10):
        """ Wraps an OpenAi environment so that it can be used 
        in combination with the baseline DQNs."""
        super(OpenAiEnv, self).__init__(env)
        self.pixel_skip = 3
        self.height, self.width = 768, 1024
        self.env.observation_space.shape = (int(np.ceil(self.height/self.pixel_skip)), int(np.ceil(self.width/self.pixel_skip)), 1) # Note that every other pixel row/column is dropped in the preprocessing step.
        
        self.keys = ['w', 'a', 's', 'd', 'r']
        self.actions = [[[spaces.KeyEvent.by_name(key, down=down_or_up)]] for key in self.keys for down_or_up in [True, False]] 
        
        # If I see it correctly the game sreen is rendered in a rectangle with coordinates ul_x=30, ul_y=95, lr_x=666, lr_y=575.
        # In order to have the AI play BulletFury (which I think would be cool) I'll give it three possible positions to place the mouse at, so that it can turn left and right and aim at the center. Also, I'll give it an action for clicking (=firing) at the center.
        self.game_ulx, self.game_uly, self.game_lrx, self.game_lry = mouse_area if mouse_area != None else (0, 0, self.height, self.width)
        
        self.mouse_grid_stride = mouse_grid_stride
        if self.mouse_grid_stride != None:   
            for x in range(self.game_ulx, self.game_lrx + self.mouse_grid_stride, self.mouse_grid_stride):
                for y in range(self.game_uly, self.game_lry + self.mouse_grid_stride, self.mouse_grid_stride):
                    self.actions.append([[spaces.PointerEvent(x, y, 0)]]) # moves mouse to location on the grid.
                    self.actions.append([[spaces.PointerEvent(x, y, 1)]]) # clicks at location on the grid.       
        
        self.env.action_space.n = len(self.actions)
        self.black_screen = np.zeros(self.env.observation_space.shape)
        
        self.manual_reward = 0
        
    def _step(self, action):
        action = self.actions[int(action)]
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs)            
        return obs, np.sum(reward) + self.manual_reward, done[-1], info

    def _reset(self):
        obs = self.env.reset()
        return self.preprocess(obs)
    
    def preprocess(self, obs):
        if obs is not None and obs[-1] is not None:
            return np.expand_dims(np.uint8(np.mean(np.array(obs[-1]['vision'])[::self.pixel_skip, ::self.pixel_skip], axis = 2)), axis = 2)
        else:
            return self.black_screen

def wrap_openai_universe_game(env, enable_mouse, nbr_frames_stacked=4, 
                              train_reward_approximation = True, 
                              use_approximated_reward = False,
                              train_freq=1,
                              grad_norm_clipping = 10,
                              lr = 1e-4,
                              learning_starts = 2000):
    """Does some normalization steps much like baselines.common.atari_wrappers_deprecated.wrap_dqn 
    but for OpenAI universe environments."""
    env = OpenAiEnv(env) if enable_mouse else OpenAiEnv(env, mouse_area = None, mouse_grid_stride = None)
    env = baselines.MaxAndSkipEnv(env, skip=4)
    env = baselines.FrameStack(env, nbr_frames_stacked)    
    env = RewardFuncApproximator(env, train_reward_approximation=train_reward_approximation, use_approximated_reward = use_approximated_reward, train_freq=train_freq, grad_norm_clipping = grad_norm_clipping, lr = lr, learning_starts = learning_starts)
    return env

def unwrap_openai_env(env):
    while not isinstance(env, OpenAiEnv):
        env = env.env
    return env