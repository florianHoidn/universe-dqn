from baselines.common.atari_wrappers import wrap_deepmind
import gym
import numpy as np

class OpenAiEnv(gym.Wrapper):
    def __init__(self, env=None):
        """ Wraps an OpenAi environment so that it can be used 
        in combination with the baseline DQNs."""
        super(OpenAiEnv, self).__init__(env)
        self.pixel_skip = 3
        self.env.observation_space.shape = [int(np.ceil(768/self.pixel_skip)), int(np.ceil(1024/self.pixel_skip)), 1] # Note that every other pixel row/column is dropped in the preprocessing step.
        
        keys = ['up', 'left', 'down', 'right', 'x']
        self.key_events = [[[spaces.KeyEvent.by_name(key, down=down_or_up)]] for key in keys for down_or_up in [True, False]] 
        self.env.action_space.n = len(self.key_events)
        self.black_screen = np.zeros(self.env.observation_space.shape)
        
    def _step(self, action):
        action = self.key_events[int(action)]
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs)            
        return obs, np.sum(reward), done[-1], info

    def _reset(self):
        obs = self.env.reset()
        return self.preprocess(obs)
    
    def preprocess(self, obs):
        if obs is not None and obs[-1] is not None:
            return np.expand_dims(np.uint8(np.mean(np.array(obs[-1]['vision'])[::self.pixel_skip, ::self.pixel_skip], axis = 2)), axis = 2)
        else:
            return self.black_screen

def wrap_openai_universe_game(env):
    """Does some normalization steps much like baselines.common.atari_wrappers_deprecated.wrap_dqn 
    but for OpenAI universe environments."""
    env = OpenAiEnv(env)
    env = baseline.MaxAndSkipEnv(env, skip=4)
    env = baseline.FrameStack(env, 4)
    return env            
