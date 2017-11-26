import gym
import universe
from baselines import deepq
import time
import tensorflow as tf

from universe_env_wrapper import wrap_openai_universe_game
from learn_func import learn_off_policy


def main():    
    with tf.Session() as sess:
        
        # The list with all available games can be found in universe/__init__.py
        env = gym.make('flashgames.BulletFury-v0')
        #env = gym.make('flashgames.NeonRace-v0')
        env.configure(remotes=1) # use this tok start a new docker container.

        # Use "docker ps" in a terminal to find the <id> of an existing OpenAi container and then "docker restart <id>" 
        # to restart it (if it isn't started already). You can then reconnect to it which is super convenient (much fewer logs!).
        #env.configure(remotes='vnc://localhost:5900+15900') 

        train_reward_approximation = False
        train_rl_algo = True
        use_approximated_reward = True
        train_freq = 2
        grad_norm_clipping=10
        max_timesteps=200000
        learning_starts = 5000
        
        env = wrap_openai_universe_game(env, enable_mouse=True, 
                            train_reward_approximation = train_reward_approximation,
                            use_approximated_reward = use_approximated_reward,
                            train_freq = train_freq,
                            lr=1e-4,
                            grad_norm_clipping = grad_norm_clipping,
                            learning_starts = learning_starts)        
        env.reset()

        model = deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], #[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=True
        )
        
        checkpoint_freq=2000
        learn_res = learn_off_policy( 
            env = env,
            q_func = model,
            sess = sess,
            lr=1e-6,
            max_timesteps=max_timesteps,
            buffer_size=50000,
            exploration_fraction= 0.9, # It takes exploration_fraction * max_timesteps many steps unil epsilon reaches exploration_final_eps.
            exploration_final_eps=0.01,
            train_freq = train_freq, 
            batch_size=32,
            checkpoint_freq=checkpoint_freq,
            learning_starts=learning_starts,
            target_network_update_freq=2000,
            gamma=0.99,
            num_cpu=0, # Tensorflow's default: appropriate number of cores is picked automatically.
            prioritized_replay=True,
            debug_info_freq = 500,
            train_reward_approximation = train_reward_approximation,
            train_rl_algo = train_rl_algo,
            grad_norm_clipping=grad_norm_clipping
        )
                    
if __name__ == '__main__':
    main()