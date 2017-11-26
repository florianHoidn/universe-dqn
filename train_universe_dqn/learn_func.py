import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile
import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.simple import ActWrapper, load
from universe_pyglet_controller import PygletController

def learn_off_policy(env,
          q_func,
          sess,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          num_cpu=16,
          callback=None, 
          debug_info_freq = 20,
          train_reward_approximation = True,
          train_rl_algo = True,
          grad_norm_clipping=10):
    """ This slight modification of OpenAI's baselines.deepq.simple.learn function allows you 
    to take control of an OpenAI Universe game manually and have the DQN learn off-policy.

    """
    # Create all the functions necessary to train the model
    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name) 
    
    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=grad_norm_clipping
    )
    
    # this is the only slight change that I made:
    act = PygletController(act, openai_env = env, height=env.observation_space.shape[0], width=env.observation_space.shape[1])
    
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }
    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
       
    path = "./checkpoints/checkpoint_for_" + str(env.spec.id)  
    reward_func_path = path + "/reward_func/"
    rl_algo_path = path + "/rl_algo/"
    reward_func_saver, rl_algo_saver = init_tf_vars(sess, reward_func_path, rl_algo_path, update_target, train_reward_approximation, train_rl_algo)
    reward_func_path += "model.cptk"
    rl_algo_path += "model.cptk"
    
    episode_rewards = [0.0]
    #saved_mean_reward = None
    obs = env.reset()
    
    #with tempfile.TemporaryDirectory() as td:
    model_saved = False
    #model_file = os.path.join(td, "model")   
    
    for t in range(max_timesteps):
        if callback is not None:
            if callback(locals(), globals()):
                break
        # Take action and update exploration to the newest value
        obs_t = np.array(obs)[None]
        action = act(obs_t, update_eps=exploration.value(t))[0]
        new_obs, rew, done, _ = env.step(action)
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
        
        if train_rl_algo:
            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
  
            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                print("Updating target network.")
                update_target()

        if (checkpoint_freq is not None and t % checkpoint_freq == 0):
              
            if train_reward_approximation:                
                saved_at = reward_func_saver.save(sess, reward_func_path)
                print("Reward approximator model saved at %s" % saved_at) 
            if train_rl_algo:                
                saved_at = rl_algo_saver.save(sess, rl_algo_path)
                print("RL model saved at %s" % saved_at)
            model_saved = True
   
    return ActWrapper(act, act_params), replay_buffer, beta_schedule, train, reward_func_saver, reward_func_path, rl_algo_saver, rl_algo_path

def init_tf_vars(sess, reward_func_path, rl_algo_path, update_target, train_reward_approximation, train_rl_algo):
    # I'll create a TensorFlow saver that saves and restores just the reward function approximator's variables.
    # Get all variables in scopes starting with "reward_func":
    reward_func_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reward_func'))
    reward_func_saver =  tf.train.Saver(var_list=reward_func_vars)
    rl_algo_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) - reward_func_vars
    rl_algo_saver =  tf.train.Saver(var_list=rl_algo_vars)
    
    if not os.path.exists(reward_func_path):
        print ("Initializing reward approximator's TF variables.")
        if train_reward_approximation:
            os.makedirs(reward_func_path)
        sess.run(tf.variables_initializer(reward_func_vars))
    else: 
        print("Restoring reward function approximator.")
        reward_func_saver.restore(sess, reward_func_path + "model.cptk")
        
    if not os.path.exists(rl_algo_path):
        print ("Initializing RL algorithm's TF variables.")
        if train_rl_algo:
            os.makedirs(rl_algo_path) 
        sess.run(tf.variables_initializer(rl_algo_vars))        
        update_target()
    else: 
        print("Restoring RL algorithm approximator.")
        rl_algo_saver.restore(sess, rl_algo_path + "model.cptk")
        
    return reward_func_saver, rl_algo_saver