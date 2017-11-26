import numpy as np
import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as tf_utils
import os
import universe_env_wrapper

from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class RewardFuncApproximator(gym.Wrapper):    
    def __init__(self, env=None, 
                 train_reward_approximation = True, 
                 use_approximated_reward = False, 
                 convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], 
                 hiddens=[256], 
                 max_timesteps=2000000,
                 buffer_size=50000,
                 batch_size=32,
                 exploration_fraction=0.1,
                 exploration_final_eps=0.01,
                 train_freq= 1,
                 grad_norm_clipping = 10,
                 lr = 1e-4,
                 learning_starts=2000,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta0=0.4,
                 prioritized_replay_beta_iters=10000,
                 prioritized_replay_eps=1e-6,
                 reward_thresh = 30):
        """A neural network that intercepts the observations, actions and rewards in an OpenAI Universe game and 
        tries to approximate the reward function. This can be useful in games that lack a reward function. One 
        can either train the net on similar games that do have a reward function and see how it performs in 
        games without one, or one can manually generate rewards to train it. Learning to approximate 
        the reward function will often be much simpler than learning to approximate, say, the action value 
        function - think how easy it would, e.g., be to approximate a reward function of chess (high 
        positive/negative reward when you take/lose the king etc.), but how hard it is to approximate its 
        action value function. Training a RewardApproximator should, hence, require much less time than training a 
        proper RL algorithm (imagine sitting through douzands of hours of training your RL algorithm and
        creating rewards manually). 
        In addition, a RewardApproximator might also be useful in games that do come with a reward function:
        For instance, it can be used to train your RL algorithm to do things other than maximizing the 
        game's inbuilt rewards - if you want to train it to do very specific things like picking up certain 
        items or even just running in circles, that should become possible.
        
        Parameters
        -------
        env : gym.Env
            OpenAI Universe environment to train on.
        """        
        super(RewardFuncApproximator, self).__init__(env)
        
        self.train_reward_approximation = train_reward_approximation
        self.use_approximated_reward = use_approximated_reward
        self.grad_norm_clipping = grad_norm_clipping
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.sess = tf_utils.get_session()
        
        # 1. step: Build reward approximator net. I'll try to make sure it resembles the RL algo structurally - 
        # like that, one can also check how well the net's architecture works in a given game.
        self.reward_func, self.observation_input, self.action_input = self._build_reward_func(convs, hiddens)
        
        # 2. step: Build update operation. Again, nothing too fancy. just an AdamOptimizer minimizing the error in the predicted reward.
        self.errors, self.train_reward_func, self.true_rewards = self._build_train_reward_func(self.reward_func, self.observation_input, self.action_input, optimizer)
        
        # 3. step: Build prioritized experience replay buffer (just like the RL algorithm, this function 
        # approximator will receive highly correlated inputs directly from the game, so it's important to mix up 
        # the training batches a little bit. Also, actual rewards will often be sparse and we want to make sure 
        # that we get the most out of situations that actually were rewarded - again just like with the RL 
        # algorithm. So, a prioritized experience replay buffer is just right, here).
        self.step_counter = 0
        if self.train_reward_approximation:
            self.batch_size = batch_size
            self.prioritized_replay_eps = prioritized_replay_eps
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
            self.exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)            
            self.train_freq = train_freq
            self.obs = None # will be initialized on _reset.
                
        self.out = create_csv_for_plotting(["reward", "approximated reward"])
        self.last_rewards_and_approximations = []
        self.learning_starts = learning_starts
        self.reward_thresh = reward_thresh
        
    def _step(self, action):
        new_obs, reward, done, info = self.env.step(action)        

        
        if self.train_reward_approximation:
            # 1. step: Write to replay buffer.
            self.replay_buffer.add(self.obs, action, reward, new_obs, float(done))
            self.obs = new_obs

            if self.step_counter >= self.learning_starts and self.step_counter % self.train_freq == 0:
                # 2. step: draw sample from replay buffer and train.
                self.train_from_replay_buffer(self.step_counter)
        
        self.step_counter += 1

        orig_reward = reward
                   
        if self.use_approximated_reward:
            reward = self.sess.run(self.reward_func, feed_dict={
                self.observation_input:np.array(new_obs)[None],
                self.action_input:[action]
            })[0,0]
            
            reward = reward if np.abs(reward) >= self.reward_thresh else 0 # drop small rewards to reduce noise.

            approx_reward = reward # keep for plotting
        else: 
            approx_reward = self.sess.run(self.reward_func, feed_dict={
                self.observation_input:np.array(new_obs)[None],
                self.action_input:[action]
            })[0,0]            
            
        self.last_rewards_and_approximations.append([orig_reward, approx_reward])
        if self.out is not None and self.step_counter % 20 == 0:
            write_values_to_csv(self.out, self.last_rewards_and_approximations)
            self.last_rewards_and_approximations = []
            
        return new_obs, reward, done, info

    def _reset(self):
        self.obs = self.env.reset()
        return self.obs
    
    def _build_reward_func(self, convs, hiddens):
        with tf.variable_scope("reward_func"):
            observation_input_ph = tf.placeholder(shape=[None] + list(self.env.observation_space.shape), dtype=tf.float32, name = "observation_input")
            cnn_out = observation_input_ph
            
            # 1. step: Build convolution to process frame.
            with tf.variable_scope("cnn"):
                for num_outputs, kernel_size, stride in convs:
                    cnn_out = layers.convolution2d(cnn_out,
                                               num_outputs=num_outputs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               activation_fn=tf.nn.relu)
            cnn_out = layers.flatten(cnn_out)
            
            # 2. step: Append one-hot representation of the chosen action (like that, it will be possible to reward specific sequences of actions, like moving in a circle, say).
            action_input_ph = tf.placeholder(tf.int32, [None], name="action_input")
            out = tf.concat([cnn_out, tf.one_hot(action_input_ph, self.env.action_space.n)], axis=1)        
            
            with tf.variable_scope("hidden"):
                reward_approximation = out
                for hidden in hiddens:
                    reward_approximation = layers.fully_connected(reward_approximation, num_outputs=hidden, activation_fn=tf.nn.relu)
            
            with tf.variable_scope("output"):   
                return layers.fully_connected(reward_approximation, num_outputs=1, activation_fn=None), observation_input_ph, action_input_ph 
            
    def _build_train_reward_func(self, reward_func, observation_input_ph, action_input_ph, optimizer):
        with tf.variable_scope("reward_func_optimizer"):
            true_rewards_ph = tf.placeholder(tf.float32, [None], name="true_rewards")                   
            #loss = tf.metrics.mean_squared_error(reward_func, true_rewards_ph)
            true_rewards = tf.expand_dims(true_rewards_ph, axis=1)
            
            #loss = tf.reduce_mean(tf.losses.huber_loss(reward_func, true_rewards), name = "loss") # Maybe a bit more robust.
            errors = reward_func - true_rewards
            loss = tf.reduce_mean(tf_utils.huber_loss(errors), name="loss")
            gradients = optimizer.compute_gradients(loss)
            for i, (grad, var) in enumerate(gradients):
                gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)
            train_reward_func = optimizer.apply_gradients(gradients)

            return errors, train_reward_func, true_rewards_ph
    
    def _close(self):
        if self.out is not None:
            self.out.close()
        self.env.close()
        
    def train_from_replay_buffer(self, step_counter):
        # 1. step: Sample from experience buffer.                
        experience = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(step_counter))
        obses_t, actions, rewards, _, _, _, batch_idxes = experience

        # 2. step: Train.
        approx_rews, batch_errors, _ = self.sess.run([self.reward_func, self.errors, self.train_reward_func], feed_dict={
            self.observation_input:obses_t, 
            self.action_input:actions,
            self.true_rewards:rewards
        })

        # 3. step: Update replay buffer priorities.
        new_priorities = np.abs(batch_errors) + self.prioritized_replay_eps
        self.replay_buffer.update_priorities(batch_idxes, new_priorities)
        return batch_errors
        
def create_csv_for_plotting(headers, plot_file = "csv_for_plotting.txt"):
    
    if plot_file is None:
        return None

    if os.path.exists(plot_file):
        os.remove(plot_file)
    out = open(plot_file, 'w+')
    out.write(",".join(headers) + "\n")
    out.flush()

    return out

def write_values_to_csv(out, list_of_value_tuples):
    for values in list_of_value_tuples:
        csv_line = ",".join([str(round(v,3)) for v in values]) + "\n"
        out.write(csv_line)
    out.flush()