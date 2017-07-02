import gym
import universe
from baselines import deepq
import time

def main():
    env = gym.make('flashgames.NeonRace-v0')
    #env = gym.make('flashgames.CoasterRacer-v0') #There seems to be an open issue with git ifs or something.
    env.configure(remotes=1)
    env = wrap_openai_universe_game(env)
    env.reset()

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], #[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )

    act = learn_off_policy( 
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=2000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq= 4,
        learning_starts=100,
        target_network_update_freq=1000,
        gamma=0.99,
        num_cpu=0, # Tensorflow's default: appropriate number of cores is picked automatically.
        prioritized_replay=True
    )
    act.save("openai_universe_model_for_"+str(self.env.spec.id)+".pkl")
    env.close()

if __name__ == '__main__':
    main()