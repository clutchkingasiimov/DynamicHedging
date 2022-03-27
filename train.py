from statistics import mean
from env import TradingEnv
from agent.dqn import DQN 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm


#Training scheme
'''
1. Create the training data and the environment 
'''

#Create the environment 
mean_reward_per_epoch = []
# for _ in tqdm(range(5),desc="Epoch"):
env = TradingEnv(total_episodes=100,num_contracts=1,multiplier=1.0,
tick_size=0.1,kappa=0.1)

state_space = env.state_space
action_space = env.action_space

reward_per_episode = []

EPISODES = env.total_episodes
#Initialize the agent 
agent = DQN(state_space,action_space,load_model=False)

for e in tqdm(range(EPISODES),desc="Episode"):
    done = False
    state = env.reset()
    state = np.reshape(state, [1, state_space])
    score = 0 

    while not done: 
        action = agent.initialize_agent(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_space])

        #Save [S,A,R,S',D] in replay memory 
        agent.update_replay_memory(state,action,reward,next_state,done)
        #Every time step, train the model 
        agent.train_agent()
        score += reward
        state = next_state

        if done:
            #Update the target model to be the same with the prediction model 
            agent.update_target_model()
            reward_per_episode.append(score)
    # mean_reward_per_epoch.append(np.mean(reward_per_episode))
# plt.plot(mean_reward_per_epoch)
# plt.show()
plt.plot(reward_per_episode)
plt.show()


print('Done!')





