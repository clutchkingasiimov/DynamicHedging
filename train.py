from env import TradingEnv
from agent.dqn import DQN 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm

#Create the environment
env = TradingEnv(init_price=100,sample_size=300,num_contracts=1,multiplier=1.0,
tick_size=0.1,kappa=0.1)

STATE_SPACE = env.state_space
ACTION_SPACE = env.action_space
EPISODES = env.sample_size


#Create the agent 
agent = DQN(STATE_SPACE,ACTION_SPACE,load_model=False)

mean_reward_per_epoch = []
rewards = []

for i in range(10):
    #Generate new data by reseting the simulation
    print(f'Epoch {i+1}')
    env.set_params()

    #Track the reward each episode
    reward_per_episode = []

    for e in tqdm(range(EPISODES),desc="Episode"):
        done = False
        state = env.reset()
        state = np.reshape(state, [1, STATE_SPACE])
        score = 0 

        while not done: 
            action = agent.initialize_agent(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, STATE_SPACE])

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
                rewards.append(score)
    mean_reward_per_epoch.append(np.mean(reward_per_episode))

    #Save model each epoch 
    agent.prediction_model.save_weights(f"/home/sauraj/Desktop/DynamicHedging/model_saves/DQN_epoch_{i+1}.h5")
    print('Model Saved!')


#Save the average reward each episode
reward_csv = pd.DataFrame({'Epoch':[1,2,3,4,5],'Mean_Reward':mean_reward_per_epoch})
reward_csv.to_csv('Mean_Reward.csv')
print('Reward CSV saved!')
plt.plot(mean_reward_per_epoch)
plt.show()


print('Done!')





