from env import TradingEnv
from agent.dqn import DQN 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm

#Create the environment
env = TradingEnv(init_price=100,sample_size=30_000,num_contracts=1,multiplier=1.0,
tick_size=0.1,kappa=0.1)

STATE_SPACE = env.state_space
ACTION_SPACE = env.action_space
EPISODES = env.sample_size

#Create the agent 
agent = DQN(STATE_SPACE,ACTION_SPACE,load_model=True)

#Generate new data by reseting the simulation
env.set_params()

#Track the reward each episode
stock_position = []
stock_pnl = []
option_pnl = []
costs = []
total_pnl = []
rewards = []

for e in tqdm(range(EPISODES),desc="Episode"):
    done = False
    state = env.reset()
    state = np.reshape(state, [1, STATE_SPACE])
    score = 0 

    while not done: 
        #Make prediction of the optimal action 
        action = agent.initialize_agent(state)
        next_state, reward, done = env.step(action)

        #Store all the outputs
        option_profit = env.option_pnl(next_state[2])
        stock_profit = env.wealth_of_trade(next_state[0],next_state[2])
        trading_cost = env.cost_of_trade(next_state[2])
        stock_position.append(next_state[2])
        total_pnl.append(option_profit+stock_profit)
        rewards.append(reward)

        #Transition to next state
        next_state = np.reshape(next_state, [1, STATE_SPACE])

        state = next_state

        #Append all the values 
        stock_pnl.append(stock_profit)
        option_pnl.append(option_profit)
        costs.append(trading_cost)


# plt.plot(stock_position,color='blue')
# plt.plot(stock_pnl,color='red')
# plt.plot(option_pnl,color='green')
# plt.plot(costs,color='orange')
# plt.plot(total_pnl,color='purple')
# plt.xlabel('Days')
# plt.ylabel('Value')
# plt.plot(rewards)
# pnl = pd.DataFrame({'Stock_PnL': stock_pnl,'Option_PnL': option_pnl,'Total_PnL':total_pnl,
# 'Costs':costs})

# pnl.to_csv('pnl.csv')
# plt.legend(['Stock Position','Stock PnL','Option PnL','Trading Costs','Total PnL'])
# plt.show()


print('Done!')
