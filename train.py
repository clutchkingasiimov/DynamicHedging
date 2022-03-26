from env import TradingEnv
from agent.dqn import DQN 
import numpy as np 


#Training scheme

#Create the environment 
env = TradingEnv(total_episodes=50_000,num_contracts=1,multiplier=1.0,
tick_size=0.1,kappa=0.1)

