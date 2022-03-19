from tracemalloc import start
from unicodedata import name
import gym 
from gym import spaces 
from gym.utils import seeding 
import numpy as np 
from Simulator.simulations import OptionSimulation


class TradingEnv(gym.Env):
    trading_days = 252 #Number of trading days in one year 
    num_of_shares = 100 #Vanilla options contract size
    """
    Trading Enviroment class with all the modules related 
    to performing trading under a controlled simulation environment.

    Parameters:
        num_simulations: The number of GBM and BS simulations to run for the agent to train on
        num_contracts: The number of contracts the agent will hold.
        multiplier: Float value required for the intensity of the bid-offer spread 
        tick_size: Used for computing the cost relative to the midpoint of the bid-offer spread
        kappa: The risk factor of the portfolio
    """

    def __init__(self,num_simulations=int,num_contracts=int,
    multiplier=float,tick_size=float,kappa=float):

        self.num_simulations = num_simulations
        self.num_contracts = num_contracts 
        self.multiplier = multiplier 
        self.tick_size = tick_size
        self.kappa = kappa  

        os = OptionSimulation(100,self.num_simulations) 

        self.sim_prices = os.GBM(10,5,0.01,time_increment=1)
        self.days_to_expiry = os.ttm/self.trading_days #Creates an array of days left to expiry 
        self.option_price_path, self.option_delta_path = os.BS_call(self.days_to_expiry,self.sim_prices,100,0.05,0,0)

        #Action space (Discrete)
        self.num_actions = self.num_contracts*self.num_of_shares #Number of actions 
        self.action_space = spaces.Discrete(1001,start=-self.num_actions) #Discrete action space 

        if self.num_contracts > 10:
            raise ValueError("The maximum number of contracts in the simulation cannot be more than 10.")

    def _cost_of_trade(self,n):
        #n: Number of shares 
        cost = self.multiplier * self.tick_size * (np.abs(n) * 0.01*n*n)
        return cost 

    def _wealth_of_trade(self,pt,n):
        #W_{t} = q_{t} - c_{t} (pt: Price of the stock at time 't')
        ct = self._cost_of_trade(n)
        wt = pt - ct 
        return wt 

    def reward(self, pt, n):
        '''
        Computes the reward given to the agent

        Parameters:
            pt: Price at time 't'
            n: Number of shares at time 't'
        '''
        wt = self._wealth_of_trade(pt, n)
        rwd = wt - (self.kappa*0.5)*(wt**2) 
        return rwd 

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    env = TradingEnv(num_simulations=100,num_contracts=5,multiplier=1.0,
    tick_size=0.1,kappa=0.1)


