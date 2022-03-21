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

        self.num_simulations = 100
        self.num_contracts = num_contracts 
        self.multiplier = multiplier 
        self.tick_size = tick_size
        self.kappa = kappa  

        os = OptionSimulation(100,self.num_simulations) 
        
        #can add a maturity term
        self.sim_prices = os.GBM(50,0.5,time_increment=1)
        self.days_to_expiry = os.ttm #Creates an array of days left to expiry 
        self.option_price_path, self.option_delta_path = os.BS_call(self.days_to_expiry,self.sim_prices,100,0.01,0,0)
        
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
    
    def reset(self, path):
        # repeatedly go through available simulated paths (if needed)
        self.t = 0
        self.path = path
        ttm = self.days_to_expiry[0]-1

        n = 500 #no of shares
        #print(self.sim_prices[self.path])
        #print(self.sim_prices[self.path,0])
        #print(self.option_delta_path[self.path,self.t])
        
        #K : strike price (to be given)

        price =  round(self.sim_prices[self.path,self.t])
        self.nt = n
        price_ttm = round(self.sim_prices[self.path,ttm])
        
        self.state = [price , ttm, self.nt, price_ttm]

        return self.state
    
    def delta(self, ttm):
        delta = self.option_delta_path[self.path, ttm]
        return delta

    def step(self,action):
        
        self.t = self.t + 1 
        price =  round(self.sim_prices[self.path,self.t])
        self.nt = self.nt + action
        ttm = self.days_to_expiry[self.t] 
        price_ttm = round(self.sim_prices[self.path,ttm])
        
        R = round(self.reward(price, self.nt)) 
        
        self.state = [price , ttm, self.nt, price_ttm]
        
        if ttm == 0:
            done = True
        else:
            done = False
        
        return self.state, R, done


    
if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    env = TradingEnv(num_simulations=100,num_contracts=5,multiplier=1.0,
    tick_size=0.1,kappa=0.1)


