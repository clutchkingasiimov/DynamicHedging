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
    
    def __init__(self,num_simulations=int,num_contracts=int,
    multiplier=float,tick_size=float,kappa=float):
        
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

        self.num_simulations = num_simulations
        self.num_contracts = num_contracts 
        self.multiplier = multiplier 
        self.tick_size = tick_size
        self.kappa = kappa  

        os = OptionSimulation(100,self.num_simulations) 
        
        #can add a maturity term
        self.sim_prices = os.GBM(50,0.1,time_increment=1)
        self.days_to_expiry_normalized = os.ttm/self.trading_days #Only to be used for the calculation of BS call price
        self.days_to_expiry = os.ttm #Creates an array of days left to expiry 
        self.option_price_path, self.option_delta_path = os.BS_call(self.days_to_expiry_normalized,self.sim_prices,100,0.01,0,0)
        
        #Action space (Discrete)
        self.num_actions = self.num_contracts*self.num_of_shares #Number of actions 
        self.action_space = spaces.Discrete(1001,start=-self.num_actions) #Discrete action space 

        if self.num_contracts > 10:
            raise ValueError("The maximum number of contracts in the simulation cannot be more than 10.")

    @classmethod
    def change_base_params(cls,shares=None,days=None):
        cls.num_of_shares = shares 
        cls.trading_days = days 
        print(f'Number of shares per contract changed to {cls.num_of_shares} shares\n')
        print(f'Number of trading days changed to {cls.trading_days} shares\n')
            

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

        Returns: 
            rwd: The reward value from the trade
        '''
        wt = self._wealth_of_trade(pt, n)
        rwd = wt - (self.kappa*0.5)*(wt**2) 
        return rwd 

    def take_action(self,ttm,nt):
        '''
        Takes the next action according to the policy

        Parameters: 
            ttm: Time remaining to option's maturity 
            nt: Number of shares held at time 't'
        '''
        return -100 * round(self.delta(ttm)) - nt    
    
    def reset(self, path):
        '''
        Resets the environment in order to start a new episode for the simulation

        Parameters:
            path: The path the agent is following 

        Returns: 
            self.state: The state vector of the agent
        '''
        # repeatedly go through available simulated paths (if needed)
        self.t = 0
        self.path = path
        ttm = self.days_to_expiry[0]-1

        n = self.num_of_shares #no of shares
        
        #K : strike price (to be given)

        price =  round(self.sim_prices[self.path,self.t])
        self.nt = n #Number of shares at time 't'
        price_ttm = round(self.sim_prices[self.path,ttm])
        
        self.state = [price, ttm, self.nt, price_ttm]

        return self.state
    
    def delta(self, ttm):
        #Returns the option delta 
        delta = self.option_delta_path[self.path, ttm]
        return delta

    def step(self,action):
        '''
        Step function to allow the agent to transition into the next state of the episode 

        Parameters: 
            action: The action the agent takes

        Returns: 
            self.state: The state vector of the agent 
            R: The reward value 
            done: Boolean value of whether the episode is over or not
        '''
        self.t = self.t + 1 
        price =  round(self.sim_prices[self.path,self.t],2)
        self.nt = self.nt + action
        ttm = self.days_to_expiry[self.t] 
        price_ttm = round(self.sim_prices[self.path,ttm],2)
        
        reward = round(self.reward(price, self.nt)) 
        self.state = [price, ttm, self.nt, price_ttm]
        
        if ttm == 0:
            done = True
        else:
            done = False
        
        return self.state, reward, done


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    env = TradingEnv(num_simulations=100,num_contracts=5,multiplier=1.0,
    tick_size=0.1,kappa=0.1)


