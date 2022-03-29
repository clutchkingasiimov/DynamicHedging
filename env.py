import gym 
from gym import spaces 
import numpy as np 
from simulations import OptionSimulation

class TradingEnv(gym.Env,OptionSimulation):
    trading_days = 252 #Number of trading days in one year 
    num_of_shares = 100 #Vanilla options contract size
    
    def __init__(self, init_price, sample_size, num_contracts=int, multiplier=float,
    tick_size=float, kappa=float):
        super().__init__(init_price, sample_size)
        """
        Parameters:
        init_price: The starting price point of the simulation
        sample_size: The number of GBM and BS simulations to run for the agent to train on
        num_contracts: The number of contracts the agent will hold.
        multiplier: Float value required for the intensity of the bid-offer spread 
        tick_size: Used for computing the cost relative to the midpoint of the bid-offer spread
        kappa: The risk factor of the portfolio
        """

        # os = OptionSimulation(100,self.total_episodes) 
        self.num_contracts = num_contracts 
        self.multiplier = multiplier 
        self.tick_size = tick_size
        self.kappa = kappa

         #Action space (Discrete)
        self.num_actions = self.num_contracts*self.num_of_shares #Number of actions 
        self.action_range = (self.num_actions * 2)+1 
        self.state_space = 3 
        self.action_space = spaces.Discrete(self.action_range,start=-self.num_actions) #Discrete action space   

    @classmethod
    def change_base_params(cls,shares=None,days=None):
        cls.num_of_shares = shares 
        cls.trading_days = days 
        print(f'Number of shares per contract changed to {cls.num_of_shares} shares\n')
        print(f'Number of trading days changed to {cls.trading_days} shares')


    def set_params(self):
        '''
        Method to be invoked before starting the training in order to set the parameters

        '''
        self.sim_prices = super().GBM(50,0.01,time_increment=1)
        self.days_to_expiry_normalized = self.ttm/self.trading_days #Only to be used for the calculation of BS call price
        self.days_to_expiry = self.ttm #Creates an array of days left to expiry 
        self.option_price_path, self.option_delta_path = super().BS_call(self.days_to_expiry_normalized,self.sim_prices,100,0.01,0,0)
        
        #Track the index of simulated path is use 
        self.sim_episode = -1
        #Track time step within an episode
        self.t = None 

        if self.num_contracts > 10:
            raise ValueError("The maximum number of contracts in the simulation cannot be more than 10.")


    def cost_of_trade(self,n):
        #n: Number of shares 
        cost = self.multiplier * self.tick_size * (np.abs(n) + 0.01*n*n)
        return cost 

    def wealth_of_trade(self,pt,n):
        #W_{t} = q_{t} - c_{t} (pt: Price of the stock at time 't')
        ct = self.cost_of_trade(n)
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
        wt = self.wealth_of_trade(pt, n)
        rwd = wt - (self.kappa*0.5)*(wt**2) 
        return rwd 

    def take_action(self,t,nt):
        '''
        Takes the next action according to the policy

        Parameters: 
            t: Time index 't'
            nt: Number of shares held at time 't'
        '''
        return (-100 * round(self.delta(t),1)) - nt
    
    def reset(self):
        '''
        Resets the environment in order to start a new episode for the simulation

        Parameters:
            path: The path the agent is following 

        Returns: 
            self.state: The state vector of the agent
        '''
        # repeatedly go through available simulated paths (if needed)
        self.t = 0
        self.option_t = 0
        self.path = (self.sim_episode+1) % self.sample_size
        ttm = self.days_to_expiry[0]

        price = round(self.sim_prices[self.path,self.t])
        self.nt = self.num_of_shares #Number of shares at time 't'
        # price_ttm = round(self.sim_prices[self.path,ttm])
        self.state = [price, ttm, self.nt]

        return self.state
    
    def delta(self, t):
        '''
        Computes the option delta at time 't'

        Parameters: 
            t: The time index 't'
        '''
        #Returns the option delta 
        delta = self.option_delta_path[self.path, t]
        # delta = self.option_price_path[self.path, 49] - self.option_price_path[self.path, t]
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
        self.t += 1 
        price =  round(self.sim_prices[self.path,self.t],2)
        self.nt += action
        ttm = self.days_to_expiry[self.t] 
        # price_ttm = round(self.sim_prices[self.path,ttm],2)
        
        reward = round(self.reward(price, self.nt)) 
        self.state = [price, ttm, self.nt]

        #If tomorrow is the end of episode
        if ttm == 0:
            done = True
        else:
            done = False

        return self.state, reward, done

          # if ttm == 1 & self.path == (self.num_simulations-1):
        #     done = 1 #1 = True 
        # elif ttm == 1:
        #     episode = self.path + 1 
        #     self.reset(episode)
        #     done = 0 #0 = False 
        # else:
        #     done = 0

    def option_pnl(self,action):
        '''
        Computes the option PnL

        Parameters: 
            action: The action taken by the agent (The number of shares to hold)
        '''
        self.option_t += 1
        option_price = round(self.option_price_path[self.path,self.option_t],2)
        profit_option = action * option_price
        return profit_option
         


