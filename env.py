import gym 
from gym import spaces 
from gym.utils import seeding 
import numpy as np 
from Simulator.simulations import OptionSimulation


class TradingEnv(gym.Env):
    os = OptionSimulation(100,50) 
    trading_days = 252 #Number of trading days in one year 
    """
    Trading Enviroment class with all the modules related 
    to performing trading under a controlled simulation environment.

    Parameters:
        num_simulations: The number of GBM and BS simulations to run for the agent to train on
        num_contracts: The number of contracts the agent will hold.
    """

    def __init__(self,num_simulations=int,num_contracts=int):

        self.num_simulations = num_simulations
        self.num_contracts = num_contracts 
        self.sim_prices = self.os.GBM(10,5,0.05,time_increment=1)

        self.days_to_expiry = self.os.ttm/self.trading_days #Creates an array of days left to expiry 

        self.option_price_path, self.option_delta_path = self.os.BS_call(self.days_to_expiry,self.sim_prices,10,0.05,0,0)

        if self.num_contracts > 10:
            raise ValueError("The maximum number of contracts in the simulation cannot be more than 10.")


