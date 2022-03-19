from unicodedata import name
import random 
import numpy as np
from scipy.stats import norm

 
class OptionSimulation:
    seed_state = 10 #Seed state for all the simulations 
    random.seed(seed_state) #Set seed state 

    def __init__(self, init_price, sample_size):
        self.init_price = init_price  #S_0
        self.sample_size = sample_size #Number of episodes we wish to have 
        self.ttm = None #Time to maturity array that will be used for BSM pricing 
        

    def GBM(self, T, D, std, time_increment=int):
        '''
        Simulates Geometric Brownian Motion with pre-specified parameters of mu = 0.05 and dt = 0.01 

        Parameters: 
            T : Number of time-steps in one episode 
            D : Number of time frames in a given time step 
            std : The standard deviation for the simulation process 
            time_increment: The time increment for the time to expiry 

            Example: There are T*D days left to expiry, and we want to increment the days to expiry 
            under 1-day increments, then the days will be incremented as 

            [T*D, T*D-1, T*D-2,....,T*D-T*D]

        Returns: 
            A list of simulated prices from GBM of size 'self.sample_size'.

        How the periods are computed: 
            Suppose we have 10 days of trading, with each day having 5 intervals, then 
            the total number of observations will be 10*5 = 50 simulated observations per episode
        '''
        mu = 0.05 #Initialize the stochastic process with a mean of 0.05
        dt = 0.01 #Keep a drift factor to a realistic value of 0.01 
        num_period = T*D
        self.ttm = np.arange(num_period,0,-time_increment) 


        z = np.random.normal(size=(self.sample_size, num_period))

        a_price = np.zeros((self.sample_size, num_period))
        a_price[:, 0] = self.init_price

        for t in range(num_period - 1):
            a_price[:, t + 1] = a_price[:, t] * np.exp(
                (mu - (std ** 2) / 2) * dt + std * np.sqrt(dt) * z[:, t]
            )
        return a_price


    # BSM Call Option Pricing Formula & BS Delta formula
    def BS_call(self, T, S, K, sigma, r, q):

        '''
        Computes the price of an European Call Option using the Black-Scholes equation

        Parameters: 
            T : Time remaining to expiry 
            S : Underlying price 
            K : Strike price
            sigma : Volatility 
            r : Risk-free interest rate 
            q : Continuously compounded dividend rate 

        Returns: 
            bs_price: The BS computed price of the call option (num_path x num_period)
            bs_delta: The BS computed Delta of the call option (num_path x num_period)
        '''
        
        d1 = (np.log(S / K) + (r - q + sigma * sigma / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        call_delta = np.exp(-q * T) * norm.cdf(d1)
        
        return call_price, call_delta


    def BS_put(self, T, S, K, sigma, r, q):

        '''
        Computes the price of a European Put Option using the Black-Scholes equation
        
        Parameters: 
            T : Time remaining to expiry 
            S : Underlying price 
            K : Strike price
            sigma : Volatility 
            r : Risk-free interest rate 
            q : Continuously compounded dividend rate 

        Returns: 
            bs_price: The BS computed price of the put option (num_path x num_period)
            bs_delta: The BS computed Delta of the put option (num_path x num_period)
        '''
        d1 = (np.log(S / K) + (r - q + sigma * sigma / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1) 
        put_delta = np.exp(-q * T) * (norm.cdf(d1)-1)
        
        return put_price, put_delta


if __name__ == "__main__":
    # import matplotlib.pyplot as plt 
    '''
    We simulate 50 different GBM series with an initial price of 100, for a total of 100 timepoints 
    stamped under 10-tick intervals, and a time increment of 1.

    Initial parameters: 

    Sigma: 0.05 
    K: 100 
    r: 0
    q: 0

    We only use Delta as one of the computed parameters for the training phase
    '''
    optsim = OptionSimulation(init_price=100, sample_size=50)
    sim_prices = optsim.GBM(10,5,0.05,time_increment=1)
    days_to_expiry = optsim.ttm/optsim.trading_days #Get the days to expiry array 
    call_prices, call_deltas = optsim.BS_call(days_to_expiry,sim_prices,100,0.05,0,0)
    # plt.plot(call_prices[0])
    # plt.show()
    