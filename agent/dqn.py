import gym 
import random 
import numpy as np 
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam


class DQN:

    def __init__(self,state,action,load_model=bool):

        '''
        Initializes DQN for training the RL agent 

        Parameters: 
            state: The state vector of the environment 
            action: The action set taken by the agent 
            load_model: Whether to load a previously trained model
        '''
        #Dimension of the state and action vector
        self.state_size = state 
        self.action = action
        self.load_model = load_model 

        #Setting up the hyperparameters 
        self.discount_factor = 0.95
        self.step_size = 0.001
        self.epsilon = 0.20
        self.epsilon_decay = 0.0001
        self.epsilon_min = 0.01
        self.batch_size = None
        self.train_state = None 

        #Memory size for experience replay 
        self.memory_replay = deque(maxlen=1000)

        #Create the main model and the target model
        self.prediction_model = self.build_agent()
        self.target_model = self.build_agent()

        #Initialize the target model 
        self.update_target_model()

        
    #Builds the network for DQN
    def build_agent(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation='linear'))

        #Compile model 
        model.compile(loss='mse',optimizer=Adam(lr=self.step_size))
        return model

    #Update the target model after 
    def update_target_model(self):
        self.target_model.set_weights(self.prediction_model.get_weights())

    #Initialize the agent by taking an action under epsilon-greedy policy 
    def initialize_agent(self, state):
        random_action_prob = np.random.random()
        if random_action_prob <= self.epsilon:
            #Choose a random action to explore 
            action = np.random.choice(self.action)
        else:
            #Else go for the greedy off-policy action selection
            q_value = self.model.predict(state)
            action = np.argmax(q_value[0])
        return action

    def update_replay_memory(self, state, action, reward, next_state, done):
        #Updates the replay memory for training
        update_vector = (state,action,reward,next_state,done)
        self.memory_replay.append(update_vector)

        #Perform epsilon-decay 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 


if __name__ == "__main__":
    dqn = DQN()

    


