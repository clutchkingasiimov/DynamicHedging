import random 
import numpy as np 
from collections import deque, defaultdict
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam


class DQN:

    def __init__(self,state_size,action_size,load_model=bool):

        '''
        Initializes DQN for training the RL agent 

        Parameters: 
            state: The state vector dimension of the environment 
            action: The action set dimension taken by the agent 
            load_model: Whether to load a previously trained model
        '''
        #Dimension of the state and action vector
        self.state_size = state_size 
        self.action_size = action_size
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
        #Store the transition information in a defaultdict of list type
        self.transition = defaultdict(list)

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
        '''
        Initialize the agent for training under epsilon-greedy policy
        '''
        random_action_prob = np.random.random()
        if random_action_prob <= self.epsilon:
            #Choose a random action to explore 
            action = np.random.choice(self.action)
        else:
            #Else go for the greedy off-policy action selection
            q_value = self.model.predict(state)
            action = np.argmax(q_value[0])
        return action

    def update_replay_memory(self, state_vector):
        #Updates the memory replay with the state_vector 
        if len(state_vector) != 5:
            raise ValueError('The size of the state vector must be 5')
        self.memory_replay.append(state_vector)

        #Perform epsilon-decay 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_agent(self):
        batch_size = min(self.batch_size, len(self.memory_replay))
        #Sampling a random minibatch of transition state 
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            self.transition['Action'].append(mini_batch[i][1])
            self.transition['Reward'].append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            self.transition['Done'].append(mini_batch[i][4])

        target = self.prediction_model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            #Q learning: Use off-policy to find the max Q-value at S' from target model
            if self.transition['Done'][i]:
                target[i][self.transition['Action'][i]] = self.transition['Reward'][i]
            else:
                target[i][self.transition['Action'][i]] = self.transition['Reward'][i] + self.discount_factor * np.max(target_val[i])

        #Fit the model
        self.prediction_model.fit(update_input, target, batch_size=self.batch_size, epochs=1)


    


