import random 
from tqdm import tqdm 
import numpy as np 
from collections import deque, defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam


class DQN:

    def __init__(self,state_size,action_space,load_model=bool):
        '''
        Initializes DQN for training the RL agent 

        Parameters: 
            state: The state vector dimension of the environment 
            action: The action set dimension taken by the agent 
            load_model: Whether to load a previously trained model
        ''' 

         #Dimension of the state and action vector
        self.state_size = state_size #Size of the state vectpr 
        self.action_space = action_space #The action space (env.Discrete object)
        self.action_size = action_space.n #Size of the action space
        self.load_model = load_model

        #Setting up the hyperparameters 
        self.discount_factor = 0.9
        self.step_size = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 100
        self.train_start = 1000

        #Memory size for experience replay 
        self.memory_replay = deque(maxlen=2000)
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
        model.add(Dense(8, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(8,activation='relu'))
        model.add(BatchNormalization())
        # model.add(Dense(16,activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dense(16,activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dense(16,activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dense(16,activation='relu'))
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
            action = self.action_space.sample()
        else:
            #Else go for the greedy off-policy action selection
            q_value = self.prediction_model.predict(state)
            action = np.argmax(q_value[0])
        return action

    def update_replay_memory(self, state, action, reward, next_state, done):
        #Updates the memory replay with the state_vector 
        state_vector = (state,action,reward,next_state,done)
        if len(state_vector) != 5:
            raise ValueError('The size of the state vector must be 5')
        self.memory_replay.append(state_vector)

        #Perform epsilon-decay 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_agent(self):
        #Skip if the length of memory replay is less than the train_start condition
        if len(self.memory_replay) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory_replay))
        #Sampling a random minibatch of transition state 
        mini_batch = random.sample(self.memory_replay, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0] #S 
            self.transition['Action'].append(mini_batch[i][1]) #A
            self.transition['Reward'].append(mini_batch[i][2]) #R
            update_target[i] = mini_batch[i][3] #S'
            self.transition['Done'].append(mini_batch[i][4]) #Done 

        target = self.prediction_model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            #Q learning: Use off-policy to find the max Q-value at S' from target model
            if self.transition['Done'][i]:
                target[i][self.transition['Action'][i]] = self.transition['Reward'][i]
            else:
                target[i][self.transition['Action'][i]] = self.transition['Reward'][i] + self.discount_factor * np.amax(target_val[i])

        #Fit the model
        self.prediction_model.fit(update_input, target, batch_size=self.batch_size, epochs=1,verbose=0)


    


