import numpy as np 
from random import randint
import random
import tensorflow
from collections import deque
from keras import losses
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Adamax
from keras.initializers import Zeros, Ones
from ENV_TRAIN import Retail_Environment
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from pandas import DataFrame


class DQN_Agent():

	def __init__(self, state_size, action_size, gamma, epsilon_decay, epsilon_min, learning_rate, epochs, env, batch_size, update, iteration, base_stock, factor, x):

		self.state_size = state_size
		self.action_size = action_size

		self.memory = deque(maxlen = 20000)

		self.gamma = gamma
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.env = env
		self.batch_size = batch_size
		self.update = update
		self.epoch_counter = 0
		self.epsilon = 1.0
		self.iteration = iteration

		self.model = self.build_model()
		self.target_model = self.build_model()

		self.base_stock = base_stock
		self.factor = factor

		self.x = x


	def build_model(self):

		model = Sequential()

		model.add(Dense(32, input_dim = self.state_size, activation = 'relu'))
		model.add(Dense(32, activation = 'relu'))
		model.add(Dense(self.action_size, activation = 'linear'))

		model.compile(loss = losses.mean_squared_error, optimizer = Adam(learning_rate = self.learning_rate))

		return model


	def act(self, state):

		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])


	def remember(self, state, action, reward, next_state, done):

		self.memory.append((state, action, reward, next_state, done))


	def replay(self):

		#experience replay from replay memory
		minibatch = random.sample(self.memory, self.batch_size)
		current_states = np.array([experience[0] for experience in minibatch])
		current_qs_list = np.zeros((self.batch_size, 1, self.env.max_order + 1))
		for k in range(self.batch_size):
			current_qs_list[k] = self.model.predict(current_states[k])

		new_states = np.array([experience[3] for experience in minibatch])
		future_qs_list = np.zeros((self.batch_size, 1, self.env.max_order + 1))
		for k in range(self.batch_size):
			future_qs_list[k] = self.target_model.predict(new_states[k])

		x = []
		y = []

		for i, (current_state, action, reward, next_state, done) in enumerate(minibatch):

			if not done:
				max_fut_q = np.max(future_qs_list[i])
				new_q = reward + self.gamma*max_fut_q
			else:
				new_q = reward

			current_qs = current_qs_list[i]
			current_qs[0][action] = new_q
			x.append(current_state[0])
			y.append(current_qs[0])

		self.model.fit(np.array(x), np.array(y), batch_size = self.batch_size, verbose = 0, shuffle = False)

		#decay epsilon
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		#update target network
		if self.epoch_counter % self.update == 0:
			self.update_target_model()


	def update_target_model(self):

		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i]
		self.target_model.set_weights(target_weights)
		print('***** Target network updated *****')


	def train(self):

		scores = []

		for e in range(self.epochs):
			
			done = False
			score = 0
			state, _ = self.env.reset()

			prev_val = 0 

			while not done:
				state = np.reshape(state, [1, self.state_size])
				action = self.act(state)
				next_state, reward, done, _ = self.env.step(action)
				score += reward
				next_state = np.reshape(next_state, [1, self.state_size])

				in_inv = 0
				for i in range(self.env.leadtime + self.env.lifetime - 1):
					in_inv += state[0][i] # sommo sui prodotti degli ultimi due giorni che sono entrambi il mio inventario (ho piu' unita' dello stesso prodotto)
				bsp_order = max(0, self.base_stock - in_inv) #l 'azioe che voglio e' quella per ripristinare lo stock in base stock fissato quindi devo agire ordinando la differenza
				cur_val = -self.factor * abs(bsp_order - action)
				F = cur_val - ((1/self.gamma)*prev_val)
				prev_val = cur_val
				total = reward + F
				self.remember(state, action, total, next_state, done)
				state = next_state

			avg_score = score / self.env.time 
			self.epoch_counter += 1

			print('Epoch ' + str(self.epoch_counter) + ' | Avg score per period: ' + str(-avg_score))

			if len(self.memory) > self.batch_size:
				self.replay()

			scores.append(-avg_score)

		path = r'..\\rewardshaping\\test_dqn_shapedB'
		df = DataFrame({'Reward': scores})
		# df.to_excel(str(path) + '/EVAL' + str(self.x) + '/Lifetime ' + str(self.env.lifetime) + ' - iteration ' + str(self.iteration) + '.xlsx')
		self.model.save(str(path) + str(self.x) + '\\Lifetime ' + str(self.env.lifetime) + ' - iteration ' + str(self.iteration) + '.h5')
		return scores

	def save(self, name):
		self.model.save_weights(name)

	def load(self, name):
		self.model.load_weights(name)

	def get_qs(self, state):
		return self.model.predict(np.array(state).reshape(-1, *self.state_size))[0]
