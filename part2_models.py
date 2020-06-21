# Reference
# 1. https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/dqn-cartpole-9.6.1.py
# 2. City University, INM707 Deep Learning 3: Optimization, Workshop 6\
# 3. https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
# 4. https://github.com/pythonlessons/Reinforcement_Learning/tree/master/03_CartPole-reinforcement-learning_Dueling_DDQN
# 5. https://github.com/pythonlessons/Reinforcement_Learning/tree/master/02_CartPole-reinforcement-learning_DDQN
# 6  https://github.com/gsurma/cartpole
# 7. https://github.com/lazyprogrammer/machine_learning_examples


import gym
import numpy as np
from collections import deque
import random

from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input,RepeatVector, Lambda,Flatten,Subtract, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class DDQN_Agent:
	"""This agent runs both DQN and DDQN depending on the model
		param fed in"""
	def __init__(self, env, model='ddqn'):

		self.env = env
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		self.max_steps = 250
		self.memory = deque(maxlen=1500)

		self.gamma = 0.95
		self.epsilon = 1.0
		self.batch_size = 64
		self.training_iter = 100
		self.learning_rate = 0.001
		self.epsilon_min = 0.01
		self.epsilon_decay = self.epsilon_min / self.epsilon
		self.epsilon_decay = self.epsilon_decay ** (1. / float(2000))

		self.round_reward = 0
		self.all_rewards = []
		self.all_iterations = []
		self.total_run = 0
		self.step_count = []

		self.model_type = model
		self.model = self.create_nn()
		self.target_model = self.create_nn()

	def create_nn(self):

		"""creates neural network for online or target model"""

		model = Sequential()
		model.add(Dense(32, input_dim=self.state_size, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def train(self,env, iter_n=2000):

		"""Trains the agent with stopping criteria 195 score
		or higher for 10 epochs"""

		for i in range(iter_n):
			if i > 50:
				if all(reward > 195 for reward in self.step_count[-10:]):
					print('solved at episode {}'.format(i))
					break
			state = self.env.reset()
			state = np.reshape(state, [1, self.state_size])

			episode_complete = False
			step = 0
			while not episode_complete and (step < self.max_steps):
				action = self.define_action(state)
				new_state, reward, episode_complete, info = env.step(action)
				new_state = np.reshape(new_state, [1, self.state_size])

				self.memory.append((state, action, reward, new_state, episode_complete))
				self.round_reward += reward
				state = new_state
				step += 1
				if episode_complete:
					self.round_reward += -10
					self.update_target_model()
					self.print_results(i, iter_n, step)
					if i != 0:  # Update totals in memory if not the first run
						self.update_totals(i, step)
				if len(self.memory) > self.training_iter:
					self.replay()
			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay

		return self.all_iterations, self.all_rewards, self.step_count


	def print_results(self, i, iter_n, step):
		print('episode {} of {}, total steps: {}, epsilon: {}'.format(i, iter_n, step, self.epsilon))

	def update_totals(self, i, no_steps):
		"""Updates totals for that iteration in agents memory"""
		self.all_rewards.append(self.round_reward)
		self.round_reward = 0
		self.all_iterations.append(i)
		self.step_count.append(no_steps)

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def define_action(self, state):
		if np.random.random() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			return np.argmax(self.model.predict(state))

	def split_memories(self, memory_sample):
		"""Split memories of the agent to allow prediction of all states
		rather than iterating and predicting each one"""

		mems_dict = {'action_lst': [memory_sample[i][1] for i in range(self.batch_size)],
					 'rewards_lst': [memory_sample[i][2] for i in range(self.batch_size)],
					 'episode_complete_lst': [memory_sample[i][4] for i in range(self.batch_size)],
					 'state': np.zeros((self.batch_size, self.state_size)),
					 'new_state': np.zeros((self.batch_size, self.state_size))}

		for i in range(self.batch_size):
			mems_dict['state'][i] = memory_sample[i][0]
			mems_dict['new_state'][i] = memory_sample[i][3]

		return mems_dict

	def replay(self):
		# Reference 3/4/5
		memory_sample = random.sample(self.memory, self.batch_size)
		memory_sample_sz = len(memory_sample)

		memories_dictionary = self.split_memories(memory_sample)

		target_pred = self.model.predict(memories_dictionary['state'])
		model_new = self.model.predict(memories_dictionary['new_state'])
		target_new = self.target_model.predict(memories_dictionary['new_state'])

		for i in range(memory_sample_sz):
			if not memories_dictionary['episode_complete_lst'][i]:
				if self.model_type == 'ddqn':
					target_pred[i][memories_dictionary['action_lst'][i]] = memories_dictionary['rewards_lst'][i] + self.gamma * (target_new[i][np.argmax(model_new[i])])
				else:
					target_pred[i][memories_dictionary['action_lst'][i]] = memories_dictionary['rewards_lst'][i] + self.gamma * (np.amax(model_new[i]))
			else:
				target_pred[i][memories_dictionary['action_lst'][i]] = memories_dictionary['rewards_lst'][i]

		self.model.fit(memories_dictionary['state'], target_pred, batch_size=self.batch_size, verbose=0)


class DDDQN_Agent:
	"""This agent runs the DDDQN model"""
	def __init__(self, env, episodes=2000):

		self.episodes = episodes
		self.batch_size = 64
		self.memory = []
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.1
		self.epsilon_decay = self.epsilon_min / self.epsilon
		self.epsilon_decay = self.epsilon_decay ** (1. / float(episodes))
		self.state_size = env.observation_space.shape[0]
		self.action_size = env.action_space.n
		self.action_space = env.action_space


		self.q_model = self.build_model(self.state_size, self.action_size, trainable=True)
		self.target_q_model = self.build_model(self.state_size, self.action_size, trainable=False)

		self.update_weights()
		self.replay_counter = 0

	def train(self, env):
		"""Trains the agent with stopping criteria 195 score
		or higher for 10 epochs"""

		min_average_reward_for_stopping = 195
		consecutive_successful_episodes_to_stop = 10
		last_10_rewards = deque(maxlen=consecutive_successful_episodes_to_stop)

		num_Episodes = []
		Episode_Rewards = []

		for episode in range(self.episodes):
			state = env.reset()
			state = np.reshape(state, [1, self.state_size])
			done = False
			total_reward = 0

			while not done:
				action = self.act(state)
				next_state, reward, done, _ = env.step(action)
				next_state = np.reshape(next_state, [1, self.state_size])
				self.remember(state, action, reward, next_state, done)
				state = next_state
				total_reward += reward

			num_Episodes.append(episode)
			Episode_Rewards.append(total_reward)
			last_10_rewards.append(total_reward)
			last_10_avg_reward = np.mean(last_10_rewards)
			print(episode, last_10_avg_reward)

			# call experience relay
			if len(self.memory) >= self.batch_size:
				self.replay(self.batch_size)
			# Stopping criteria
			if len(
					last_10_rewards) == consecutive_successful_episodes_to_stop \
					and last_10_avg_reward > min_average_reward_for_stopping:
				print("Solved after {} epsiodes".format(episode))
				break


	def build_model(self, n_inputs, n_outputs, trainable=True):
		"""creates neural network for online or target model"""

		# common layers
		comm_input = Input(shape=(n_inputs,))
		X = Dense(32, activation='relu', name="val0", trainable=trainable)(comm_input)
		X = Dense(64, activation='relu', name="vaaal3", trainable=trainable)(X)
		X = Dense(64, activation='relu', name="valdd3", trainable=trainable)(X)

		# value network
		val_head = Dense(32, activation='relu', name="val3", trainable=trainable)(X)
		val_head = Dense(1, activation='linear', name="val4", trainable=trainable)(val_head)
		val_head = RepeatVector(n_outputs)(val_head)
		val_head = Flatten(name='meanActivation')(val_head)

		# advantage network
		adv_head = Dense(32, activation='tanh', name="val2", trainable=trainable)(X)
		adv_head = Dense(n_outputs, activation='linear', name='Activation', trainable=trainable)(adv_head)

		m_adv_head = Lambda(lambda layer: layer - K.mean(layer))(adv_head)
		# adv_head= Subtract()([adv_head,m_adv_head])

		# Merge both
		q_values = Add(name="Q-value")([val_head, adv_head])
		model = Model(inputs=[comm_input], outputs=q_values)
		model.compile(loss='mse', optimizer=Adam(lr=0.001))
		model.summary()
		return model

	def update_weights(self):
		self.target_q_model.set_weights(self.q_model.get_weights())

	def act(self, state):
		if np.random.rand() < self.epsilon:
			return self.action_space.sample()
		q_values = self.q_model.predict(state)
		action = np.argmax(q_values[0])
		return action

	def remember(self, state, action, reward, next_state, done):
		item = (state, action, reward, next_state, done)
		self.memory.append(item)

	def get_target_q_value(self, next_state, reward):
		q_value = np.amax(self.target_q_model.predict(next_state)[0])
		q_value *= self.gamma
		q_value += reward
		return q_value

	def replay(self, batch_size):

		reply_batch_batch = random.sample(self.memory, batch_size)
		state_batch, q_values_batch = [], []

		for state, action, reward, next_state, done in reply_batch_batch:
			# Predict best action using Q-network
			q_values = self.q_model.predict(state)
			# Gets Q valuesing using target network
			q_value = self.get_target_q_value(next_state, reward)
			# correction on the Q value for the action used
			q_values[0][action] = reward if done else q_value
			# collect batch state-q_value mapping
			state_batch.append(state[0])
			q_values_batch.append(q_values[0])

		# train the Q-network
		self.q_model.fit(np.array(state_batch),
						 np.array(q_values_batch),
						 batch_size=batch_size,
						 epochs=1,
						 verbose=0)
		# update exploration-exploitation probability
		self.update_epsilon()
		# copy new params on old target after
		# every 10 training updates
		if self.replay_counter % 10 == 0:
			self.update_weights()
		self.replay_counter += 1

	def update_epsilon(self):
		"""decrease the exploration, increase exploitation"""
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def get_target_q_value(self, next_state, reward):
		action = np.argmax(self.q_model.predict(next_state)[0])
		q_value = self.target_q_model.predict(next_state)[0][action]
		q_value *= self.gamma
		q_value += reward
		return q_value

def run_DDQN(model='dqn'):
	env_name = 'CartPole-v0'
	env = gym.make(env_name)
	agent = DDQN_Agent(env, model=model)
	all_iterations, all_rewards, step_count = agent.train(env)
	plot_reward(all_iterations, all_rewards)

def run_DDDQN():
	env = gym.make('CartPole-v0')
	agent = DDDQN_Agent(env)
	all_iterations, all_rewards, step_count = agent.train(env)
	# plot_reward(all_iterations, all_rewards)


if __name__ == "__main__":
	run_DDQN()
	# run_DDDQN()

