# Reference
# 1. https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/dqn-cartpole-9.6.1.py
# 2. City University, INM707 Deep Learning 3: Optimization, Workshop 6\
# 3. https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/cartpole_dqn.ipynb
# 4. https://github.com/pythonlessons/Reinforcement_Learning/tree/master/03_CartPole-reinforcement-learning_Dueling_DDQN
# 5. https://github.com/pythonlessons/Reinforcement_Learning/tree/master/02_CartPole-reinforcement-learning_DDQN
# 6  https://github.com/gsurma/cartpole
# 7. https://github.com/lazyprogrammer/machine_learning_examples

"""Contains helper function that support saving and plotting results"""

import numpy as np
import matplotlib.pyplot as plt


def plot_reward(all_iterations, all_rewards):
	""" plots the iterations vs rewards for training"""
	fig = plt.figure()
	plt.plot(all_iterations, all_rewards)
	fig.suptitle('Iterations vs rewards')
	plt.xlabel('No. Iterations')
	plt.ylabel('reward')
	plt.ioff()
	plt.show()


def plot_steps(all_iterations, step_count):
	""" plots the iterations vs rewards for training"""
	fig = plt.figure()
	plt.plot(all_iterations, step_count)
	fig.suptitle('Iterations vs step')
	plt.xlabel('No. Iterations')
	plt.ylabel('steps')
	plt.ioff()
	plt.show()


def save_results(all_iterations, all_rewards, step_count):

	data = [all_iterations, all_rewards, step_count]
	np.savetxt('/Users/matthewgalloway/Documents/RF/q_learning/policy_inv/random.csv', data, delimiter=",")



def get_mini_stations_dict():
	stations = {0: 'Baker Street',
				1: 'Warrent Street',
				2: 'Goodge Street ',
				3: 'Bond Street ',
				4: 'Oxford Circus ',
				5: 'Tottenham Court Rd.',
				6: 'Holborn',
				7: 'Chancery Lane',
				8: 'Covent Garden ',
				9: 'Green Park',
				10: 'Piccadilly Circus ',
				11: 'Leicester Square',
				}
	return stations


def get_stations_dict():
	stations = {0: 'Angel ',
				1: 'Baker Street ',
				2: 'Bank ',
				3: 'Barbican ',
				4: 'Blackfriars ',
				5: 'Bond Street ',
				6: 'Cannon Street ',
				7: 'Chancery Lane ',
				8: 'Charing Cross ',
				9: 'Covent Garden ',
				10: 'Embankment ',
				11: 'Euston ',
				12: 'Euston Square ',
				13: 'Farringdon ',
				14: 'Goodge Street ',
				15: 'Green Park ',
				16: 'Holborn ',
				17: 'Kings Cross St Pancras LU ',
				18: 'Leicester Square ',
				19: 'London Bridge ',
				20: 'Mansion House ',
				21: 'Monument ',
				22: 'Moorgate ',
				23: 'Old Street ',
				24: 'Oxford Circus ',
				25: 'Piccadilly Circus ',
				26: 'Regents Park ',
				27: 'Russell Square ',
				28: 'St James Park ',
				29: 'St Pauls ',
				30: 'Temple ',
				31: 'Tottenham Court Road ',
				32: 'Victoria ',
				33: 'Warren Street ',
				34: 'Westminster '}
	return stations
