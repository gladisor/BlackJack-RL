import numpy as np
from utils import *

def train(gamma=0.5,l_r=0.05, episodes=20000):
	q_table = np.ones((2,22,11,2)) ## Usable aces, player hand, dealer showing, actions
	pi = np.zeros((2, 22, 11)) ## Usable aces, player hand, dealer showing
	for episode in range(episodes):
		states, actions, rewards = generate_episode(pi)
		G = 0
		for i in range(len(states)-1, -1, -1):
			G = gamma*G + rewards[i]
			## Using update rule
			Q_St_At = q_table[states[i][0]][states[i][1]][states[i][2]][int(actions[i])]
			q_table[states[i][0]][states[i][1]][states[i][2]][int(actions[i])] = Q_St_At + l_r*(G - Q_St_At)
			## A = greedy action
			A = np.argmax(q_table[states[i][0]][states[i][1]][states[i][2]])
			pi[states[i][0]][states[i][1]][states[i][2]] = A
	return q_table, pi

if __name__ == "__main__":

	q_, pi_ = train_td()
	plotPI_(pi_)
	