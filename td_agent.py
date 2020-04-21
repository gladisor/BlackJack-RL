import numpy as np
from utils import *

def train_td(gamma=1, alpha=0.1, episodes=20000):
	q_table = np.ones((2,22,11,2))
	pi = np.ones((2,22,11))
	for episode in range(episodes):
		states, actions, rewards = generate_episode(pi)
		state = states[0]
		for i in range(1,len(states)):
			current_q = q_table[state[0]][state[1]][state[2]][int(actions[i])]
			next_q = q_table[states[i][0]][states[i][1]][states[i][2]][int(actions[i])]

			## Temporal difference update
			q_table[state[0]][state[1]][state[2]][int(actions[i])] = current_q + alpha*(rewards[i] + gamma*next_q - current_q)

			## Greedify
			A = np.argmax(q_table[state[0]][state[1]][state[2]])
			pi[state[0]][state[1]][state[2]] = A

			## Old state <-- new state
			state = states[i]
	return q_table, pi

if __name__ == "__main__":
	q_, pi_ = train_td()
	plotPI_(pi_)