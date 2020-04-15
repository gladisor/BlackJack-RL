from monte import *
import numpy as np


pi = np.zeros((2, 22, 11)) ## Usable aces, player hand, dealer showing
for episode in range(10):
	states, actions, rewards = generate_episode(pi)
	for s, a, r in zip(states, actions, rewards):
		print(s,a,r)
	print()