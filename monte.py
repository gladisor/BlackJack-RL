import numpy as np

def make_decks():
	card_types = [1,2,3,4,5,6,7,8,9,10,10,10,10]
	new_deck = []
	for j in range(4):
		new_deck.extend(card_types)
	np.random.shuffle(new_deck)
	return new_deck

def get_ace_values(temp_list):
    sum_array = np.zeros((2**len(temp_list), len(temp_list)))
    for i in range(len(temp_list)):
        n = len(temp_list) - i
        half_len = int(2**n * 0.5)
        for rep in range(int(sum_array.shape[0]/half_len/2)):
            sum_array[rep*2**n : rep*2**n+half_len, i]=1
            sum_array[rep*2**n+half_len : rep*2**n+half_len*2, i]=11
    return list(set([int(s) for s in np.sum(sum_array, axis=1) if s<=21]))

def ace_values(num_aces):
    temp_list = []
    for i in range(num_aces):
        temp_list.append([1,11])
    return get_ace_values(temp_list)

def count_aces(hand):
	aces = 0
	total = 0
	for card in hand:
		if card != 1:
			total += card
		else:
			aces += 1
	return aces, total

def total_up(hand):
	aces, total = count_aces(hand)
	ace_value_list = ace_values(aces)
	final_totals = [i+total for i in ace_value_list if i+total<=21]
	if final_totals == []:
		return min(ace_value_list) + total
	else:
		return max(final_totals)

def deal(deck):
	return [deck.pop() for i in range(2)]

def hit(hand, deck):
	hand.append(deck.pop())

def usable_ace(hand):
	if count_aces(hand)[0] > 0:
		return 1
	else:
		return 0

def bust(hand):
	if total_up(hand) > 21:
		return True
	else:
		return False

def update_state(agent_hand, dealer_hand):
	return usable_ace(agent_hand), total_up(agent_hand), dealer_hand[0]

def generate_episode(pi, epsilon=0.4):
	p_random_a = 1 - epsilon + epsilon/2
	
	deck = make_decks()
	agent_hand = deal(deck)

	dealer_showing = 0
	dealer_hand = deal(deck)
	state = update_state(agent_hand, dealer_hand)

	states = [state]
	actions = []
	rewards = []

	done = False
	while not done:
		## Agent turn:
		while True:
			if np.random.random() > p_random_a:
				action = np.random.randint(2)
			else:
				action = pi[state[0]][state[1]][state[2]]

			actions.append(action)
			## Player action
			if action == 1:
				hit(agent_hand, deck)
				if bust(agent_hand):
					rewards.append(-1)
					done = True
					## We do not append state here because the agent will not be making
					## additional actions this turn
					## END OF EPISODE
					return states, actions, rewards
				else:
					rewards.append(0)
					state = update_state(agent_hand, dealer_hand)
					states.append(state)
			else:
				break

		## Dealer turn
		while True:
			## Dealer hits if total < 13
			if total_up(dealer_hand) < 13:
				hit(dealer_hand, deck)
				if bust(dealer_hand):
					rewards.append(1)
					done = True
					return states, actions, rewards
			else:
				if total_up(dealer_hand) < total_up(agent_hand):
					rewards.append(1)
				elif total_up(dealer_hand) == total_up(agent_hand):
					rewards.append(0)
				else:
					rewards.append(-1)
				return states, actions, rewards
	return states, actions, rewards

def train(gamma=0.5,l_r=0.1, episodes=20000):
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

import matplotlib.pyplot as plt
def plotPI_(pi_):
	ax1=plt.subplot(121)
	ax2=plt.subplot(122)

	ax1.imshow(pi_[0])
	ax1.set_title("No usable ace")
	ax1.set_xlabel("Dealer showing")
	ax2.set_ylabel("Agent showing")

	ax2.imshow(pi_[1])
	ax2.set_title("Usable ace")
	ax1.set_xlabel("Dealer showing")
	ax2.set_ylabel("Agent showing")
	plt.show()

if __name__ == "__main__":

	q_, pi_ = train()
	plotPI_(pi_)
	