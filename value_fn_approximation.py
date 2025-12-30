import numpy as np
np.random.seed(42)
def draw_from_deck(black: bool = False) -> int:
    """
    Args:
    black: if True the function will return a positive valued card (Start of game)
    Returns: a sampled card from the deck
    """
    
    card = np.random.choice(10) + 1
    return card if (np.random.choice(2, p= [1/3, 2/3]) or black) else -card

def step(s: list, a :str = 'hit'): 
    """
    Args:
    s: State (list like), dealer's first card 1-10  and the player's sum 1-21
    a: Action, hit or stick

    Returns: a sample of the next state s' (which may be terminal if the game is finished) and reward r
    """
    player_sum = s[1]
    dealer_sum = s[0]

    if a == 'stick':
        while dealer_sum < 17:
            dealer_sum += draw_from_deck()
            if dealer_sum < 1 or dealer_sum > 21:
                return [dealer_sum, player_sum], 1
        
        if dealer_sum > player_sum:
            return [dealer_sum, player_sum], -1
        elif dealer_sum == player_sum:
            return [dealer_sum, player_sum], 0
        else:
            return [dealer_sum, player_sum], 1
    else:
        player_sum += draw_from_deck()
        if player_sum < 1 or player_sum > 21:
            return [dealer_sum, player_sum], -1
        else:
            return [dealer_sum, player_sum], None
        
def feature_match(state: list, actions: int) -> np.ndarray:
    """
    Args:
    state: State (list like), dealer's first card 1-10  and the player's sum 1-21
    actions: Action, 0 for stick and 1 for hit

    Returns: feature vector (numpy array) of shape (36,)
    3 dealer bins: 1-4, 4-7, 7-10
    6 player bins: 1-6, 4-9, 7-12, 10-15, 13-18, 16-21
    2 actions: stick, hit
    """
    dealer = np.zeros(3, dtype=np.float32)
    if 1 <= state[0] <= 4:
        dealer[0] = 1
    if 4 <= state[0] <= 7:
        dealer[1] = 1
    if 7 <= state[0] <= 10:
        dealer[2] = 1

    player = np.zeros(6, dtype=np.float32)
    if 1 <= state[1] <= 6:
        player[0] = 1
    if 4 <= state[1] <= 9:
        player[1] = 1
    if 7 <= state[1] <= 12:
        player[2] = 1
    if 10 <= state[1] <= 15:
        player[3] = 1
    if 13 <= state[1] <= 18:
        player[4] = 1
    if 16 <= state[1] <= 21:
        player[5] = 1
    
    action = np.zeros(2, dtype=np.float32)
    action[actions] = 1

    return (dealer[:, None, None] * player[None, :, None] * action[None, None, :]).flatten()


# Weights
weights = np.zeros(shape = 3 * 6 * 2)

# Policy
policy = ['greedy', 'random']

# Exploration
e_t = 0.05

# Step size
step_size = 0.01

# Move dictionary
move_dictionary = {
    'stick' : 0,
    'hit' : 1
}

# Episodes
num_episodes = 100000

# Load q-table
monte_carlo_q_table = np.load('monte_carlo_q_table.npy')
mse_list = []
l0_list = []
l1_list = []

for l in np.linspace(0, 1.0, num = 11):
    for i in range(num_episodes):
        # Initial state
        state = [draw_from_deck(True), draw_from_deck(True)] # Initial card draws for dealer (s[0]) and player (s[1])
        reward = None

        eligibility_table = np.zeros_like(weights)

        # Initial action
        action = np.random.choice(policy, p= [1-e_t, e_t])
        
        if action == 'random':
            first_move = np.random.choice(['hit', 'stick'])
        else:
            if feature_match(state, 0).T @ weights >= feature_match(state, 1).T @ weights:
                first_move = 'stick'
            else: 
                first_move = 'hit'

        # Episode
        while reward == None:
            # Take action, observe R, S' 
            next_state, reward = step(state, first_move)
            if reward == None:
                # Choose A' from S' using ε-greedy policy
                action = np.random.choice(policy, p= [1-e_t, e_t])

                if action == 'random':
                    next_move = np.random.choice(['hit', 'stick'])
                else:
                    if feature_match(next_state, 0).T @ weights >= feature_match(next_state, 1).T @ weights:
                        next_move = 'stick'
                    else: 
                        next_move = 'hit'
                
                # Calculate δ
                delta = 0 + 1 * feature_match(next_state, move_dictionary[next_move]).T @ weights - feature_match(state, move_dictionary[first_move]).T @ weights
                
                # Update eligibility trace
                eligibility_table = 1 * l * eligibility_table + feature_match(state, move_dictionary[first_move])
                
                # Weight updates:
                weights += step_size * delta * eligibility_table

                state = next_state
                first_move = next_move
            else: # Final State
                delta = reward + 0 - feature_match(state, move_dictionary[first_move]).T @ weights

                # Update eligibility trace:
                eligibility_table = 1 * l * eligibility_table + feature_match(state, move_dictionary[first_move])
                
                # Weight updates:
                weights += step_size * delta * eligibility_table
        
        # Calculate learning progress for l= 0 and l=1
        if l == 0.0 and i% 1000 ==0 :
            q_table = np.zeros(shape = (10,21,2), dtype= float)
            for dealer in range(1,11):
                for player in range(1,22):
                    for actions in range(2):
                        q_table[dealer - 1, player - 1, actions] = feature_match([dealer, player], actions).T @ weights
            l0_list.append(np.mean((monte_carlo_q_table - q_table) ** 2))
        if l == 1.0 and i% 1000 ==0 :
            q_table = np.zeros(shape = (10,21,2), dtype= float)
            for dealer in range(1,11):
                for player in range(1,22):
                    for actions in range(2):
                        q_table[dealer - 1, player - 1, actions] = feature_match([dealer, player], actions).T @ weights
            l1_list.append(np.mean((monte_carlo_q_table - q_table) ** 2))
                
    # Caclulate q table from weights
    q_table = np.zeros(shape = (10,21,2), dtype= float)
    for dealer in range(1,11):
        for player in range(1,22):
            for actions in range(2):
                q_table[dealer - 1, player - 1, actions] = feature_match([dealer, player], actions).T @ weights
                
    mse_list.append(np.mean((monte_carlo_q_table - q_table) ** 2))


# Plot the MSE vs lambda
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
fig.suptitle('MSE between SARSA(λ) and Monte Carlo Q-Value Estimates')

axs[0].plot(np.linspace(0, 1.0, num = 11), mse_list, marker='o')
axs[0].set_xlabel('λ (Lambda)')
axs[0].set_ylabel('MSE')
axs[0].set_title('MSE vs λ')
axs[0].grid()

axs[1].plot(range(len(l0_list)), l0_list, marker='o', color='orange')
axs[1].set_xlabel('Episodes')
axs[1].set_ylabel('MSE')
axs[1].set_title('MSE over Episodes for λ=0')
axs[1].grid()

axs[2].plot(range(len(l1_list)), l1_list, marker='o', color='green')
axs[2].set_xlabel('Episodes')
axs[2].set_ylabel('MSE')
axs[2].set_title('MSE over Episodes for λ=1')
axs[2].grid()

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.show()