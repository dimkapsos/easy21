import numpy as np
import matplotlib.pyplot as plt
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
        
# Setup parameters
# N
N_0 = 10000
n_table_state = np.zeros(shape= (10,21), dtype = int)
n_table_action = np.zeros(shape= (10,21,2), dtype = int)

# q table
q_table = np.zeros(shape = (10,21,2), dtype= float) # 0 for staying, 1 for hitting

# Policy
policy = ['greedy', 'random']

num_episodes = 1000000
for i in range(num_episodes):
    state_zero = [draw_from_deck(True), draw_from_deck(True)] #Initial card draws for dealer (s[0]) and player (s[1])

    moves = [state_zero]
    reward = None

    # Episode
    while reward == None:
        # Choose move
        e_t = N_0 / (N_0 + n_table_state[moves[-1][0] - 1, moves[-1][1] - 1])

        action = np.random.choice(policy, p= [1-e_t, e_t])

        if action == 'random':
            next_move = np.random.choice(['hit', 'stick'])
        else:
            if q_table[moves[-1][0] - 1, moves[-1][1] - 1][0] >= q_table[moves[-1][0] - 1, moves[-1][1] - 1][1]:
                next_move = 'stick'
            else: 
                next_move = 'hit'

        # Move
        state, reward = step(moves[-1], next_move)
        moves.append(state)
    
    for i in range(len(moves) - 1):
        n_table_state[moves[i][0] - 1, moves[i][1] - 1] += 1    # Doesn't matter if we hit or stayed

        if moves[i][1] == moves[i+1][1]:  # We stayed
            n_table_action[moves[i][0] - 1, moves[i][1] - 1][0] += 1

            q_table[moves[i][0] - 1, moves[i][1] - 1][0] += (1 /n_table_action[moves[i][0] - 1, moves[i][1] - 1][0]) * (reward - q_table[moves[i][0] - 1, moves[i][1] - 1][0])
        else:   # We hit
            n_table_action[moves[i][0] - 1, moves[i][1] - 1][1] += 1

            q_table[moves[i][0] - 1, moves[i][1] - 1][1] += (1 /n_table_action[moves[i][0] - 1, moves[i][1] - 1][1]) * (reward - q_table[moves[i][0] - 1, moves[i][1] - 1][1])


# Create a value table, storing the best action value for each state
value_table = np.zeros(shape= (10,21), dtype= float)
policy_table = np.empty(shape= (10,21), dtype= object)
for i in range(10):
    for j in range(21):
        value_table[i,j] = max(q_table[i,j][0], q_table[i,j][1])
        policy_table[i,j] = 'stay' if q_table[i,j][0] > q_table[i,j][1] else 'hit'




# Plot the optimal value function as a 3d grid plot with black and white colormap
fig = plt.figure()
fig.set_size_inches(14, 10)
ax = fig.add_subplot(111, projection='3d')
X = np.arange(1,11)
Y = np.arange(1,22)
X, Y = np.meshgrid(X, Y)
# Black and white colormap
ax.plot_surface(X, Y, value_table.T, cmap='plasma')
ax.set_xlabel("Dealer's showing card")
ax.set_ylabel("Player's sum")



# Set x ticks and y ticks
ax.set_xticks(np.arange(1, 11, 1))
ax.set_yticks(np.arange(1, 22, 1))

# Title 
ax.set_title("Optimal Value Function for Simplified Blackjack")

plt.show()


# Interpretation of results
dealer_card = 10
your_card = 15
value_table[dealer_card-1, your_card-1] # Your expected return starting from this state
policy_table[dealer_card-1, your_card-1] # Your optimal action from this state
print(f"From state (dealer's card: {dealer_card}, your card: {your_card}), your expected return is {value_table[dealer_card-1, your_card-1]:.2f} and your optimal action is to {policy_table[dealer_card-1, your_card-1]}.")

# Save q-table
np.save('monte_carlo_q_table.npy', q_table)