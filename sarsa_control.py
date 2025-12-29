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
        
# N
N_0 = 10000
num_episodes = 10000
# Policy
policy = ['greedy', 'random']

move_dictionary = {
    'stick' : 0,
    'hit' : 1
}

# Load q-table
monte_carlo_q_table = np.load('monte_carlo_q_table.npy')
mse_list = []
l0_list = []
l1_list = []

for l in np.linspace(0, 1.0, num = 11):
    n_table_state = np.zeros(shape= (10,21), dtype = int)
    n_table_action = np.zeros(shape= (10,21,2), dtype = int)

    # q table
    q_table = np.zeros(shape = (10,21,2), dtype= float) # 0 for staying, 1 for hitting


    for i in range(num_episodes):
        # Initial state
        state = [draw_from_deck(True), draw_from_deck(True)] # Initial card draws for dealer (s[0]) and player (s[1])
        reward = None

        eligibility_table = np.zeros_like(q_table)

        n_table_state[state[0] - 1,state[1] - 1] += 1

        # Initial action
        e_t = N_0 / (N_0 + n_table_state[state[0] - 1,state[1] - 1])
        action = np.random.choice(policy, p= [1-e_t, e_t])
        if action == 'random':
            first_move = np.random.choice(['hit', 'stick'])
        else:
            if q_table[state[0] - 1,state[1] - 1][0] >= q_table[state[0] - 1,state[1] - 1][1]:
                first_move = 'stick'
            else: 
                first_move = 'hit'
        
        n_table_action[state[0] - 1,state[1] - 1][move_dictionary[first_move]] += 1

        # Episode
        while reward == None:
            # Take action, observe R, S' 
            next_state, reward = step(state, first_move)
            if reward == None:
                n_table_state[next_state[0] - 1,next_state[1] - 1] += 1

                # Choose A' from S' using ε-greedy policy
                e_t = N_0 / (N_0 + n_table_state[next_state[0] - 1,next_state[1] - 1])

                action = np.random.choice(policy, p= [1-e_t, e_t])

                if action == 'random':
                    next_move = np.random.choice(['hit', 'stick'])
                else:
                    if q_table[next_state[0] - 1,next_state[1] - 1][0] >= q_table[next_state[0] - 1,next_state[1] - 1][1]:
                        next_move = 'stick'
                    else: 
                        next_move = 'hit'
                
                n_table_action[next_state[0] - 1,next_state[1] - 1][move_dictionary[next_move]] += 1
                # Calculate δ
                delta = 0 + 1 * q_table[next_state[0] - 1,next_state[1] - 1][move_dictionary[next_move]] - q_table[state[0] - 1,state[1] - 1][move_dictionary[first_move]]
                
                eligibility_table[state[0] - 1,state[1] - 1][move_dictionary[first_move]] += 1
                
                # State Action Function updates (vectorized):
                alpha = np.zeros_like(q_table, dtype=float)
                np.divide(1.0, n_table_action, out=alpha, where=n_table_action>0)
                q_table += alpha * delta * eligibility_table
                eligibility_table *= 1 * l


                state = next_state
                first_move = next_move
            else: # Final State
                delta = reward + 0 - q_table[state[0] - 1,state[1] - 1][move_dictionary[first_move]]

                eligibility_table[state[0] - 1,state[1] - 1][move_dictionary[first_move]] += 1
                
                # State Action Function updates (vectorized):
                alpha = np.zeros_like(q_table, dtype=float)
                np.divide(1.0, n_table_action, out=alpha, where=n_table_action>0)
                q_table += alpha * delta * eligibility_table
                eligibility_table *= 1 * l
        if l == 0.0:
            l0_list.append(np.mean((monte_carlo_q_table - q_table) ** 2))
        elif l == 1.0:
            l1_list.append(np.mean((monte_carlo_q_table - q_table) ** 2))
    # Calculate the mean squared error between the monte carlo q-table and the current q-table
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

axs[1].plot(range(num_episodes), l0_list, marker='o', color='orange')
axs[1].set_xlabel('Episodes')
axs[1].set_ylabel('MSE')
axs[1].set_title('MSE over Episodes for λ=0')
axs[1].grid()

axs[2].plot(range(num_episodes), l1_list, marker='o', color='green')
axs[2].set_xlabel('Episodes')
axs[2].set_ylabel('MSE')
axs[2].set_title('MSE over Episodes for λ=1')
axs[2].grid()

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.show()