import numpy as np
np.random.seed(42)

########################################
# Environment (unchanged)
########################################

def draw_from_deck(black=False):
    card = np.random.choice(10) + 1
    return card if (np.random.choice(2, p=[1/3, 2/3]) or black) else -card

def step(s, a):
    player_sum, dealer_sum = s[1], s[0]

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

########################################
# Feature representation
########################################

def state_features(state):
    """
    Features for critic V(s)
    3 dealer bins × 6 player bins = 18
    """
    dealer = np.zeros(3)
    player = np.zeros(6)

    if 1 <= state[0] <= 4: dealer[0] = 1
    if 4 <= state[0] <= 7: dealer[1] = 1
    if 7 <= state[0] <= 10: dealer[2] = 1

    if 1 <= state[1] <= 6: player[0] = 1
    if 4 <= state[1] <= 9: player[1] = 1
    if 7 <= state[1] <= 12: player[2] = 1
    if 10 <= state[1] <= 15: player[3] = 1
    if 13 <= state[1] <= 18: player[4] = 1
    if 16 <= state[1] <= 21: player[5] = 1

    return np.outer(dealer, player).flatten()

def policy_features(state, action):
    """
    Features for actor π(a|s)
    Same bins, duplicated per action
    """
    phi_s = state_features(state)
    phi = np.zeros((2, len(phi_s)))
    phi[action] = phi_s
    return phi.flatten()

########################################
# Softmax policy
########################################

def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)

########################################
# Parameters
########################################

num_episodes = 1000000
gamma = 1.0
lam = 0.3

alpha_actor = 0.01
alpha_critic = 0.05

# Actor parameters (policy)
theta = np.zeros(2 * 18)

# Critic parameters (value function)
w = np.zeros(18)

########################################
# Training loop
########################################

for episode in range(num_episodes):

    # Initial state
    state = [draw_from_deck(True), draw_from_deck(True)]
    reward = None

    # Eligibility traces
    e_theta = np.zeros_like(theta)
    e_w = np.zeros_like(w)

    while reward is None:

        # ----- ACTOR: sample action from policy -----
        prefs = np.array([
            policy_features(state, 0) @ theta,
            policy_features(state, 1) @ theta
        ])
        probs = softmax(prefs)
        action = np.random.choice([0, 1], p=probs)
        action_name = 'stick' if action == 0 else 'hit'

        # ----- Environment step -----
        next_state, reward = step(state, action_name)

        # ----- CRITIC: TD error -----
        v_s = state_features(state) @ w
        v_next = 0 if reward is not None else state_features(next_state) @ w
        delta = (reward if reward is not None else 0) + gamma * v_next - v_s

        # ----- Update eligibility traces -----

        # Critic trace: ∇w V(s)
        e_w = gamma * lam * e_w + state_features(state)

        # Actor trace: ∇θ log π(a|s)
        grad_log_pi = policy_features(state, action) \
                      - np.sum([
                          probs[a] * policy_features(state, a)
                          for a in [0, 1]
                      ], axis=0)
        e_theta = gamma * lam * e_theta + grad_log_pi

        # ----- Parameter updates -----
        w += alpha_critic * delta * e_w
        theta += alpha_actor * delta * e_theta

        state = next_state

# Plot the learned policy
import matplotlib.pyplot as plt
policy_grid = np.zeros((10, 21))
for dealer in range(1, 11):
    for player in range(1, 22):
        state = [dealer, player]
        prefs = np.array([
            policy_features(state, 0) @ theta,
            policy_features(state, 1) @ theta
        ])
        action = np.argmax(prefs)
        policy_grid[dealer-1, player-1] = action  # 0: stick, 1: hit

plt.imshow(policy_grid, origin='lower', aspect='auto', cmap='coolwarm')
plt.colorbar(label='Action (0: stick, 1: hit)')
plt.xlabel('Player Sum')
plt.ylabel('Dealer Showing')
plt.title('Learned Policy')
plt.show()