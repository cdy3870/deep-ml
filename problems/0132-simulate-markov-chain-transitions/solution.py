import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    # Your code here
    
    n_states = transition_matrix.shape[0]
    states = [initial_state]
    current_state = states[0]

    for i in range(num_steps):
        probabilities = transition_matrix[current_state]
        state = np.random.choice(n_states, p=probabilities)
        states.append(state)
        current_state = state

    return states