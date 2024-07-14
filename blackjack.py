import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def transition_matrix(n_episodes=100000):
    """
    Compute the transition matrix and reward matrix by playing episodes.
    Parameters
    ----------
    n_episodes : int
        The number of episodes to play.
    Returns
    -------
    P : (n, nd, na, n, nd) array
        The transition matrix.
    R : (n, nd, na, n, nd) array
        The reward matrix.
    """
    env = gym.make("Blackjack-v1")
    
    # Initialize transition matrix
    n_states = env.observation_space[0].n 
    nd_states = env.observation_space[1].n 
    n_actions = env.action_space.n
    P = np.zeros((n_states, nd_states, n_actions, n_states + 1, nd_states + 1))
    R = np.zeros((n_states, nd_states, n_actions, n_states + 1, nd_states + 1))
    # Play episodes
    for _ in range(n_episodes):
        s, info = env.reset()
        done = False
        
        while not done:
            
            a = env.action_space.sample() 
            next_s, reward, done, _, info = env.step(a)
            
            # Update reward matrix only when the game is over
            if done:
                P[s[0]-1, s[1]-1, a, n_states, nd_states] += 1
                R[s[0]-1, s[1]-1, a, n_states, nd_states] += reward
            else:
                 # Update transition matrix
                P[s[0]-1, s[1]-1, a, next_s[0]-1, next_s[1]-1] += 1
                
            s = next_s   
    
    # Normalize  P 
    for s in range(n_states):
        for d in range(nd_states):
            for a in range(n_actions):
                tot = np.sum(P[s,d,a,:,:])
                if tot > 0:
                    P[s,d,a,:,:] /= tot
                
    # Print with 3 decimal places    
    print(np.round(P, 3))
    
    return P,R

def modified_policy_iteration(P, R, gamma, k):
    """
    Perform modified policy iteration with initial policy of always asking for another card if the value is less than 21.
    Using policy evaluation and policy improvement.
    
    Parameters
    ----------
    P : (n, nd, na, n, nd)
        The transition matrix.
    R : (n, nd, na, n, nd)
        The reward matrix.
    gamma : float
        The discount factor.
    k : int
        The number of iterations to perform.
    Returns
    -------
    pi : (n, nd, na) array
        The final policy.
    V : (n+1, nd+1) array
    """
    # Initial policy
    n_states = P.shape[0]
    nd_states = P.shape[1]
    n_actions = P.shape[2]
    pi = np.ones((n_states, nd_states, n_actions)) 
    pi[:,:,0] = 0 # always draw a card
    pi[21:,:,1], pi[21:,:,0] = 0 , 1 # if bigger then 21, the game is over and the player loses
    
    # Initialize value function
    V = np.zeros((n_states + 1, nd_states + 1))
    
    # Algorithm
    for it in range(k):
        # Policy Evaluation
        V = policy_evaluation(pi, P, R, gamma, n_states, nd_states, V)  
        
        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            for d in range(nd_states):
                old_a = np.argmax(pi[s,d,:])
                for a in range(n_actions):
                    pi[s,d,a] = np.sum(P[s,d,a,:,:] * (gamma * V + R[s,d,a,:,:]))
                if old_a != np.argmax(pi[s,d,:]):
                    policy_stable = False
                
        if policy_stable:
            return pi, V
    
    return pi, V
    

def policy_evaluation(pi, P, R, gamma, n_states, nd_states, V):
    """
    Policy Evaluation.
    Prefer to use the Bellman equation for the value function:
    V(s) = sum_a(pi(s,a) * sum_s'(P(s,a,s') * (R(s,a,s') + gamma * V(s'))))
    Parameters
    ----------
    pi : (n, nd, na) array
        The policy.
    P : (n, nd, na, n, nd) array
        The transition matrix.
    R : (n, nd, na, n, nd) array
        The reward matrix.
    gamma : float
        The discount factor.
    n_states : int
        The number of states.
    nd_states : int
        The number of dealer states.
    V : (n+1, nd+1) array
        The value function.
    Returns
    -------
    V : (n+1, nd+1) array
        The value function.
    """
    # Iterate until convergence
    while True:
        delta = 0
        
        # Update value function for each state
        for s in range(n_states):
            for d in range(nd_states):     
                v = V[s,d]
                a = np.argmax(pi[s,d,:])
                V[s,d] = np.sum(P[s,d,a,:,:] * (gamma * V + R[s,d,a,:,:]))
                delta = max(delta, abs(v - V[s,d]))
        
        # Check for convergence
        if delta < 1e-6:
            break
    
    return V

#%%3A- PLAY GAME GIVEN A POLICY and anv

def play_game(env, policy):
    """
    Play a game with the given policy.
    Parameters
    ----------
    env : gymnasium.Env
        The environment to play the game in.
    policy : (n, nd, na) array
        The policy to follow.
        
    Returns
    -------
    average_reward : float
        The average reward over 100 episodes.
    """

    total_reward = 0
    num_episodes = 100

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(policy[state[0], state[1], :])
            next_state, reward, done, _, _ = env.step(action)

            total_reward += reward
            state = next_state

    average_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")
    
    return average_reward

#%% 3B - print state value:    
def value_function_q3(state, value_function):
    """
    With the given value function, print the value of the given state.
    Parameters
    ----------
    state : (n, nd) tuple
        The state to print the value of.
    value_function : (n+1, nd+1) array
        The value function.
    """

    s, d = state
    print(f"Value of state ({s}, {d}): {value_function[s, d]}")



    


def policy_iteration_with_initial(P, R, gamma, k, initial_policy, V):
    """
    Perform modified policy iteration with an initial policy.
    Parameters
    ----------
    P : (n, nd, na, n, nd) array
        The transition matrix.
    R : (n, nd, na, n, nd) array
        The reward matrix.
    gamma : float
        The discount factor.
    k : int
        The number of iterations to perform.
    initial_policy : (n, nd, na) array
        The initial policy.
    V : (n+1, nd+1) array
        The initial value function.
    Returns
    -------
    current_policy : (n, nd, na) array
        The final policy.   
    V : (n+1, nd+1) array
        The final value function.
    """
    n_states = P.shape[0]
    nd_states = P.shape[1]
    n_actions = P.shape[2]

    current_policy = initial_policy

    for _ in range(k):
        # Policy Evaluation
        V = policy_evaluation(current_policy, P, R, gamma, n_states, nd_states, V)

        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            for d in range(nd_states):
                old_a = np.argmax(pi[s,d,:])
                current_policy[s,d,:] = 0
                for a in range(n_actions):
                    current_policy[s,d,a] = np.sum(P[s,d,a,:,:] * (gamma * V + R[s,d,a,:,:]))
                if old_a != np.argmax(current_policy[s,d,:]):
                    policy_stable = False

        if policy_stable:
            return current_policy, V

    return current_policy, V


def plot_policy_value_iterations(env, initial_policy, gamma, max_iterations):
    """
    Plot the improvement of policy value over iterations.
    """
    n_states = env.observation_space[0].n 
    nd_states = env.observation_space[1].n
    current_policy = initial_policy
    current_value_function = np.zeros((n_states+1, nd_states+1))
    avg_values = []
    avg_rewards = []

    for iteration in range(max_iterations):
        current_policy, current_value_function = policy_iteration_with_initial(P, R, gamma, k=1, initial_policy=current_policy,V = current_value_function)
        avg_value = np.mean(current_value_function)
        avg_values.append(avg_value)

    # Plot the improvement of policy value
    plt.plot(range(1, max_iterations + 1), avg_values)
    plt.xlabel('Iteration Number')
    plt.ylabel('Average Value of Policy')
    plt.title('Improvement of Policy Value Over Iterations')
    plt.show()
    

    
def initial_policy(env):
    """
     the initial policy: always asks for another card if the value is less than 21.
    """
    n_states = env.observation_space[0].n
    nd_states = env.observation_space[1].n 
    n_actions = env.action_space.n

    initial_policy = np.ones((n_states, nd_states, n_actions))
    initial_policy[:, :, 0] = 0  # Always stick 
    initial_policy[21:, :, 0], initial_policy[21:, :, 1] = 1, 0  # If bigger than 21, the game is over and the player loses
    
    # for s in range(n_states):
    #     for d in range(nd_states):
    #         initial_policy[s, d, 1] = 1  # Always ask for another card if the value is less than 21

    return initial_policy

def ploting(V,pi):
    """Plot the value function and policy."""
    # print image of value function
    plt.imshow(V, cmap='hot', interpolation='nearest')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')
    plt.title('Value Function')
    plt.colorbar()
    plt.show()



    # priny image of action for state and diller state with policy
    # show only 4-21 for policy
    plt.imshow(np.argmax(pi, axis=2)[4:21,:], cmap='magma', interpolation='nearest')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')
    plt.title('Policy')
    plt.colorbar()
    plt.show()


# Call the functions
P, R = transition_matrix()
pi, V = modified_policy_iteration(P, R, gamma=1, k=10)

ploting(V,pi)


env = gym.make("Blackjack-v1")
play_game(env, pi)

initial_pi = initial_policy(env)

# Plot the improvement of policy value over iterations
plot_policy_value_iterations(env, initial_pi, gamma=1, max_iterations=10)
