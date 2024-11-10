import random
import gym
import numpy as np
import matplotlib.pyplot as plt

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    max_q_sprime = np.max(Q[sprime])  
    Q[s][a] = Q[s][a] + alpha * (r + gamma * max_q_sprime - Q[s][a])
    return Q   

def epsilon_greedy(Q, s, epsilon):
    if random.uniform(0, 1) > epsilon:
        return np.argmax(Q[s]) 
    else:
        return env.action_space.sample()  

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    alpha = 0.1
    gamma = 0.9
    epsilon = 1 
    max_epsilon = 1
    min_epsilon = 0.01
    epsilon_step = 0.0009
    n_epochs = 10000 
    test_episode = 100
    max_itr_per_epoch = 100 
    rewards = []
    eL = []
    epL = []

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)
            Sprime, R, done, _, info = env.step(A)
            r += R
            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)
            S = Sprime
            if done:
                break

        if e % 100 == 0:
            print(f"Mean reward at episode {e}: {np.mean(rewards[-100:])}")
        
        print("episode #", e, " : r = ", r," epsilon : ", epsilon)
        eL.append(e)
        rewards.append(r)
        epL.append(epsilon)
        #epsilon = max(0.01, epsilon * np.exp(-epsilon_step * e))
        epsilon =min_epsilon+ ((max_epsilon-min_epsilon)*np.exp(-epsilon_step*e))
        #epsilon = np.sin(np.pi*)
        
    print("Average reward = ", np.mean(rewards))

    plt.plot(eL, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards Over Episodes")
    plt.show()

    plt.plot(eL, epL)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Episodes")
    plt.show()

    print("Training finished.\n")

    env.close()
