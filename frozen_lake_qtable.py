#Here is a implementation of Frozen lake game with a simple Q table implementatoin
import gym
import numpy as np

env = gym.make("FrozenLake-v0")

#implement the Q-learning algorithm
Q = np.zeros([env.observation_space.n, env.action_space.n])
print(env.observation_space.n, env.action_space.n)
#learning parameters
lr = .8 #learning rate of alg
y = .95 #max discounted
num_episodes = 2000 #how many iter

#create lists to contain total rewards and steps per episode
reward_List = []
for i in range(num_episodes):
    #reset the enc and get first obs
    s = env.reset()#this is current state
    #env.render()
    reward_All = 0
    done = False
    j = 0
    #The q-table learning alg
    while j < 99:
        j+= 1
        #choose an action by greedily picking from Q table
        action = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #get new state and reward from env
        next_state, reward, done, _ = env.step(action)
        #update q-table with new knowledge
        Q[s,action] = Q[s,action]+ lr*(reward + y*np.argmax(Q[next_state, :])- Q[s,action])
        reward_All += reward
        s = next_state
        if done:
            break
    reward_List.append(reward_All)
#env.close()

print("Score over time: ", str(sum(reward_List)/num_episodes))
print("Final Q table values:")
print(Q)