import retro
import time
from sfenv import SFENV


env = SFENV(True)
env2 = SFENV(True)
print(env.actions_names)
print(env.actions_space, "Action size")
try:
    env.reset()
    env2.reset()
    done = False
    while not done:
        action = env.sample()
        action2 = env2.sample()
        ob, reward, done, info = env.step(action)
        env2.step(action2)
        print(reward, done, env.info['enemy_matches_won'], env.actions_names[action])
    env2.close()
    env.close()
except KeyboardInterrupt:
        env.close()
        env2.close()
        exit()

