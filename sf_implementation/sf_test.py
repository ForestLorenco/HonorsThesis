import retro
import time
from sfenv import SFENV


env = SFENV(True)
print(env.actions_names)
print(env.actions_space)
try:
    env.reset()
    done = False
    while not done:
        action = env.sample()
        ob, reward, done, info = env.step(action)
        print(reward, done, env.actions_names[action])

    env.close()
except KeyboardInterrupt:
        env.close()
        exit()

