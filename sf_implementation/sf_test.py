import retro
import time
from sfenv import SFENV


env = SFENV(True)
print(env.actions_names)
print(env.actions_space, "Action size")
try:
    env.reset()
    done = False
    while not done:
        action = env.sample()
        ob, reward, done, info = env.step(action)
        print(reward, done, env.info['enemy_matches_won'], env.actions_names[action])

    env.close()
except KeyboardInterrupt:
        env.close()
        exit()

