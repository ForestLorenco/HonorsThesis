import retro
import time
from sfenv import SFENV
import cv2


env = SFENV(True,multi=False, skip=False)
#env2 = SFENV(False)
print(env.actions_names)
print(env.actions_space, "Action size")
try:
    env.reset()
    #env2.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.sample()
        #action2 = env2.sample()
        ob, reward, done, info = env.step(action)
        #x_t1 = cv2.resize(cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY), (84, 84))
        #cv2.imwrite('before.png',ob)
        #cv2.imwrite('after.png',x_t1)
        #break
        print()
        #env2.step(action2)
        total_reward += reward
        print(info)
        break
        #print("Reward:{}, Done:{}, E_Matches:{}, matches, {}, Action:{}, Health:{}, Enemy Health:{}".format(total_reward, done, env.info['enemy_matches_won'], env.info["matches_won"],env.actions_names[action],info["health"], info["enemy_health"]))
        '''
        if env.info["matches_won"] > 0:
            break
        if env.info["enemy_matches_won"] > 0:
            break
        '''
    #env2.close()
    env.close()
except KeyboardInterrupt:
        env.close()
        #env2.close()
        exit()

