import retro
from constants import ACTIONS
from constants import COMBOS
from constants import SKIP_FRAMES
import random as rand

class SFENV:
    def __init__(self, render=False):
        self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
        self.render = render
        self.ob, self.reward, self.done, self.info = None, None, None, None
        self.actions_space = len(ACTIONS)+len(COMBOS)
        self.actions_names = list(ACTIONS.keys())
        self.actions_names.extend(['HURRICANE_KICK', 'SHORYUKEN', 'HADOKEN'])

    def reset(self):
        self.env.reset()
        for i in range(SKIP_FRAMES):
            action = ACTIONS["neutral"][0]
            self.ob, _, self.done, self.info = self.env.step(action)
    
    def close(self):
        self.env.close()

    def sample(self):
        return rand.randint(0,self.actions_space-1)

    def wait(self, frames):
        info = self.info
        for i in range(frames):
            if self.render:
                self.env.render()
            self.ob, _, self.done, info = self.env.step(ACTIONS["neutral"][0])
            #time.sleep(.5)
        return self.ob, self.done, info

    def execute_combo(self, combo):
        info = self.info
        for a in combo:
            if self.render:
                self.env.render()
            self.ob, _, self.done, info = self.env.step(a)
        return self.ob, self.done, info


    def step(self, a):
        try:
            if a >= len(ACTIONS):
                action = COMBOS[a%len(ACTIONS)]
                self.ob, self.done, info = self.execute_combo(action[0])
                self.ob, self.done, info = self.wait(action[1])
                self.reward = (self.info["enemy_health"] - info["enemy_health"]) - (self.info["health"] - info["health"]) 
                self.info = info
                return self.ob, self.reward, self.done, self.info
            else:
                action = ACTIONS[self.actions_names[a]]
                
            self.ob, _, self.done, self.info = self.env.step(action[0])
            self.ob, self.done, info = self.wait(action[1])
            self.reward = (self.info["enemy_health"] - info["enemy_health"]) - (self.info["health"] - info["health"]) 
            self.info = info
            return self.ob, self.reward, self.done, self.info
        except IndexError:
            print("Index too large for action space")