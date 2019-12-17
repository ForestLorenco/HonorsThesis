import retro
from sf_constants import ACTIONS
from sf_constants import COMBOS
from sf_constants import SKIP_FRAMES
import random as rand

class SFENV:
    def __init__(self, render=False):
        '''
        Wrapper class for street fighter II environment. This has implementations for simple
        movements for easy training as well as wait times for move animations. 
        '''
        self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
        self.render = render
        self.ob, self.reward, self.done, self.info = None, None, None, None
        self.actions_space = len(ACTIONS)+len(COMBOS)
        self.actions_names = list(ACTIONS.keys())
        self.actions_names.extend(['HURRICANE_KICK', 'SHORYUKEN', 'HADOKEN'])

    def reset(self):
        '''
        Resets the environment and skips starting frames
        '''
        self.env.reset()
        for i in range(SKIP_FRAMES):
            action = ACTIONS["neutral"][0]
            self.ob, _, self.done, self.info = self.env.step(action)
    
    def close(self):
        '''
        Closes the environment
        '''
        self.env.close()

    def sample(self):
        '''
        Returns a random move from the action space
        '''
        return rand.randint(0,self.actions_space-1)

    def wait(self, frames):
        '''
        Forces agent to wait before inputing a new move until after animation 
        frames are done, should speed up training
        '''
        info = self.info
        for i in range(frames):
            if self.render:
                self.env.render()
            self.ob, _, self.done, info = self.env.step(ACTIONS["neutral"][0])
            #time.sleep(.5)
        return self.ob, self.done, info

    def execute_combo(self, combo):
        '''
        Given a combo (a list of actions), executes the combo
        '''
        info = self.info
        for a in combo:
            if self.render:
                self.env.render()
            self.ob, _, self.done, info = self.env.step(a)
        return self.ob, self.done, info


    def step(self, a):
        '''
        #TODO- implement a version of done that lets the agent fight all 12 fighters
        Note in order for Done to work, must change value in scenario.json from 10 to 9
        '''
        try:
            if a >= len(ACTIONS):
                if self.render:
                    self.env.render()
                action = COMBOS[a%len(ACTIONS)]
                self.ob, self.done, info = self.execute_combo(action[0])
                self.ob, self.done, info = self.wait(action[1])
                self.reward = (self.info["enemy_health"] - info["enemy_health"]) - (self.info["health"] - info["health"]) 
                self.info = info
                #Temporary for now, might change this done condition to go to other characters
                if self.info["enemy_health"] == 0 and self.info["matches_won"] == 2:
                    self.done == True
                return self.ob, self.reward, self.done, self.info
            else:
                action = ACTIONS[self.actions_names[a]]
            if self.render:
                self.env.render()
            self.ob, _, self.done, self.info = self.env.step(action[0])
            self.ob, self.done, info = self.wait(action[1])
            self.reward = (self.info["enemy_health"] - info["enemy_health"]) - (self.info["health"] - info["health"]) 
            self.info = info
            #Temporary for now, might change this done condition to go to other characters
            if self.info["enemy_health"] == 0 and self.info["matches_won"] == 2:
                    self.done == True
            return self.ob, self.reward, self.done, self.info
        except IndexError:
            print("Index too large for action space")