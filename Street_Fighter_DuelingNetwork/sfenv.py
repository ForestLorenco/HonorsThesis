import retro
import retrowrapper
from sf_constants import ACTIONS
from sf_constants import COMBOS
from sf_constants import SKIP_FRAMES
import random as rand

class SFENV:
    def __init__(self, render=False, multi=True, skip=True):
        '''
        Wrapper class for street fighter II environment. This has implementations for simple
        movements for easy training as well as wait times for move animations. 
        '''
        if multi:
            self.env = retrowrapper.RetroWrapper('StreetFighterIISpecialChampionEdition-Genesis')
        else:
            self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
        self.render = render
        self.ob, self.reward, self.done, self.info = None, None, None, None
        self.actions_space = len(ACTIONS)+len(COMBOS)
        self.actions_names = list(ACTIONS.keys())
        self.actions_names.extend(['HURRICANE_KICK', 'SHORYUKEN', 'HADOKEN'])
        self.dead = False
        self.skip = skip

    def reset(self):
        '''
        Resets the environment and skips starting frames
        '''
        self.env.reset()
        for _ in range(SKIP_FRAMES):
            action = ACTIONS["neutral"][0]
            self.ob, _, self.done, self.info = self.env.step(action)
        return self.ob
    
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
        reward = 0
        info = self.info
        for _ in range(frames):
            if self.render:
                self.env.render()
            self.ob, _, self.done, n_info = self.env.step(ACTIONS["neutral"][0])
            #time.sleep(.5)
            if(not self.dead):
                reward += (info["enemy_health"] - n_info["enemy_health"]) - (info["health"] - n_info["health"])
            info = n_info
        return self.ob, self.done, info, reward

    def execute_combo(self, combo):
        '''
        Given a combo (a list of actions), executes the combo
        '''
        info = self.info
        reward = 0
        for a in combo:
            if self.render:
                self.env.render()
            self.ob, _, self.done, n_info = self.env.step(a)
            if(not self.dead):
                reward += (info["enemy_health"] - n_info["enemy_health"]) - (info["health"] - n_info["health"])
            info = n_info
        return self.ob, self.done, info, reward


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

                reward = 0
                if self.skip:
                    self.ob, self.done, self.info,reward = self.execute_combo(action[0])
                    self.ob, self.done, info, w_reward = self.wait(action[1])
                    reward+= w_reward
                else:
                    self.ob, self.done, info, reward = self.execute_combo(action[0])
                #reward calculations
                if self.dead:
                    if self.info["health"] == 176 or self.info["enemy_health"] == 176:
                        self.dead = False
                    self.reward = 0
                if (self.info["health"] <= 0) or (self.info["enemy_health"] <= 0):
                    self.dead = True
                    self.reward = reward
                else:
                    if not self.skip:
                        reward += (self.info["enemy_health"] - info["enemy_health"]) - (self.info["health"] - info["health"]) 
                    self.reward = reward

                if self.info["matches_won"] < info["matches_won"]:
                    
                    self.reward += 176

                if self.info["enemy_matches_won"] < info["enemy_matches_won"]:
                    
                    self.reward -= 176
                self.info = info
                #Temporary for now, might change this done condition to go to other characters
                if self.info["matches_won"] == 2 or self.info["enemy_matches_won"] == 2:
                    self.done = True
                return self.ob, self.reward, self.done, self.info
            else:
                action = ACTIONS[self.actions_names[a]]
                if self.render:
                    self.env.render()
                
                reward = 0
                if (a >= 10) and self.skip:
                    self.ob, _, self.done, self.info = self.env.step(action[0])
                    self.ob, self.done, info, reward = self.wait(action[1])
                else:
                    self.ob, _, self.done, info = self.env.step(action[0])
                #reward calculations
                if self.dead:
                    #print("we are dead")
                    if self.info["health"] == 176 or self.info["enemy_health"] == 176:
                        self.dead = False
                    self.reward = 0
                if (self.info["health"] <= 0) or (self.info["enemy_health"] <= 0):
                    self.dead = True
                    self.reward = reward
                else:
                    if a < 10 or (not self.skip):
                        reward += (self.info["enemy_health"] - info["enemy_health"]) - (self.info["health"] - info["health"]) 
                    self.reward = reward

                if self.info["matches_won"] < info["matches_won"]:
                    
                    self.reward += 176
                    
                if self.info["enemy_matches_won"] < info["enemy_matches_won"]:
                    
                    self.reward -= 176
                self.info = info
                #Temporary for now, might change this done condition to go to other characters
                if self.info["matches_won"] == 2 or self.info["enemy_matches_won"] == 2:
                    self.done = True
                return self.ob, self.reward, self.done, self.info
        except IndexError:
            print("Index too large for action space")