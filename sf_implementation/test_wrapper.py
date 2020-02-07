import retrowrapper
import retro
if __name__ == "__main__":

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env1 = retrowrapper.RetroWrapper(game)
    env2 = retrowrapper.RetroWrapper(game)
    _obs = env1.reset()
    _obs = env2.reset()

    done = False
    while not done:
        action = env1.action_space.sample()
        _obs, _rew, done, _info = env1.step(action)
        env1.render()

        action = env2.action_space.sample()
        _obs, _rew, done, _info = env2.step(action)
        env2.render()