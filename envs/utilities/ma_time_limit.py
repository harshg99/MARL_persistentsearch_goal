import gym

'''Main difference from gym wrappers time_limit.py is that it updates a
    done dict in step rather than a done boolean
    Edited by Christopher Hsu
'''

class maTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(maTimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        #assert max_episode_steps, f"env spec {self.env.spec}"
        print(self._max_episode_steps)

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done['__all__'] = True
            # done = True       #og gym version
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)