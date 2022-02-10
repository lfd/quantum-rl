import gym

from TF.dqn.policies import Policy
from TF.dqn.replay import Transition

class Sampler:

    def __init__(self, policy: Policy, env: gym.Env) -> None:
        self._policy = policy
        self._env = env

        self._state = self._env.reset()


    def step(self) -> Transition:
        """Samples the next transition"""

        action, _ = self._policy(self._state)

        next_state, reward, done, _ = self._env.step(action)

        transition = Transition(
            state=self._state,
            action=action,
            reward=reward,
            is_terminal=done,
            next_state=next_state if not done else None
        )

        self._state = self._env.reset() if done else next_state
        return transition
