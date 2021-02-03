import numpy as np
import gym

from dqn.policies import Policy


def validate(env: gym.Env, policy: Policy, num_steps: int) -> float:

    returns = []
    total_reward = 0

    obs, done = env.reset(), False

    for _ in range(num_steps):

        action = policy(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            returns.append(total_reward)
            total_reward = 0
            obs, done = env.reset(), False

    return np.mean(returns) if len(returns) > 0 else total_reward
