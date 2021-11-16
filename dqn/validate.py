import numpy as np
import gym

from dqn.policies import Policy


def validate(env: gym.Env, policy: Policy, num_steps: int = None, num_trials=None) -> float:

    returns = []
    total_reward = 0

    obs, done = env.reset(), False

    if num_steps:
        for _ in range(num_steps):

            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                returns.append(total_reward)
                total_reward = 0
                obs, done = env.reset(), False

    elif num_trials:
        for _ in range(num_trials):
            done=False
            
            while not done:
                action = policy(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            returns.append(total_reward)
            total_reward = 0
            obs, done = env.reset(), False

    return np.mean(returns) if len(returns) > 0 else total_reward
