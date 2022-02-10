import numpy as np
import gym
import torch

from PT.dqn.policies import Policy


def validate(env: gym.Env, policy: Policy, num_steps: int = None, num_trials=None, on_validation_step=None) -> float:

    returns = []
    total_reward = 0

    obs, done = env.reset(), False

    if num_steps:
        for i in range(num_steps):

            action, q_value = policy(torch.as_tensor(obs).float())
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                returns.append(total_reward)
                total_reward = 0
                _, done = env.reset(), False

    elif num_trials:
        for i in range(num_trials):
            done=False
            j=0            
            while not done:
                obs_base=obs
                action, q_value = policy(torch.as_tensor(obs).float())
                obs, reward, done, _ = env.step(action)
                total_reward += reward

                if on_validation_step:
                    on_validation_step(i+j, obs_base, q_value)
                    
                j+=1

            returns.append(total_reward)
            total_reward = 0
            _, done = env.reset(), False

    return np.mean(returns) if len(returns) > 0 else total_reward
