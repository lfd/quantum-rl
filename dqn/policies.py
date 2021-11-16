"""Classes to represent Reinforcement Learning policies."""

import abc
import random

import gym
import tensorflow as tf

from typing import Union, List


class Policy(abc.ABC):
    """Abstract base class for policies."""
    
    @abc.abstractmethod
    def action(self, state: tf.Tensor) -> int:
        """Returns action if len(states) == 1"""
        pass
    
    def __call__(self, state: tf.Tensor) -> int:
        return self.action(state)


class ActionValuePolicy(Policy):
    """Selects the greedy action with respect to a set of action values (a.k.a. 
    Q-values), predicted by a model based on a given state."""
    
    def __init__(self, model: tf.Module):
        super(ActionValuePolicy, self).__init__()
        self.model = model
        
        
    def values(self, states: tf.Tensor) -> tf.Tensor:
        return self.model(states)
    
    
    def action(self, state: tf.Tensor) -> int:
        
        # TODO check that state is correct shape?
        
        # Add an artificial batch dimension (Some models, e.g. CNNs only work
        # with batches)
        inputs = tf.expand_dims(state, axis=0)
        
        q_values = self.values(inputs)
        return int(tf.argmax(q_values, axis=-1))


class EpsilonGreedyPolicy(Policy):
    """Selects an action from the underlying `policy` with a probability of
    `epsilon`, or uniformly at random otherwise."""
    
    def __init__(self, policy: Policy, action_space: gym.Space, epsilon: float = 1.0):
        super(EpsilonGreedyPolicy, self).__init__()
        
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f'Epsilon must be in range[0,1], but was {epsilon}')
            
        self.policy = policy
        self.action_space = action_space
        self.epsilon = epsilon
        
        self._did_explore = False
        

    @property
    def did_explore(self):
        return self._did_explore
    
    
    def action(self, state: tf.Tensor) -> int:
        self._did_explore = random.random() < self.epsilon
        action = self.action_space.sample() if self.did_explore else self.policy(state)
        
        return action


class LinearDecay:
    """Let's the `epsilon` parameter of the underlying `EpsilonGreedyPolicy`
    decay linearly from a `start` to `end` over a set number of steps."""

    def __init__(self, policy: EpsilonGreedyPolicy, num_steps: int, start: float = 1.0, end: float = 0.0):

        if not 0.0 <= start <= 1.0:
            raise ValueError(f'start must be in range [0,1], but was {start}')

        if not 0.0 <= end <= 1.0:
            raise ValueError(f'end must be in range [0,1], but was {end}')

        if num_steps <= 0:
            raise ValueError(f'num_steps must be a positive integer, but was {num_steps}')

        self.policy = policy
        self.start = start 
        self.end = end
        self.num_steps = num_steps
        
        self.policy.epsilon = self.start

        self._step = 0
        self._slope = (end - start) / (num_steps - 1)
        
    
    def step(self) -> None:

        self._step += 1
        
        if self._step < self.num_steps:
            self.policy.epsilon = self.start + self._slope * self._step

class LinearIncrease:
    """Let's the `update_every` parameter increase linearly from a `start` to `end` 
    over a set number of steps."""

    def __init__(self, num_steps: int, start: int = 10, end: int = 1000):

        if not start <= end:
            raise ValueError(f'start must be smaller than or equal to end, but was start: {start}, end: {end}')

        if num_steps <= 0:
            raise ValueError(f'num_steps must be a positive integer, but was {num_steps}')

        self.start = start 
        self.end = end
        self.num_steps = num_steps
        
        self.current = self.start

        self._step = 0
        self._slope = (end - start) / (num_steps - 1)
        
    
    def step(self) -> None:

        self._step += 1
        
        if self._step < self.num_steps:
            self.current = int(self.start + self._slope * self._step)