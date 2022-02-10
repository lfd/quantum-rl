"""Memory replay utilities."""

import random
from dataclasses import dataclass, astuple

import tensorflow as tf

from typing import Any, Iterable, Optional


@dataclass
class Transition:
    state: tf.Tensor
    action: int
    reward: float
    is_terminal: bool
    next_state: Optional[tf.Tensor]
        
        
@dataclass
class Batch:
    states: tf.Tensor
    actions: tf.Tensor
    rewards: tf.Tensor
    is_terminal: tf.Tensor
    next_states: tf.Tensor


class ReplayMemory:
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items = [] 
        self._idx = 0
        
        
    def store(self, transition: Transition) -> None:
        
        item = astuple(transition)
        
        # Always appending is probably very memory inefficient
        if len(self) < self.capacity:
            self._items.append(item)
        else:
            self._items[self._idx] = item
            
        self._idx = (self._idx + 1) % self.capacity
        
        
    def sample(self, batch_size: int) -> Batch:
        """Sample a batch of transitions, uniformly at random"""
        
        assert len(self._items) >= batch_size, (
            f'Cannot sample batch of size {batch_size} from buffer with'
            f' {len(self)} elements.'
        )
        
        # Sample a new batch without replacement
        sampled = random.sample(self._items, batch_size)
        
        # Transpose List[Transition] -> List[State], List[Action], etc.
        states, actions, rewards, is_terminal, next_states = zip(*sampled)
        
        # Places resulting tensors automatically on GPU
        # TODO rethink whether manual placement is necessary?
        batch = Batch(
            states = tf.stack(states),
            actions = tf.convert_to_tensor(actions),
            rewards = tf.convert_to_tensor(rewards),
            is_terminal = tf.convert_to_tensor(is_terminal),
            next_states = tf.stack([s for s in next_states if s is not None])
        )
        
        return batch
        
        
    def __len__(self) -> int:
        return len(self._items)
    
    
    def __iter__(self) -> Iterable[Any]:
        return iter(self._items)

    
    def __str__(self) -> str:
        return str(self._items)