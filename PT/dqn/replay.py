"""Memory replay utilities."""

import random
from dataclasses import dataclass, astuple
import torch

from typing import Any, Iterable, Optional


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    is_terminal: bool
    next_state: Optional[torch.Tensor]
        
        
@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    is_terminal: torch.Tensor
    next_states: torch.Tensor


class ReplayMemory:
    
    def __init__(self, capacity: int, device=torch.device("cpu")):
        self.capacity = capacity
        self._items = [] 
        self._idx = 0
        self.device=device
        
        
    def store(self, transition: Transition) -> None:
        
        item = astuple(transition)
        
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
        
        batch = Batch(states, actions, rewards, is_terminal,
            [s for s in next_states if s is not None]
        )

        batch = Batch(
            states = torch.stack(states).float().to(self.device),
            actions = torch.as_tensor(actions),
            rewards = torch.as_tensor(rewards).float(),
            is_terminal = torch.as_tensor(is_terminal),
            next_states = torch.stack([s for s in next_states if s is not None]).float().to(self.device)
        )
        
        return batch
        
        
    def __len__(self) -> int:
        return len(self._items)
    
    
    def __iter__(self) -> Iterable[Any]:
        return iter(self._items)

    
    def __str__(self) -> str:
        return str(self._items)
