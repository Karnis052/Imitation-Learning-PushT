"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias
import torch
import torch.nn as nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""
    
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
    
    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute the loss for a batch of (state, action_chunk) pairs."""
        
    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor, *, num_steps: int = 10) -> torch.Tensor:
          """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dimns: tuple[int, ...] = (256, 256, 256),
     ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        dims = [state_dim, *hidden_dimns]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], chunk_size*action_dim))
        self.net = nn.Sequential(*layers)
        
    
    def _forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.net(state)
        return out.reshape(-1, self.chunk_size, self.action_dim)
    

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self._forward(state)
        return nn.functional.mse_loss(pred, action_chunk)
    
    
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
       return self._forward(state)
    

class FlowPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dimns: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        in_dims = state_dim + chunk_size*action_dim +1    # state + flattened x_t + scalar t
        dims =  [in_dims, *hidden_dimns]
        layers: list[nn.Module] = []
        for i, o in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(i, o))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], chunk_size*action_dim))
        self.net = nn.Sequential(*layers)
    
    def _velocity(
        self, 
        x_t: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor
        ) -> torch.Tensor:
        # x_t = (batch, chunk_size, action_dim), t: (batch, 1), state: (batch, state_dim)
        inp = torch.cat([state, x_t.flatten(1), t], dim=-1)
        out = self.net(inp)
        return out.reshape(-1, self.chunk_size, self.action_dim)
        
        
    def compute_loss(
        self,
        state: torch.Tensor,  #(batch, state_dim)
        action_chunk: torch.Tensor, #(batch, chunk_size, action_dim)
    ) -> torch.Tensor:
        x1 = action_chunk
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, device = x1.device)  #(batch, 1)
        t_broadcast = t.unsqueeze(-1)           #(batch, 1, 1)
        
        x_t = (1-t_broadcast)*x0 + t_broadcast*x1
        v_target = x1-x0
        v_pred = self._velocity(x_t, t, state)
        return nn.functional.mse_loss(v_pred, v_target)
        
    
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:    # return (batch, chunk_size, action_dim)
        
        batch = state.shape[0]
        x = torch.randn(batch, self.chunk_size, self.action_dim, device=state.device)
        dt = 1.0/num_steps
        
        for step in range(num_steps):
            t = torch.full((batch,1), step*dt, device=state.device)
            v = self._velocity(x, t, state)
            x = x + v*dt 
        return x
            

PolicyType: TypeAlias = Literal["mse", "flow"]

def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dimns: tuple[int, ...] = (256, 256, 256),
) -> BasePolicy: 
    if policy_type == "mse":
        return MSEPolicy(
            state_dim = state_dim, 
            action_dim = action_dim,
            chunk_size = chunk_size,
            hidden_dimns = hidden_dimns
        )
    if policy_type == "flow":
        return FlowPolicy(
            state_dim = state_dim,
            action_dim = action_dim,
            chunk_size = chunk_size,
            hidden_dimns = hidden_dimns
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

    