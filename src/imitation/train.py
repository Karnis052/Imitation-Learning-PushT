"""Train and evaluate a Push-T imitation policy."""


from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
from typing import Any

from dataclasses import dataclass, asdict 
from pathlib import Path
from imitation.model import PolicyType, build_policy
from imitation.data import download_pusht, load_pusht_zarr, Normalizer, PushtChunkDataset
from imitation.evaluation import Logger

LOGDIR_PREFIX = "exp"

@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    policy_type: PolicyType = "mse"
    flow_num_steps: int = 10
    chunk_size: int = 8
    
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dimns: tuple[int, ...] = (256, 256, 256)
    num_epochs: int = 400
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    log_interval: int = 100
    seed: int = 52
    wandb_project: str = "push-t-imitation"
    exp_name: str| None = None

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data

def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)
    
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    
    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dimns=config.hidden_dimns,
    ).to(device)
    
    
    exp_name = f"seed{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
        
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    
    wandb.init(
        project=config.wandb_project,
        config = config_to_dict(config),
        name=exp_name,
    )
    logger = Logger(log_dir)
    
    logger.dump_for_grading()
    
    
def main() -> None:
    config = TrainConfig()
    run_training(config)
        

if __name__ == "__main__":
    main()