# Imitation Learning on Push-T

Action-chunking imitation learning on the [Push-T](https://diffusion-policy.cs.columbia.edu/) dataset, for the Berkeley Deep RL course homework.

Two policies, same training/eval pipeline:

- **`mse`** — MLP that directly regresses an action chunk (MSE loss).
- **`flow`** — conditional flow-matching policy; learns a velocity field from noise to the action chunk, samples by Euler-integrating it over `flow_num_steps` steps.

## Setup & run

```bash
uv sync
uv run imitation
```

Downloads the Push-T dataset on first run, trains per `TrainConfig` in [`src/imitation/train.py`](src/imitation/train.py) (policy type, chunk size, LR, epochs, etc.), and logs to Weights & Biases. Every `eval_interval` steps it rolls out episodes in the Push-T gym env, logs mean reward/videos, and checkpoints the model.

## Layout

```
src/imitation/
  data.py         # dataset download, normalization, chunked (state, action) dataset
  model.py         # BasePolicy, MSEPolicy, FlowPolicy
  train.py         # training loop, TrainConfig
  evaluation.py     # env rollouts, W&B logging/checkpointing
  modal_train.py    # remote training on Modal
```