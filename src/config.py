"""Configuration dataclasses for the environment and for training.

Every run saves both configs to `config.json`, so a run can be reproduced
exactly from its folder.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EnvConfig:
    m: float = 75.0              # robot mass [kg]
    C: float = 0.5               # energy-penalty coefficient: e(t) = C/100 * ||F||^2
    dt: float = 0.1              # time step [s] (10 decisions per second)
    f_max: float = 100.0         # force limit ||F|| <= f_max [N]
    steps_per_target: int = 200  # 20 s per target
    n_targets: int = 1           # 1 for training, 5 for evaluation
    half_w: float = 30.0         # rink is 60 x 30 m: x in [-30, 30]
    half_h: float = 15.0         #                    y in [-15, 15]
    margin: float = 2.0          # starts and targets stay this far from the boundary [m]


@dataclass
class TrainConfig:
    # Frozen after tuning at m = 75, C = 0.5 on the validation episodes
    # (experiments/tune.py -> results/tuning_lr.csv, results/tuning_other.csv).
    n_iters: int = 1000
    batch: int = 64              # episodes per gradient estimate (simulated in parallel)
    lr: float = 1e-2             # Adam step size
    init_std: float = 2.0        # initial policy std, in units of 100 N (= the upper clamp)
    eval_every: int = 25         # deterministic evaluation on the validation set every k iterations
    eval_episodes: int = 20
    seed: int = 0


def save_config(path, env_cfg: EnvConfig, train_cfg: TrainConfig) -> None:
    """Write both configs to a JSON file: {"env": {...}, "train": {...}}."""
    data = {"env": asdict(env_cfg), "train": asdict(train_cfg)}
    Path(path).write_text(json.dumps(data, indent=2))


def load_config(path) -> tuple[EnvConfig, TrainConfig]:
    """Inverse of `save_config`."""
    data = json.loads(Path(path).read_text())
    return EnvConfig(**data["env"]), TrainConfig(**data["train"])
