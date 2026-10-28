# Copyright 2025 Zichong Li, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Original MARLadona PPO settings, independent of the simulator SDK."""

from dataclasses import asdict, dataclass, field


@dataclass
class PolicyReplayCfg:
    dynamic_generate_replay_level: bool = True
    shuffle: bool = False
    score_difference: float = 0.75
    load_path: str = ""
    max_num_policy_replay_level: int = 2


@dataclass
class SymmetryCfg:
    use_symmetry: bool = True
    use_augmentation: bool = True
    use_loss: bool = False
    symmetry_coeff: float = 0.0


@dataclass
class PolicyCfg:
    class_name: str = "ActorCriticBeta"
    init_noise_std: float = 1.0
    actor_hidden_dims: list[int] = field(default_factory=lambda: [128, 128, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [128, 128, 128])
    activation: str = "elu"


@dataclass
class CustomRslRlPpoAlgorithmCfg:
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.003
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    symmetry_cfg: SymmetryCfg = field(default_factory=SymmetryCfg)


@dataclass
class SoccerMARLPPORunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 48
    max_iterations: int = 10000
    save_interval: dict[str, int] = field(default_factory=lambda: {"0": 20, "200": 50, "500": 100, "2000": 500})
    experiment_name: str = "marl_soccer"
    run_name: str = ""
    empirical_normalization: bool = False
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    logger: str = "tensorboard"
    wandb_project: str = "marl_soccer"
    neptune_project: str = "marl_soccer"
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    algorithm: CustomRslRlPpoAlgorithmCfg = field(default_factory=CustomRslRlPpoAlgorithmCfg)
    policy_replay: PolicyReplayCfg = field(default_factory=PolicyReplayCfg)

    def to_dict(self):
        return asdict(self)
