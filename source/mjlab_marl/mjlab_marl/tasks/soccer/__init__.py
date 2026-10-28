import gymnasium as gym

from .soccer_marl_env import SoccerMARLEnv
from .soccer_marl_env_cfg import SoccerMARLEnvCfg, SoccerMARLEnvPlayCfg

for name, cfg_type in (("Mjlab-Soccer-v0", SoccerMARLEnvCfg), ("Mjlab-Soccer-Play-v0", SoccerMARLEnvPlayCfg)):
    gym.register(
        id=name,
        entry_point=SoccerMARLEnv,
        disable_env_checker=True,
        order_enforce=False,
        kwargs={"env_cfg_entry_point": cfg_type},
    )

__all__ = ["SoccerMARLEnv", "SoccerMARLEnvCfg", "SoccerMARLEnvPlayCfg"]
