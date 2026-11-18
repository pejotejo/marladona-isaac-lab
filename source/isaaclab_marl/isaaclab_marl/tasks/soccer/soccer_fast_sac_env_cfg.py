from isaaclab.utils import configclass

from .soccer_marl_env_cfg import SoccerMARLEnvCfg


@configclass
class SoccerFastSACEnvCfg(SoccerMARLEnvCfg):
    return_final_observations: bool = True

    def __post_init__(self):
           self.soccer_game.num_agents_per_team = 3
           self.soccer_game.num_agents_range = [3, 3]
           self.soccer_game.agent_id_control = [
               [1, 3, 3],
               [3, 3, 3],
           ]
           self.soccer_game.eval_env_ratio = 0.0

           self.scene.num_envs = 16

           super().__post_init__()
