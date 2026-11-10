import torch
from isaaclab.utils import configclass

from isaaclab_marl.config import APPLY_RANDOMIZATION, KICK_MODEL


@configclass
class SoccerBallCfg:
    ownership_radius = 0.25
    reset_type = "no_speed"  # "no_speed" or "keep_speed"
    randomize_friction = True

    if APPLY_RANDOMIZATION:
        roll_friction_range = (0.00006, 0.0001)
        friction_range = (0.0, 0.5)
    else:
        roll_friction_range = (0.00005, 0.00005)
        friction_range = (0.25, 0.25)

    if KICK_MODEL == "default":
        kick_manipulation_type = "force"  # "force" or "velocity"
    else:
        kick_manipulation_type = "velocity"

    kick_force_gain = 0.5
    kick_area = "default"  # default, fullcircle
    kick_margin = 0.2
    max_kick_speed = 1.0


@configclass
class SoccerAgentsCfg:
    if APPLY_RANDOMIZATION:
        friction_range = (0.0, 0.2)
    else:
        friction_range = (0.1, 0.1)

    translation_shape = "ellipse"  # "ellipse" or "square"

    position_control_pid_gain = [20.0, 0.2]
    velocity_control_pid_gain = [20.0, 0.2]
    velocity_p_gain = 3
    translation_force_d_gain = 10
    rotation_torque_d_gain = 0.2

    cmd_speed_limit = [
        [0.35, -0.25],
        [0.25, -0.25],
        [2.5, -2.5],
    ]  # action limit 0, 1, 2
    kick_timeout = 10.0

    kick_position_gain = 2.0
    kick_rotation_gain = 2.0
    kick_route_radius = 0.35
    kick_waypoint_step = torch.pi / 4
    kick_rear_half_angle = torch.pi / 4
    kick_stand_off_ratio = 0.8
    kick_lateral_tolerance = 0.04
    kick_yaw_tolerance = 0.12
    kick_min_target_distance = 0.001


@configclass
class SoccerBotsCfg:
    circle_around_distance = 0.35
    alignment_angle_offset = 1.5

    goto_position_gain = 10.0
    player_rotation_gain = 10.0
    ball_prediction_steps = 25
    circle_stand_off = 0.25
    circle_forward_gain = 10.0
    circle_lateral_gain = 3.0

    keeper_rotation_gain = 3.0
    keeper_prediction_distance = 1.0
    keeper_blend_min_distance = 0.3
    keeper_blend_distance = 2.0
    keeper_goal_offset = 0.2
    keeper_clear_ball_x_min = 0.5
    keeper_clear_forward_distance = 0.5

    kick_strength = 1.0


@configclass
class SoccerSpawnCfg:
    random_sampling = True
    mirror_init_pos = False
    init_rot_noise = torch.pi


@configclass
class SoccerFieldCfg:
    scale: float = 0.6

    length_levels: list[float] = [3.6, 5.4, 7.2, 9.0]
    goal_width_levels: list[float] = [6.0, 4.6, 3.4, 2.4, 1.6]

    width: float = 6.0
    outer_width: float = 7.4
    tile_size: tuple[float, float] = (8.32, 5.92)

    field_asset_length: float = 9.0
    field_asset_width: float = 6.0
    goal_asset_width: float = 1.6


@configclass
class SoccerGameCfg:
    num_teams: int = 2  # 2 teams
    num_agents_per_team: int = 3  # max 5 players per team
    num_actions = 7

    # 0 for no control, 1 for policy, 2 for policy replay, 3 for bots
    default_control = [1, 2]  # blue, red agents
    agent_id_control = [
        [1, 3, 3],
        [3, 3, 3],
    ]  # [[1, 1 ... ],[2, 2 ...]] default if not set

    eval_env_ratio = 0.15

    num_agents_range = [1, num_agents_per_team]

    level_settings = [[2, 2], [3, 2], [3, 3], [3, 4]]

    ball = SoccerBallCfg()
    agents = SoccerAgentsCfg()
    bots = SoccerBotsCfg()
    spawn = SoccerSpawnCfg()
    field: SoccerFieldCfg = SoccerFieldCfg()


    create_field_visuals = False

    def __post_init__(self):
        if not self.agent_id_control:
            self.agent_id_control = [
                [control for _ in range(self.num_agents_per_team)]
                for control in self.default_control
            ]
