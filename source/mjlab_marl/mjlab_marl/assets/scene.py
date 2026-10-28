"""Native mjlab entities and the original MARLadona field curriculum geometry."""

import math
from functools import partial
from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityCfg

from mjlab_marl.config import FIELD_LEVEL_SETTINGS, FIELD_WIDTH, GOAL_LEVEL_SETTINGS

ASSET_DIR = Path(__file__).parent / "mjcf"
FIELD_SPACING = 10.4 * 0.8


def player_spec(team):
    spec = mujoco.MjSpec.from_file(str(ASSET_DIR / "agent.xml"))
    spec.geom("player").rgba = (0.0, 0.0, 1.0, 1.0) if team == "blue" else (1.0, 0.0, 0.0, 1.0)
    return spec


def ball_spec():
    return mujoco.MjSpec.from_file(str(ASSET_DIR / "ball.xml"))


def field_spec(spec, level_settings, create_visuals):
    """Build the fixed level bank once; each world moves its players between levels."""
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.1),
        rgba=(0.12, 0.18, 0.12, 1.0),
        friction=(1.0, 0.0, 0.0),
    )
    spec.worldbody.add_light(
        name="sun", pos=(0.0, 0.0, 10.0), dir=(0.0, 0.0, -1.0), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    for level, (length_id, goal_id) in enumerate(level_settings):
        length = FIELD_LEVEL_SETTINGS[length_id]
        goal_width = GOAL_LEVEL_SETTINGS[goal_id]
        border_length = length / 9 * 10.4
        border_width = 7.4 * 0.6
        origin = level * FIELD_SPACING
        for sign in (-1, 1):
            spec.worldbody.add_geom(
                name=f"wall_x_{level}_{sign}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(origin + sign * (border_length - 0.01) / 2, 0.0, 0.125),
                size=(0.005, border_width / 2, 0.25),
                rgba=(0.3, 0.3, 0.3, 1.0),
            )
            spec.worldbody.add_geom(
                name=f"wall_y_{level}_{sign}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(origin, sign * (border_width - 0.01) / 2, 0.125),
                size=(border_length / 2, 0.005, 0.25),
                rgba=(0.3, 0.3, 0.3, 1.0),
            )
            for side in (-1, 1):
                spec.worldbody.add_geom(
                    name=f"post_{level}_{sign}_{side}",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=(origin + sign * length / 2, side * goal_width / 2, 0.395),
                    size=(0.049, 0.395),
                    rgba=(1.0, 1.0, 1.0, 1.0),
                )
        if create_visuals:
            add_field_visuals(spec, level, origin, length, goal_width)


def add_field_visuals(spec, level, origin, length, goal_width):
    visual = {"contype": 0, "conaffinity": 0, "group": 2}
    spec.worldbody.add_geom(
        name=f"pitch_{level}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(origin, 0.0, 0.0005),
        size=(length / 2 + 0.1, FIELD_WIDTH / 2 + 0.1, 0.0005),
        rgba=(0.12, 0.42, 0.16, 1.0),
        **visual,
    )
    for side in (-1, 1):
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(origin + side * length / 2, 0, 0.001),
            size=(0.008, FIELD_WIDTH / 2, 0.001),
            rgba=(1, 1, 1, 1),
            **visual,
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(origin, side * FIELD_WIDTH / 2, 0.001),
            size=(length / 2, 0.008, 0.001),
            rgba=(1, 1, 1, 1),
            **visual,
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(origin + side * (length / 2 + 0.06), 0, 0.02),
            size=(0.06, goal_width / 2, 0.12),
            rgba=(1, 0, 0, 1) if side > 0 else (0, 0, 1, 1),
            **visual,
        )
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(origin, 0, 0.001),
        size=(0.008, FIELD_WIDTH / 2, 0.001),
        rgba=(1, 1, 1, 1),
        **visual,
    )
    radius = FIELD_WIDTH / 8
    for i in range(64):
        a, b = i * 2 * math.pi / 64, (i + 1) * 2 * math.pi / 64
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=(0.008, 0),
            fromto=(
                origin + radius * math.cos(a),
                radius * math.sin(a),
                0.001,
                origin + radius * math.cos(b),
                radius * math.sin(b),
                0.001,
            ),
            rgba=(1, 1, 1, 1),
            **visual,
        )


def build_entities(game_cfg):
    entities = {}
    for team in ("blue", "red"):
        for index in range(game_cfg.num_agents_per_team):
            entities[f"agent_{team}_{index}"] = EntityCfg(
                spec_fn=partial(player_spec, team), init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.08))
            )
    entities["ball"] = EntityCfg(spec_fn=ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)))
    return entities


class FieldCurriculum:
    def __init__(self, env):
        self.env = env
        self.max_terrain_level = len(env.cfg.soccer_game.level_settings)
        self.terrain_levels = torch.randint(self.max_terrain_level, (env.num_envs,), device=env.device)
        self._set_origins(torch.arange(env.num_envs, device=env.device))

    def _set_origins(self, env_ids):
        self.env.scene.env_origins[env_ids] = 0
        self.env.scene.env_origins[env_ids, 0] = self.terrain_levels[env_ids] * FIELD_SPACING

    def update_env_origins(self, env_ids, move_up, move_down):
        levels = self.terrain_levels[env_ids] + move_up.long() - move_down.long()
        random_levels = torch.randint(self.max_terrain_level, levels.shape, device=self.env.device)
        self.terrain_levels[env_ids] = torch.where(levels >= self.max_terrain_level, random_levels, levels.clamp(min=0))
        self._set_origins(env_ids)
