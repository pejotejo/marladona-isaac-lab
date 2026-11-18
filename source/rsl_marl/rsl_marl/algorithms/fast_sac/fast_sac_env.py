from __future__ import annotations

import torch
from rsl_marl.modules.bots import Bots


class FastSACEnv:
    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.num_envs = self.unwrapped.num_envs

        game = self.unwrapped.cfg.soccer_game
        data = self.unwrapped.env_data

        self.num_actions = data.num_actions
        self._players_per_env = data.num_agents_per_env

        full_roster = (
            game.num_agents_per_team,
            game.num_agents_per_team,
        )
        if tuple(game.num_agents_range) != full_roster or game.eval_env_ratio != 0.0:
            raise ValueError(
                "FastSAC currently requires fixed, fully active rosters "
                "and no separate evaluation matches"
            )

        learner_mask = data.training_actor_index.reshape(
            self.num_envs, self._players_per_env
        )
        bot_mask = data.bots_actor_index.reshape(self.num_envs, self._players_per_env)

        if not (learner_mask.sum(dim=1) == 1).all():
            raise ValueError("FastSAC currently requires exactly one learner per match")

        if not (learner_mask | bot_mask).all():
            raise ValueError("Every robot must be controlled by the learner or a bot")

        self._learner_ids = data.training_actor_index.nonzero(as_tuple=False).flatten()
        self._bot_ids = data.bots_actor_index.nonzero(as_tuple=False).flatten()

        self.bot_policy = Bots(env, self.device)
        self._bot_roles = ["keeper"] + ["player"] * (game.num_agents_per_team - 1)

        self._actions = torch.zeros(
            (
                self.num_envs * self._players_per_env,
                self.num_actions,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._obs_dict = None

        dimensions = self.unwrapped.observation_manager.group_obs_dim
        actor_dim = dimensions["policy"][0] + dimensions["neighbor"][0]
        critic_dim = dimensions["critic"][0] + dimensions["neighbor_critic"][0]

        self.actor_obs_indices = {
            "actor_obs": {"start": 0, "end": actor_dim, "size": actor_dim}
        }
        self.critic_obs_indices = {
            "critic_obs": {"start": 0, "end": critic_dim, "size": critic_dim}
        }

        self.action_scale = torch.tensor(
            [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5],
            dtype=torch.float32,
            device=self.device,
        )
        self.action_bias = torch.tensor(
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],
            dtype=torch.float32,
            device=self.device,
        )

    def _get_observations(self, obs_dict):
        ids = self._learner_ids

        actor_obs = torch.cat(
            (obs_dict["policy"][ids], obs_dict["neighbor"][ids]),
            dim=-1,
        )
        critic_obs = torch.cat(
            (
                obs_dict["critic"][ids],
                obs_dict["neighbor_critic"][ids],
            ),
            dim=-1,
        )
        return actor_obs, critic_obs

    @torch.no_grad()
    def reset_with_critic_obs(self):
        self._actions.zero_()
        self._obs_dict, _ = self.env.reset()
        return self._get_observations(self._obs_dict)

    def reset(self):
        actor_obs, _ = self.reset_with_critic_obs()
        return actor_obs

    @torch.no_grad()
    def step(self, actions):
        if self._obs_dict is None:
            raise RuntimeError("Reset the environment before stepping")

        expected = (self.num_envs, self.num_actions)
        if actions.shape != expected:
            raise ValueError(
                f"Expected actions shaped {expected}, got {tuple(actions.shape)}"
            )

        self._actions[self._learner_ids] = actions
        self._actions[self._bot_ids] = self.bot_policy.act(
            self._obs_dict["bots"],
            self._actions,
            self._bot_ids,
            configs=self._bot_roles,
        )

        obs_dict, rewards, terminated, truncated, info = self.env.step(self._actions)

        if "final_observations" not in info:
            raise RuntimeError(
                "The soccer environment needs pre-reset "
                "final-observation support before FastSAC training"
            )

        self._obs_dict = obs_dict
        actor_obs, critic_obs = self._get_observations(obs_dict)

        terminated = terminated[self._learner_ids].bool()
        truncated = truncated[self._learner_ids].bool()

        dones = terminated | truncated
        time_outs = truncated & ~terminated

        final_dict = info["final_observations"]
        if final_dict is None:
            if time_outs.any():
                raise RuntimeError("Timeout occurred without final observations")
            final_actor_obs, final_critic_obs = actor_obs, critic_obs
        else:
            final_actor_obs, final_critic_obs = self._get_observations(final_dict)

        self._actions.view(
            self.num_envs,
            self._players_per_env,
            self.num_actions,
        )[dones] = 0.0

        extras = dict(info)
        extras["time_outs"] = time_outs
        extras["observations"] = {
            "critic": critic_obs,
            "final": {
                "actor_obs": final_actor_obs,
                "critic_obs": final_critic_obs,
            },
        }

        return (
            actor_obs,
            rewards[self._learner_ids],
            dones,
            extras,
        )

    def close(self):
        self.env.close()
