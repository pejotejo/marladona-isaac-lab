from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn, optim
from torch.amp import GradScaler, autocast

from .config import FastSACConfig
from .fast_sac import Actor, CNNActor, CNNCritic, Critic
from .fast_sac_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    save_params,
)

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def _clone_outputs(fn: Callable[..., Any]) -> Callable[..., Any]:
    """CUDA graph replays can overwrite outputs retained from earlier calls."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return torch.utils._pytree.tree_map_only(
            torch.Tensor,
            torch.Tensor.clone,
            fn(*args, **kwargs),
        )

    return wrapped


class FastSACAgent:
    def __init__(
        self,
        env,
        config: FastSACConfig,
        device: str,
        log_dir: str,
        log_metrics: Callable[[dict[str, float], int], None] | None = None,
    ):
        if config.use_symmetry:
            raise NotImplementedError("Holosoma symmetry augmentation is not ported")

        self.env = env
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.global_step = 0

        self.is_multi_gpu = False
        self.gpu_global_rank = 0
        self.gpu_local_rank = 0
        self.gpu_world_size = 1
        self.is_main_process = True
        self.log_metrics = log_metrics

    def setup(self) -> None:
        args = self.config
        device = self.device
        env = self.env

        actor_obs_keys = args.actor_obs_keys
        critic_obs_keys = args.critic_obs_keys

        self.actor_obs_indices = env.actor_obs_indices
        self.critic_obs_indices = env.critic_obs_indices

        actor_obs_dim = sum(
            self.actor_obs_indices[key]["size"] for key in actor_obs_keys
        )
        critic_obs_dim = sum(
            self.critic_obs_indices[key]["size"] for key in critic_obs_keys
        )

        self.actor_obs_dim = actor_obs_dim
        n_act = env.num_actions

        self.scaler = GradScaler(enabled=args.amp)

        self.obs_normalization = args.obs_normalization
        if args.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(
                shape=actor_obs_dim, device=device
            )
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(
                shape=critic_obs_dim, device=device
            )
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # Get action scaling parameters from the environment
        action_scale = (
            env.action_scale if args.use_tanh else torch.ones(n_act, device=device)
        )
        action_bias = env.action_bias

        # Handle CNN actor/critic
        if args.use_cnn_encoder:
            # We assume that MLP doesn't take raw encoder observations
            actor_mlp_obs_keys = [
                k for k in actor_obs_keys if k != args.encoder_obs_key
            ]
            critic_mlp_obs_keys = [
                k for k in critic_obs_keys if k != args.encoder_obs_key
            ]
        else:
            actor_mlp_obs_keys = list(actor_obs_keys)
            critic_mlp_obs_keys = list(critic_obs_keys)
        actor_cls, critic_cls = (
            (CNNActor, CNNCritic) if args.use_cnn_encoder else (Actor, Critic)
        )

        self.actor = actor_cls(
            obs_indices=self.actor_obs_indices,
            obs_keys=actor_mlp_obs_keys,
            n_act=n_act,
            num_envs=env.num_envs,
            device=device,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=args.log_std_max,
            log_std_min=args.log_std_min,
            use_tanh=args.use_tanh,
            use_layer_norm=args.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )

        print(self.actor)
        print(self.qnet)

        self.log_alpha = torch.tensor(
            [math.log(args.alpha_init)], requires_grad=True, device=device
        )
        self.policy = self.actor.explore

        self.qnet_target = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        self.target_entropy = -n_act * args.target_entropy_ratio
        self.alpha_optimizer = optim.AdamW(
            [self.log_alpha], lr=args.alpha_learning_rate, fused=True, betas=(0.9, 0.95)
        )

        logger.info(f"actor_obs_dim: {actor_obs_dim}, critic_obs_dim: {critic_obs_dim}")

        self.rb = SimpleReplayBuffer(
            n_env=env.num_envs,
            buffer_size=args.buffer_size,
            n_obs=actor_obs_dim,
            n_act=n_act,
            n_critic_obs=critic_obs_dim,
            n_steps=args.num_steps,
            gamma=args.gamma,
            device=device,
        )

    @contextmanager
    def _maybe_amp(self) -> Iterator[None]:
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    def _synchronize_model_parameters(self) -> None:
        """Synchronize actor, qnet, and log_alpha parameters across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast qnet weights from rank 0 to all other ranks
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast log_alpha parameter from rank 0 to all other ranks
        torch.distributed.broadcast(self.log_alpha.data, src=0)

        # Load qnet_target weights from synced qnet
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        logger.info(f"Synchronized model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        """Batches and all-reduces gradients across GPUs to reduce NCCL call count.

        This flattens all existing parameter gradients into a single contiguous
        tensor, performs one all_reduce, averages by world size, and then
        scatters the reduced values back into the original gradient tensors.
        """
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    def _update_main(
        self, data: TensorDict
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        args = self.config

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = (
                    actor.get_actions_and_log_probs(next_observations)
                )
                discount = args.gamma ** data["next"]["effective_n_steps"]

                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards
                    - discount
                    * bootstrap
                    * self.log_alpha.exp()
                    * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(qnet)

        scaler.unscale_(q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (
                    -self.log_alpha.exp()
                    * (next_state_log_probs.detach() + self.target_entropy)
                ).mean()

            scaler.scale(alpha_loss).backward()

            if self.is_multi_gpu:
                if self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(
                        self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM
                    )
                    self.log_alpha.grad.data.copy_(
                        self.log_alpha.grad.data / self.gpu_world_size
                    )

            scaler.unscale_(alpha_optimizer)

            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
        )

    def _update_pol(
        self, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = self.actor
        qnet = self.qnet
        actor_optimizer = self.actor_optimizer
        scaler = self.scaler
        args = self.config

        with self._maybe_amp():
            critic_observations = data["critic_observations"]

            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            # For logging, this is a bit wasteful though, but could be useful
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                # Compute policy entropy (negative log probability)
                policy_entropy = -log_probs.mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(actor)

        scaler.unscale_(actor_optimizer)

        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _sample_and_prepare_batches(
        self,
        batch_size: int,
        num_updates: int,
        normalize_obs: Callable[..., torch.Tensor],
        normalize_critic_obs: Callable[..., torch.Tensor],
    ) -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size * num_updates
        large_data = self.rb.sample(large_batch_size)
        samples_per_update = batch_size * self.env.num_envs

        if self.config.use_symmetry:
            samples_per_update *= 2

            augmented_large_data: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
                "next": {}
            }

            augmented_large_data["observations"] = (
                self.symmetry_utils.augment_observations(
                    obs=large_data["observations"],
                    env=self.env,
                    obs_list=self.config.actor_obs_keys,
                )
            )
            augmented_large_data["actions"] = self.symmetry_utils.augment_actions(
                actions=large_data["actions"]
            )
            assert isinstance(augmented_large_data["next"], dict)
            augmented_large_data["next"]["observations"] = (
                self.symmetry_utils.augment_observations(
                    obs=large_data["next"]["observations"],
                    env=self.env,
                    obs_list=self.config.actor_obs_keys,
                )
            )
            augmented_large_data["critic_observations"] = (
                self.symmetry_utils.augment_observations(
                    obs=large_data["critic_observations"],
                    env=self.env,
                    obs_list=self.config.critic_obs_keys,
                )
            )
            augmented_large_data["next"]["critic_observations"] = (
                self.symmetry_utils.augment_observations(
                    obs=large_data["next"]["critic_observations"],
                    env=self.env,
                    obs_list=self.config.critic_obs_keys,
                )
            )

            # Calculate augmentation factor and repeat non-augmented data
            observations_tensor = augmented_large_data["observations"]
            assert isinstance(observations_tensor, torch.Tensor), (
                "observations should be a Tensor after data augmentation"
            )
            num_aug = int(
                observations_tensor.shape[0] / large_data["next"]["rewards"].shape[0]
            )
            augmented_large_data["next"]["rewards"] = large_data["next"][
                "rewards"
            ].repeat(num_aug)
            augmented_large_data["next"]["dones"] = large_data["next"]["dones"].repeat(
                num_aug
            )
            augmented_large_data["next"]["truncations"] = large_data["next"][
                "truncations"
            ].repeat(num_aug)
            augmented_large_data["next"]["effective_n_steps"] = large_data["next"][
                "effective_n_steps"
            ].repeat(num_aug)

            # Override large_data
            large_data = augmented_large_data

        # Normalize all data once
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(
            large_data["next"]["observations"]
        )
        large_data["critic_observations"] = normalize_critic_obs(
            large_data["critic_observations"]
        )
        large_data["next"]["critic_observations"] = normalize_critic_obs(
            large_data["next"]["critic_observations"]
        )

        # Split into smaller batches
        prepared_batches = []

        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            # Create a slice of the large batch
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][
                            start_idx:end_idx
                        ],
                        "observations": large_data["next"]["observations"][
                            start_idx:end_idx
                        ],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][
                            start_idx:end_idx
                        ],
                    },
                    "critic_observations": large_data["critic_observations"][
                        start_idx:end_idx
                    ],
                },
                batch_size=samples_per_update,
            )
            batch_data["next"]["critic_observations"] = large_data["next"][
                "critic_observations"
            ][start_idx:end_idx]

            prepared_batches.append(batch_data)

        return prepared_batches

    def save(self, path: str) -> None:
        save_params(
            self.global_step,
            self.actor,
            self.qnet,
            self.qnet_target,
            self.log_alpha,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.actor_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.scaler,
            self.config,
            path,
            metadata={
                "algorithm": "fast_sac",
                "actor_obs_indices": self.actor_obs_indices,
                "critic_obs_indices": self.critic_obs_indices,
            },
        )

    def learn(self) -> None:
        args = self.config
        device = self.device
        uses_cudagraphs = False

        if args.compile:
            mode, options = args.compile_mode, args.compile_options

            if mode == "auto":
                mode = (
                    "reduce-overhead" if self.is_multi_gpu and options is None else None
                )

            if mode is not None and options is not None:
                raise ValueError("Set at most one of compile_mode and compile_options")

            logger.info(
                "torch.compile mode=%s options=%s",
                mode or "default",
                options,
            )

            if mode is not None:
                uses_cudagraphs = bool(
                    torch._inductor.list_mode_options(mode).get(
                        "triton.cudagraphs", False
                    )
                )
            elif options is not None:
                uses_cudagraphs = bool(options.get("triton.cudagraphs", False))

            update_main = torch.compile(self._update_main, mode=mode, options=options)
            update_pol = torch.compile(self._update_pol, mode=mode, options=options)
            policy = torch.compile(self.policy, mode=mode, options=options)
            normalize_obs = torch.compile(
                self.obs_normalizer.forward, mode=mode, options=options
            )
            normalize_critic_obs = torch.compile(
                self.critic_obs_normalizer.forward,
                mode=mode,
                options=options,
            )

            if uses_cudagraphs:
                (
                    update_main,
                    update_pol,
                    policy,
                    normalize_obs,
                    normalize_critic_obs,
                ) = (
                    _clone_outputs(fn)
                    for fn in (
                        update_main,
                        update_pol,
                        policy,
                        normalize_obs,
                        normalize_critic_obs,
                    )
                )
        else:
            update_main = self._update_main
            update_pol = self._update_pol
            policy = self.policy
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward

        qnet = self.qnet
        qnet_target = self.qnet_target
        env = self.env
        rb = self.rb

        obs, critic_obs = env.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
        dones = None

        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)

        metric_sums = {}
        metric_count = 0

        while self.global_step <= args.num_learning_iterations:
            if uses_cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()

            with torch.no_grad(), self._maybe_amp():
                norm_obs = (
                    normalize_obs(obs, update=False) if self.obs_normalization else obs
                )
                actions = policy(obs=norm_obs, dones=dones)

            next_obs, rewards, dones, infos = env.step(actions.float())
            truncations = infos["time_outs"]
            next_critic_obs = infos["observations"]["critic"]

            true_next_obs = torch.where(
                truncations[:, None] > 0,
                infos["observations"]["final"]["actor_obs"],
                next_obs,
            )
            true_next_critic_obs = torch.where(
                truncations[:, None] > 0,
                infos["observations"]["final"]["critic_obs"],
                next_critic_obs,
            )

            transition = TensorDict(
                {
                    "observations": obs,
                    "actions": torch.as_tensor(
                        actions, device=device, dtype=torch.float
                    ),
                    "next": {
                        "observations": true_next_obs,
                        "rewards": torch.as_tensor(
                            rewards, device=device, dtype=torch.float
                        ),
                        "truncations": truncations.long(),
                        "dones": dones.long(),
                    },
                },
                batch_size=(env.num_envs,),
                device=device,
            )
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs

            obs = next_obs
            critic_obs = next_critic_obs
            rb.extend(transition)

            batch_size = max(
                args.batch_size // env.num_envs // self.gpu_world_size,
                1,
            )

            if self.global_step > args.learning_starts:
                prepared_batches = self._sample_and_prepare_batches(
                    batch_size,
                    args.num_updates,
                    normalize_obs,
                    normalize_critic_obs,
                )

                for i, data in enumerate(prepared_batches):
                    (
                        buffer_rewards,
                        critic_grad_norm,
                        qf_loss,
                        qf_max,
                        qf_min,
                        alpha_loss,
                    ) = update_main(data)

                    if args.num_updates > 1:
                        if i % args.policy_frequency == 1:
                            (
                                actor_grad_norm,
                                actor_loss,
                                policy_entropy,
                                action_std,
                            ) = update_pol(data)
                    elif self.global_step % args.policy_frequency == 0:
                        (
                            actor_grad_norm,
                            actor_loss,
                            policy_entropy,
                            action_std,
                        ) = update_pol(data)

                    current_metrics = {
                        "actor_loss": actor_loss,
                        "qf_loss": qf_loss,
                        "qf_max": qf_max,
                        "qf_min": qf_min,
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                        "buffer_rewards": buffer_rewards,
                        "alpha_loss": alpha_loss,
                        "alpha_value": self.log_alpha.exp().detach().mean(),
                        "policy_entropy": policy_entropy,
                        "action_std": action_std,
                    }
                    for key, value in current_metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + value
                    metric_count += 1

                    with torch.no_grad():
                        src_ps = [p.data for p in qnet.parameters()]
                        tgt_ps = [p.data for p in qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)

                if self.global_step % args.logging_interval == 0:
                    loss_dict = {
                        key: (value / metric_count).item()
                        for key, value in metric_sums.items()
                    }
                    loss_dict["env_rewards"] = rewards.mean().item()
                    print(
                        f"\n[FastSAC] Step {self.global_step:,} / "
                        f"{args.num_learning_iterations:,}\n"
                        f"  Mean step reward: {loss_dict['env_rewards']:.4f}\n"
                        f"  Critic loss: {loss_dict['qf_loss']:.4f} | "
                        f"Actor loss: {loss_dict['actor_loss']:.4f}\n"
                        f"  Alpha: {loss_dict['alpha_value']:.4g} | "
                        f"Entropy: {loss_dict['policy_entropy']:.4f}",
                        flush=True,
                    )
                    if self.log_metrics is not None:
                        self.log_metrics(loss_dict, self.global_step)

                    metric_sums.clear()
                    metric_count = 0

                if (
                    args.save_interval > 0
                    and self.global_step > 0
                    and self.global_step % args.save_interval == 0
                ):
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{self.global_step:07d}.pt",
                        )
                    )

            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1

        self.save(
            os.path.join(
                self.log_dir,
                f"model_{self.global_step:07d}.pt",
            )
        )

        if uses_cudagraphs:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
            torch.compiler.reset()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.qnet.load_state_dict(checkpoint["qnet_state_dict"])
        self.qnet_target.load_state_dict(checkpoint["qnet_target_state_dict"])
        self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        self.critic_obs_normalizer.load_state_dict(
            checkpoint["critic_obs_normalizer_state"]
        )

        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        self.global_step = checkpoint["global_step"]

    def get_inference_policy(self):
        self.actor.eval()
        self.obs_normalizer.eval()

        @torch.no_grad()
        def policy(obs):
            if self.obs_normalization:
                obs = self.obs_normalizer(obs, update=False)
            return self.actor(obs)[0]

        return policy
