from __future__ import annotations

import asyncio
import os
from functools import wraps
from typing import Any, Callable

import pandas as pd

from trl.trainer import grpo_trainer as trl_grpo_trainer


def _append_logged_metrics(trainer, metrics: list[tuple[str, float]]) -> None:
    if not metrics:
        return

    mode = "train" if trainer.model.training else "eval"
    for name, value in metrics:
        trainer._metrics[mode][name].append(value)


class CompletionLogger:
    def __init__(self, max_text_chars: int | None = None):
        if max_text_chars is None:
            max_text_chars = int(os.environ.get("GRPO_LOG_TEXT_MAX_CHARS", "0"))
        self.max_text_chars = max_text_chars
        self._reset_step_state()

    def _reset_step_state(self) -> None:
        self._batch = {
            "prompt": [],
            "completion": [],
            "advantages": [],
        }
        self._rewards: dict[str, list[Any]] = {}
        self._extras: dict[str, list[Any]] = {}

    def _truncate_value(self, value: Any) -> Any:
        if self.max_text_chars <= 0:
            return value
        if isinstance(value, str) and len(value) > self.max_text_chars:
            return value[: self.max_text_chars] + "..."
        return value

    def _truncate_values(self, values: list[Any]) -> list[Any]:
        return [self._truncate_value(value) for value in values]

    def has_step_data(self) -> bool:
        return bool(self._batch["prompt"])

    def clear(self) -> None:
        self._reset_step_state()

    def record_batch(
        self,
        *,
        prompts: list[str],
        completions: list[str],
        advantages: list[float],
    ) -> None:
        self._batch["prompt"] = self._truncate_values(list(prompts))
        self._batch["completion"] = self._truncate_values(list(completions))
        self._batch["advantages"] = list(advantages)

    def record_reward_outputs(
        self,
        reward_name: str,
        rewards: list[Any],
        extras: dict[str, list[Any]],
    ) -> None:
        self._rewards[reward_name] = list(rewards)
        for name, values in extras.items():
            key = name if name not in self._extras else f"{reward_name}/{name}"
            self._extras[key] = self._truncate_values(list(values))

    def _validate_lengths(self, step: int, rank: int) -> int:
        row_count = len(self._batch["prompt"])
        lengths = {
            "prompt": row_count,
            "completion": len(self._batch["completion"]),
            "advantages": len(self._batch["advantages"]),
        }
        lengths.update(
            {f"reward:{name}": len(values) for name, values in self._rewards.items()}
        )
        lengths.update(
            {f"extra:{name}": len(values) for name, values in self._extras.items()}
        )

        if len(set(lengths.values())) != 1:
            trl_grpo_trainer.logger.error(
                "Local completion log length mismatch at step %s on rank %s: %s",
                step,
                rank,
                lengths,
            )
            raise ValueError(f"Local completion log length mismatch: {lengths}")
        return row_count

    def build_dataframe(self, *, step: int, rank: int, world_size: int) -> pd.DataFrame:
        row_count = self._validate_lengths(step, rank)
        table = {
            "step": [step] * row_count,
            "rank": [rank] * row_count,
            "world_size": [world_size] * row_count,
            "prompt": list(self._batch["prompt"]),
            "completion": list(self._batch["completion"]),
            "advantage": list(self._batch["advantages"]),
        }

        for key, values in self._rewards.items():
            table[key] = list(values)

        for key, values in self._extras.items():
            table[key] = list(values)

        return pd.DataFrame(table)

    def completion_dir(self, output_dir: str) -> str:
        return os.path.join(output_dir, "completions")

    def rank_completion_path(self, output_dir: str, step: int, rank: int) -> str:
        return os.path.join(
            self.completion_dir(output_dir),
            f"completions_{step:05d}_rank_{rank:02d}.parquet",
        )

    def merged_completion_path(self, output_dir: str, step: int) -> str:
        return os.path.join(
            self.completion_dir(output_dir),
            f"completions_{step:05d}.parquet",
        )

    def write_rank_shard(
        self,
        *,
        output_dir: str,
        step: int,
        rank: int,
        world_size: int,
    ) -> None:
        os.makedirs(self.completion_dir(output_dir), exist_ok=True)
        self.build_dataframe(step=step, rank=rank, world_size=world_size).to_parquet(
            self.rank_completion_path(output_dir, step, rank)
        )

    def merge_step_shards(
        self, *, output_dir: str, step: int, world_size: int
    ) -> pd.DataFrame:
        shard_paths = [
            self.rank_completion_path(output_dir, step, rank)
            for rank in range(world_size)
        ]
        frames = [pd.read_parquet(path) for path in shard_paths]
        merged_df = pd.concat(frames, ignore_index=True, copy=False)
        merged_df.to_parquet(self.merged_completion_path(output_dir, step))
        return merged_df

    def _logging_backends(self, trainer) -> list[Any]:
        logging_backends = []
        wandb = getattr(trl_grpo_trainer, "wandb", None)
        trackio = getattr(trl_grpo_trainer, "trackio", None)

        if trainer.accelerator.is_main_process and (
            trainer.args.report_to
            and "wandb" in trainer.args.report_to
            and wandb is not None
            and wandb.run is not None
        ):
            logging_backends.append(wandb)

        if trainer.accelerator.is_main_process and (
            trainer.args.report_to
            and "trackio" in trainer.args.report_to
            and trackio is not None
        ):
            logging_backends.append(trackio)

        return logging_backends

    def flush(self, trainer) -> None:
        if not self.has_step_data():
            return

        self.write_rank_shard(
            output_dir=trainer.args.output_dir,
            step=trainer.state.global_step,
            rank=trainer.accelerator.process_index,
            world_size=trainer.accelerator.num_processes,
        )
        trainer.accelerator.wait_for_everyone()

        if trainer.accelerator.is_main_process:
            merged_df = self.merge_step_shards(
                output_dir=trainer.args.output_dir,
                step=trainer.state.global_step,
                world_size=trainer.accelerator.num_processes,
            )

            if trl_grpo_trainer.is_rich_available():
                reward_columns = {
                    name: merged_df[name].tolist()
                    for name in trainer.reward_func_names
                    if name in merged_df
                }
                trl_grpo_trainer.print_prompt_completions_sample(
                    merged_df["prompt"].tolist(),
                    merged_df["completion"].tolist(),
                    reward_columns,
                    merged_df["advantage"].tolist(),
                    trainer.state.global_step,
                    trainer.num_completions_to_print,
                )

            for logging_backend in self._logging_backends(trainer):
                df = merged_df
                if trainer.log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                logging_backend.log(
                    {"completions": logging_backend.Table(dataframe=df)}
                )

        trainer.accelerator.wait_for_everyone()
        self.clear()


def wrap_reward_func(
    reward_func: Callable[..., Any],
    *,
    completion_logger: CompletionLogger,
):
    reward_name = reward_func.__name__
    trainer = None

    def bind_trainer(bound_trainer) -> None:
        nonlocal trainer
        trainer = bound_trainer

    if asyncio.iscoroutinefunction(reward_func):

        @wraps(reward_func)
        async def wrapped(*args, **kwargs):
            extra_logs: dict[str, list[Any]] = {}
            metric_logs: list[tuple[str, float]] = []

            def log_extra(name, values):
                extra_logs[name] = list(values)

            def log_metric(name, value):
                metric_logs.append((name, value))

            rewards = await reward_func(
                *args,
                log_extra=log_extra,
                log_metric=log_metric,
                **kwargs,
            )
            if trainer is not None:
                completion_logger.record_reward_outputs(
                    reward_name, rewards, extra_logs
                )
                _append_logged_metrics(trainer, metric_logs)
            return rewards

        return wrapped

    @wraps(reward_func)
    def wrapped(*args, **kwargs):
        extra_logs: dict[str, list[Any]] = {}
        metric_logs: list[tuple[str, float]] = []

        def log_extra(name, values):
            extra_logs[name] = list(values)

        def log_metric(name, value):
            metric_logs.append((name, value))

        rewards = reward_func(
            *args,
            log_extra=log_extra,
            log_metric=log_metric,
            **kwargs,
        )
        if trainer is not None:
            completion_logger.record_reward_outputs(reward_name, rewards, extra_logs)
            _append_logged_metrics(trainer, metric_logs)
        return rewards

    wrapped.bind_trainer = bind_trainer
    return wrapped


class LoggingGRPOTrainer(trl_grpo_trainer.GRPOTrainer):
    def __init__(
        self, *args, completion_logger: CompletionLogger | None = None, **kwargs
    ):
        self.completion_logger = completion_logger or CompletionLogger()
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, *args, **kwargs):
        outputs = super()._generate_and_score_completions(*args, **kwargs)
        self.completion_logger.record_batch(
            prompts=self.processing_class.batch_decode(
                outputs["prompt_ids"], skip_special_tokens=True
            ),
            completions=self.processing_class.batch_decode(
                outputs["completion_ids"], skip_special_tokens=True
            ),
            advantages=outputs["advantages"].tolist(),
        )
        return outputs

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super(trl_grpo_trainer.GRPOTrainer, self).log(logs, start_time)
        self._metrics[mode].clear()

        if self.log_completions:
            self.completion_logger.flush(self)
