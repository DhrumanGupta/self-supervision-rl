from __future__ import annotations

import math
import os
import time

from transformers import Trainer, TrainerCallback
from trl import GRPOTrainer
from trl.trainer import grpo_trainer as trl_grpo_trainer


class SelfSupervisionGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)

        prompt_ids = output.get("prompt_ids")
        prompt_mask = output.get("prompt_mask")
        if prompt_ids is not None and prompt_mask is not None:
            prompt_token_ids = [
                ids[mask.bool()].tolist()
                for ids, mask in zip(prompt_ids, prompt_mask, strict=True)
            ]
            prompts_text = self.processing_class.batch_decode(
                prompt_token_ids,
                skip_special_tokens=False,
            )
            gathered_prompts = trl_grpo_trainer.gather_object(prompts_text)
            for _ in range(len(gathered_prompts)):
                self._logs["prompt"].pop()
            self._logs["prompt"].extend(gathered_prompts)

        return output

    def record_step_metric(
        self,
        name: str,
        value: float,
        *,
        mode: str | None = None,
    ) -> None:
        if mode is None:
            mode = "train" if self.model.training else "eval"
        self._metrics[mode][name].append(float(value))

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {}
        for key, val in self._metrics[mode].items():
            valid = [value for value in val if not math.isnan(value)]
            metrics[key] = sum(valid) / len(valid) if valid else None

        if mode == "eval":
            metrics = {
                (key if key.startswith("profiling/") else f"eval_{key}"): val
                for key, val in metrics.items()
            }

        logs = {**logs, **metrics}
        Trainer.log(self, logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            logging_backends = []
            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and trl_grpo_trainer.wandb.run is not None
            ):
                logging_backends.append(trl_grpo_trainer.wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trl_grpo_trainer.trackio)

            table = {
                "mode": [mode] * len(self._logs["prompt"]),
                "step": [self.state.global_step] * len(self._logs["prompt"]),
                "prompt": self._logs["prompt"],
                "completion": self._logs["completion"],
                **self._logs["rewards"],
                **self._logs["extra"],
                "advantage": self._logs["advantages"],
            }

            df_base = trl_grpo_trainer.pd.DataFrame(table)
            completions_dir = os.path.join(self.args.output_dir, "completions", mode)
            os.makedirs(completions_dir, exist_ok=True)
            df_base.to_parquet(
                os.path.join(
                    completions_dir,
                    f"completions_{self.state.global_step:05d}.parquet",
                )
            )

            images_raw = self._logs["images"] or []

            for logging_backend in logging_backends:
                if images_raw:
                    images = []
                    for image_list in self._logs["images"]:
                        images.append(
                            [logging_backend.Image(image) for image in image_list]
                        )
                    df = trl_grpo_trainer.pd.concat(
                        [df_base, trl_grpo_trainer.pd.Series(images, name="image")],
                        axis=1,
                        copy=False,
                    )
                else:
                    df = df_base

                if self.log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])

                logging_backend.log(
                    {"completions": logging_backend.Table(dataframe=df)}
                )


class ProfilingCallback(TrainerCallback):
    def __init__(self, trainer: SelfSupervisionGRPOTrainer) -> None:
        self.trainer = trainer
        self._step_start_time: float | None = None
        self._optimizer_start_time: float | None = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start_time = time.perf_counter()
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._optimizer_start_time = time.perf_counter()
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._optimizer_start_time is not None:
            self.trainer.record_step_metric(
                "profiling/optimizer/step_s",
                time.perf_counter() - self._optimizer_start_time,
                mode="train",
            )
            self._optimizer_start_time = None
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start_time is not None:
            self.trainer.record_step_metric(
                "profiling/update/total_s",
                time.perf_counter() - self._step_start_time,
                mode="train",
            )
            self._step_start_time = None
        return control
