from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Sized

from datasets import Dataset
from torch.utils.data import Sampler
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from environments.self_supervision.dataset import DEEP_MATH_BAND_ORDER
from environments.self_supervision.trainer import SelfSupervisionGRPOTrainer


logger = logging.getLogger(__name__)

CURRICULUM_STATE_NAME = "curriculum_state.json"


@dataclass(frozen=True)
class CurriculumStageConfig:
    name: str
    sampling_weights: dict[str, float]
    frontier_band: str
    min_stage_steps: int
    promotion_threshold: float


@dataclass(frozen=True)
class CurriculumConfig:
    stages: list[CurriculumStageConfig]
    required_consecutive_evals: int = 2
    eval_examples_per_band: int = 512

    @classmethod
    def default_deepmath(
        cls, *, eval_examples_per_band: int = 512
    ) -> "CurriculumConfig":
        return cls(
            stages=[
                CurriculumStageConfig(
                    name="stage_0",
                    sampling_weights={"B0": 1.0},
                    frontier_band="B0",
                    min_stage_steps=400,
                    promotion_threshold=0.25,
                ),
                CurriculumStageConfig(
                    name="stage_1",
                    sampling_weights={"B1": 0.70, "B0": 0.30},
                    frontier_band="B1",
                    min_stage_steps=600,
                    promotion_threshold=0.18,
                ),
                CurriculumStageConfig(
                    name="stage_2",
                    sampling_weights={"B2": 0.55, "B1": 0.30, "B0": 0.15},
                    frontier_band="B2",
                    min_stage_steps=800,
                    promotion_threshold=0.12,
                ),
                CurriculumStageConfig(
                    name="stage_3",
                    sampling_weights={"B3": 0.45, "B2": 0.30, "B1": 0.15, "B0": 0.10},
                    frontier_band="B3",
                    min_stage_steps=0,
                    promotion_threshold=0.0,
                ),
            ],
            required_consecutive_evals=2,
            eval_examples_per_band=eval_examples_per_band,
        )


class CurriculumController:
    def __init__(
        self,
        config: CurriculumConfig,
        *,
        current_stage_index: int = 0,
        stage_start_step: int = 0,
        consecutive_frontier_successes: int = 0,
    ) -> None:
        self.config = config
        self.current_stage_index = current_stage_index
        self.stage_start_step = stage_start_step
        self.consecutive_frontier_successes = consecutive_frontier_successes

    @property
    def current_stage(self) -> CurriculumStageConfig:
        return self.config.stages[self.current_stage_index]

    @property
    def current_frontier_band(self) -> str:
        return self.current_stage.frontier_band

    def current_frontier_band_index(self) -> int:
        return DEEP_MATH_BAND_ORDER.index(self.current_frontier_band)

    def current_sampling_weights(self) -> dict[str, float]:
        return self.current_stage.sampling_weights

    def record_frontier_eval(self, *, global_step: int, exact_match: float) -> bool:
        if self.current_stage_index >= len(self.config.stages) - 1:
            return False

        stage = self.current_stage
        if global_step - self.stage_start_step < stage.min_stage_steps:
            self.consecutive_frontier_successes = 0
            return False

        if exact_match >= stage.promotion_threshold:
            self.consecutive_frontier_successes += 1
        else:
            self.consecutive_frontier_successes = 0
            return False

        if self.consecutive_frontier_successes < self.config.required_consecutive_evals:
            return False

        self.current_stage_index += 1
        self.stage_start_step = global_step
        self.consecutive_frontier_successes = 0
        logger.info(
            "Promoted curriculum to %s at global step %s.",
            self.current_stage.name,
            global_step,
        )
        return True

    def state_dict(self) -> dict[str, int]:
        return {
            "current_stage_index": self.current_stage_index,
            "stage_start_step": self.stage_start_step,
            "consecutive_frontier_successes": self.consecutive_frontier_successes,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.current_stage_index = int(state_dict["current_stage_index"])
        self.stage_start_step = int(state_dict["stage_start_step"])
        self.consecutive_frontier_successes = int(
            state_dict["consecutive_frontier_successes"]
        )


class CurriculumRepeatSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Sized,
        *,
        band_to_indices: dict[str, list[int]],
        controller: CurriculumController,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self.data_source = data_source
        self.band_to_indices = {
            band: list(indices) for band, indices in band_to_indices.items() if indices
        }
        self.controller = controller
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed
        self._rng = random.Random(seed)

    def __iter__(self):
        band_pools = {
            band: list(indices) for band, indices in self.band_to_indices.items()
        }
        band_positions = {band: 0 for band in band_pools}

        for pool in band_pools.values():
            if self.shuffle and len(pool) > 1:
                self._rng.shuffle(pool)

        num_chunks = self.num_samples // self.batch_size
        for _ in range(num_chunks):
            chunk = self._sample_chunk(band_pools, band_positions)
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def _sample_chunk(
        self,
        band_pools: dict[str, list[int]],
        band_positions: dict[str, int],
    ) -> list[int]:
        chunk = []
        used_indices = set()
        blocked_bands = set()

        while len(chunk) < self.batch_size:
            candidate_bands = [
                band
                for band in self.controller.current_sampling_weights()
                if self.controller.current_sampling_weights()[band] > 0.0
                and band in band_pools
                and band not in blocked_bands
            ]
            if not candidate_bands:
                raise ValueError(
                    "Curriculum sampler could not assemble a full unique chunk for the current stage."
                )

            band = self._choose_band(candidate_bands)
            index = self._take_next_unique_index(
                band,
                used_indices=used_indices,
                band_pools=band_pools,
                band_positions=band_positions,
            )
            if index is None:
                blocked_bands.add(band)
                continue

            used_indices.add(index)
            chunk.append(index)

        return chunk

    def _choose_band(self, candidate_bands: list[str]) -> str:
        weights = [
            self.controller.current_sampling_weights()[band] for band in candidate_bands
        ]
        return self._rng.choices(candidate_bands, weights=weights, k=1)[0]

    def _take_next_unique_index(
        self,
        band: str,
        *,
        used_indices: set[int],
        band_pools: dict[str, list[int]],
        band_positions: dict[str, int],
    ) -> int | None:
        pool = band_pools[band]
        if not pool:
            return None

        for _ in range(len(pool)):
            if band_positions[band] >= len(pool):
                band_positions[band] = 0
                if self.shuffle and len(pool) > 1:
                    self._rng.shuffle(pool)

            index = pool[band_positions[band]]
            band_positions[band] += 1
            if index not in used_indices:
                return index

        return None

    def __len__(self) -> int:
        return (
            (self.num_samples // self.batch_size)
            * self.batch_size
            * self.mini_repeat_count
            * self.repeat_count
        )


class CurriculumCallback(TrainerCallback):
    def __init__(self, trainer: "CurriculumGRPOTrainer") -> None:
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.set_curriculum_frontier_eval_dataset()
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.log_history:
            return control

        exact_match = state.log_history[-1].get("eval_self_reward/exact_match")
        if exact_match is None:
            return control

        self.trainer.curriculum_controller.record_frontier_eval(
            global_step=state.global_step,
            exact_match=float(exact_match),
        )
        self.trainer.set_curriculum_frontier_eval_dataset()
        return control


class CurriculumGRPOTrainer(SelfSupervisionGRPOTrainer):
    def __init__(
        self,
        *,
        curriculum_controller: CurriculumController,
        curriculum_train_band_indices: dict[str, list[int]],
        curriculum_eval_datasets: dict[str, Dataset],
        **kwargs,
    ) -> None:
        self.curriculum_controller = curriculum_controller
        self.curriculum_train_band_indices = {
            band: list(indices)
            for band, indices in curriculum_train_band_indices.items()
        }
        self.curriculum_eval_datasets = curriculum_eval_datasets
        super().__init__(**kwargs)

    def set_curriculum_frontier_eval_dataset(self) -> None:
        self.eval_dataset = self.curriculum_eval_datasets[
            self.curriculum_controller.current_frontier_band
        ]

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler[int]:
        if dataset is None:
            dataset = self.train_dataset
        return CurriculumRepeatSampler(
            data_source=dataset,
            band_to_indices=self.curriculum_train_band_indices,
            controller=self.curriculum_controller,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _init_training_state(
        self,
        max_steps,
        num_update_steps_per_epoch,
        num_train_epochs,
        resume_from_checkpoint,
        trial,
    ) -> tuple[int, int]:
        epochs_trained, steps_trained_in_current_epoch = super()._init_training_state(
            max_steps,
            num_update_steps_per_epoch,
            num_train_epochs,
            resume_from_checkpoint,
            trial,
        )
        if resume_from_checkpoint is not None:
            self._load_curriculum_state(resume_from_checkpoint)
        self.set_curriculum_frontier_eval_dataset()
        return epochs_trained, steps_trained_in_current_epoch

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if not self.args.should_save:
            return

        checkpoint_dir = os.path.join(
            self._get_output_dir(trial=trial),
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )
        self._save_curriculum_state(checkpoint_dir)

    def _save_curriculum_state(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        state_path = os.path.join(output_dir, CURRICULUM_STATE_NAME)
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(
                self.curriculum_controller.state_dict(),
                handle,
                indent=2,
                sort_keys=True,
            )

    def _load_curriculum_state(self, checkpoint_dir: str) -> None:
        state_path = os.path.join(checkpoint_dir, CURRICULUM_STATE_NAME)
        if not os.path.isfile(state_path):
            return

        with open(state_path, encoding="utf-8") as handle:
            self.curriculum_controller.load_state_dict(json.load(handle))

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs = dict(logs)
        prefix = "" if self.model.training else "eval_"
        stage = self.curriculum_controller.current_stage
        sampling_weights = self.curriculum_controller.current_sampling_weights()
        logs[f"{prefix}curriculum/stage"] = (
            self.curriculum_controller.current_stage_index
        )
        logs[f"{prefix}curriculum/frontier_band_index"] = (
            self.curriculum_controller.current_frontier_band_index()
        )
        logs[f"{prefix}curriculum/frontier_threshold"] = stage.promotion_threshold
        for band in DEEP_MATH_BAND_ORDER:
            logs[f"{prefix}curriculum/mix_{band.lower()}"] = sampling_weights.get(
                band,
                0.0,
            )
        super().log(logs, start_time)
