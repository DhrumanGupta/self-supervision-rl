from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.train_grpo_self_reward import (
    DEEP_MATH_DATASET_NAME,
    _resolve_vllm_prefix_map,
    patch_vllm_param_name_fix,
    parse_args,
    should_enable_curriculum,
    validate_supported_model_name,
)


class ValidateSupportedModelNameTests(unittest.TestCase):
    def test_qwen3_accepts(self) -> None:
        validate_supported_model_name("Qwen/Qwen3-8B-Base")
        validate_supported_model_name("Qwen/Qwen3-8B-Instruct")

    def test_qwen35_accepts(self) -> None:
        validate_supported_model_name("Qwen/Qwen3.5-9B-Base")
        validate_supported_model_name("Qwen/Qwen3.5-9B-Instruct")

    def test_rejects_others(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Qwen/Qwen3"):
            validate_supported_model_name("meta-llama/Llama-3.1-8B")


class ParseArgsTests(unittest.TestCase):
    def test_num_generations_eval_override(self) -> None:
        with patch.object(sys, "argv", [
            "train_grpo_self_reward.py", "--model_name", "Qwen/Qwen3-8B-Base",
            "--output_dir", "outputs/test", "--num_generations_eval", "4",
        ]):
            args = parse_args()
        self.assertEqual(args.num_generations_eval, 4)

    def test_curriculum_enabled_by_default_for_deepmath(self) -> None:
        with patch.object(sys, "argv", [
            "train_grpo_self_reward.py", "--model_name", "Qwen/Qwen3-8B-Base",
            "--dataset_name", DEEP_MATH_DATASET_NAME, "--output_dir", "outputs/test",
        ]):
            args = parse_args()
        self.assertTrue(should_enable_curriculum(args))

    def test_disable_curriculum_flag_overrides(self) -> None:
        with patch.object(sys, "argv", [
            "train_grpo_self_reward.py", "--model_name", "Qwen/Qwen3-8B-Base",
            "--dataset_name", DEEP_MATH_DATASET_NAME, "--output_dir", "outputs/test",
            "--disable_curriculum",
        ]):
            args = parse_args()
        self.assertFalse(should_enable_curriculum(args))


class PatchVllmParamNameFixTests(unittest.TestCase):
    def test_qwen35_causal_lm(self) -> None:
        trainer = _make_fake_trainer(architectures=["Qwen3_5ForCausalLM"])
        patch_vllm_param_name_fix(trainer)
        self.assertEqual(
            trainer.vllm_generation._fix_param_name_to_vllm("model.layers.0.mlp.down_proj.weight"),
            "language_model.model.layers.0.mlp.down_proj.weight",
        )

    def test_qwen35_conditional_generation(self) -> None:
        trainer = _make_fake_trainer(architectures=["Qwen3_5ForConditionalGeneration"])
        patch_vllm_param_name_fix(trainer)
        self.assertEqual(
            trainer.vllm_generation._fix_param_name_to_vllm("model.language_model.layers.0.self_attn.o_proj.weight"),
            "language_model.model.layers.0.self_attn.o_proj.weight",
        )

    def test_qwen35_from_class_name(self) -> None:
        trainer = _make_fake_trainer(class_name="Qwen3_5ForCausalLM")
        self.assertTrue(patch_vllm_param_name_fix(trainer))

    def test_qwen35_from_model_type(self) -> None:
        trainer = _make_fake_trainer(model_type="qwen3_5_text")
        self.assertTrue(patch_vllm_param_name_fix(trainer))

    def test_qwen3_unaffected(self) -> None:
        trainer = _make_fake_trainer(architectures=["Qwen3ForCausalLM"])
        self.assertFalse(patch_vllm_param_name_fix(trainer))


class ResolveVllmPrefixMapTests(unittest.TestCase):
    def test_resolves_from_nested_peft(self) -> None:
        nested = _make_fake_model(class_name="Qwen3_5ForCausalLM", model_type="qwen3_5_text")
        peft_like_model = _make_fake_model(class_name="PeftModelForCausalLM")
        peft_like_model.model = nested
        peft_like_model.base_model = SimpleNamespace(model=nested)
        peft_like_model.get_base_model = lambda: nested

        prefix_map = _resolve_vllm_prefix_map(peft_like_model)
        self.assertIsNotNone(prefix_map)
        self.assertEqual(prefix_map["model."], "language_model.model.")


def _make_fake_model(class_name="Qwen3ForCausalLM", architectures=None, model_type=None):
    model_class = type(class_name, (), {})
    model = model_class()
    model.config = SimpleNamespace(architectures=architectures, model_type=model_type)
    return model


def _make_fake_trainer(architectures=None, class_name="Qwen3ForCausalLM", model_type=None):
    class _FakeVLLMGeneration:
        @staticmethod
        def _fix_param_name_to_vllm(name: str, extra_prefixes=None) -> str:
            prefixes = ["_checkpoint_wrapped_module."] + list(extra_prefixes or [])
            for prefix in prefixes:
                name = name.replace(prefix, "")
            return name

    model = _make_fake_model(class_name=class_name, architectures=architectures, model_type=model_type)
    vllm_generation = _FakeVLLMGeneration()
    vllm_generation.model = model
    return SimpleNamespace(model=model, vllm_generation=vllm_generation)


if __name__ == "__main__":
    unittest.main()