from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.self_supervision.rollout import _extract_completion_ids


class ExtractCompletionIdsTests(unittest.TestCase):
    def test_trims_completion_ids_at_eos_before_batch_padding(self) -> None:
        input_ids = torch.tensor([[11, 12, 13], [21, 22, 23]])
        generated_ids = torch.tensor(
            [
                [11, 12, 13, 101, 102, 99, 0, 0],
                [21, 22, 23, 201, 99, 0, 0, 0],
            ]
        )

        completion_ids = _extract_completion_ids(
            generated_ids,
            input_ids,
            eos_token_id=99,
            pad_token_id=0,
        )

        self.assertEqual(completion_ids, [[101, 102], [201]])

    def test_trims_completion_ids_at_pad_when_no_eos_is_present(self) -> None:
        input_ids = torch.tensor([[11, 12], [21, 22]])
        generated_ids = torch.tensor(
            [
                [11, 12, 101, 102, 0, 0],
                [21, 22, 201, 202, 203, 0],
            ]
        )

        completion_ids = _extract_completion_ids(
            generated_ids,
            input_ids,
            eos_token_id=99,
            pad_token_id=0,
        )

        self.assertEqual(completion_ids, [[101, 102], [201, 202, 203]])


if __name__ == "__main__":
    unittest.main()
