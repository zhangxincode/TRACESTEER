from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from our_model.our_model_otherdata import eval_feedback_flow_otherdata as evaluation
from our_model.our_model_otherdata import train_feedback_flow_otherdata as training


class TupleLayer(nn.Module):
    def forward(self, hidden):
        return hidden, None


class TinyDream(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="Dream",
            hidden_size=4,
            num_hidden_layers=2,
            mask_token_id=15,
            _name_or_path="/models/dream-Instruct",
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TupleLayer(), TupleLayer()])
        self.anchor = nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, attention_mask=None, position_ids=None, **_kwargs):
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        logits = torch.zeros(*input_ids.shape, 16, device=input_ids.device)
        logits.scatter_(-1, input_ids.unsqueeze(-1), 10.0)
        return SimpleNamespace(logits=logits)


class RecordingController:
    def __init__(self):
        self.response_mask = None
        self.previous_entropy = None

    def prepare_feedback(self, _feedback_ids, **kwargs):
        self.response_mask = kwargs["response_mask"].clone()
        self.previous_entropy = kwargs["previous_entropy"].clone()


class OtherdataBackboneTests(unittest.TestCase):
    def test_train_and_eval_parsers_select_dream(self):
        with mock.patch.object(sys, "argv", ["train", "--dataset", "copa", "--backbone", "dream"]):
            train_args = training.parse_args()
        with mock.patch.object(
            sys,
            "argv",
            ["eval", "--dataset", "copa", "--flow-path", "flow.pt", "--backbone", "dream"],
        ):
            eval_args = evaluation.parse_args()
        self.assertEqual(train_args.backbone, "dream")
        self.assertEqual(eval_args.backbone, "dream")
        self.assertTrue(train_args.model_dir.endswith("dream-Instruct"))
        self.assertTrue(eval_args.model_dir.endswith("dream-Instruct"))

    def test_previous_prediction_uses_dream_token_alignment(self):
        model = TinyDream()
        input_ids = torch.tensor([[2, 4, 6]])
        attention_mask = torch.ones_like(input_ids)
        prediction, _entropy = training.previous_prediction(model, input_ids, attention_mask)
        self.assertTrue(torch.equal(prediction, torch.tensor([[2, 2, 4]])))

    def test_training_feedback_uses_dream_prediction_positions(self):
        model = TinyDream()
        controller = RecordingController()
        batch = {
            "input_ids": torch.tensor([[2, 15, 15]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "response_mask": torch.tensor([[False, True, True]]),
            "timestep": torch.tensor([0.0]),
        }
        training.prepare_feedback(controller, model, batch, mask_id=15)
        self.assertTrue(
            torch.equal(controller.response_mask, torch.tensor([[True, True, False]]))
        )
        _prediction, token_entropy = training.previous_prediction(
            model, batch["input_ids"], batch["attention_mask"]
        )
        expected_entropy = torch.zeros_like(token_entropy)
        expected_entropy[:, :-1] = token_entropy[:, 1:]
        self.assertTrue(torch.equal(controller.previous_entropy, expected_entropy))

    def test_eval_rejects_llada_artifact_for_dream(self):
        model = TinyDream()
        payload = {"metadata": {"backbone": "llada", "model_type": "llada"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluation.validate_artifact_backbone(model, path)

    def test_default_outputs_are_separated_by_backbone(self):
        self.assertEqual(training.output_dir_for_backbone("llada"), training.DEFAULT_OUTPUT_DIR)
        self.assertEqual(
            training.output_dir_for_backbone("dream"), training.DEFAULT_OUTPUT_DIR / "dream"
        )


if __name__ == "__main__":
    unittest.main()
