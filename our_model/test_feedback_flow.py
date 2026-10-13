from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from our_model.feedback_flow import (  # noqa: E402
    FEEDBACK_FLOW_ARTIFACT_KIND,
    FeedbackFlowBlock,
    FeedbackFlowController,
    TimestepResidualGate,
    attach_feedback_flow,
    clear_feedback_flow,
    feedback_flow_context,
    forward_with_feedback,
    load_feedback_flow_block,
    normalized_token_entropy,
    save_feedback_flow_artifact,
)
from our_model.generate_feedback import generate_with_feedback, previous_prediction_feedback_ids  # noqa: E402
from our_model.losses import FeedbackFlowLossWeights, feedback_flow_loss  # noqa: E402
from our_model.train_feedback_flow import prepare_feedback, sample_timestep  # noqa: E402


class AddBlock(nn.Module):
    def __init__(self, amount: float = 0.0):
        super().__init__()
        self.amount = amount

    def forward(self, hidden, **_kwargs):
        return hidden + self.amount, None


class DummyCore(nn.Module):
    def __init__(self, hidden_size: int = 4, amounts=(0.0, 0.0)):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList([AddBlock(amount) for amount in amounts])
        self.anchor = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden):
        for block in self.transformer.blocks:
            hidden, _ = block(hidden)
        return hidden


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyCore(hidden_size=4, amounts=(0.0, 0.0))
        self.config = SimpleNamespace(
            d_model=4,
            n_layers=2,
            model_type="llada",
            _name_or_path="/models/toy-llada",
            mask_token_id=9,
        )

    @property
    def device(self):
        return self.model.anchor.device

    def forward(self, input_ids, attention_mask=None, **_kwargs):
        hidden = F.one_hot(input_ids.remainder(4), num_classes=4).float()
        hidden = self.model(hidden)
        logits = torch.zeros(*hidden.shape[:2], 10, device=hidden.device)
        logits[..., 1] = 1.0 + hidden[..., 0]
        return SimpleNamespace(logits=logits)


class MeanFeedbackFlow(nn.Module):
    def forward(self, hidden, feedback, hidden_attention_mask=None, feedback_attention_mask=None):
        if feedback_attention_mask is None:
            pooled = feedback.mean(dim=1, keepdim=True)
        else:
            weights = feedback_attention_mask.to(feedback.device, dtype=feedback.dtype).unsqueeze(-1)
            pooled = (feedback * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return hidden + pooled


class LinearTimestepGate(nn.Module):
    def forward(self, timestep, batch_size, device, dtype):
        timestep = torch.as_tensor(timestep, device=device, dtype=torch.float32).view(batch_size)
        return (timestep / 10.0).to(dtype=dtype).view(batch_size, 1, 1)


class GatedMeanFeedbackFlow(MeanFeedbackFlow):
    def __init__(self):
        super().__init__()
        self.timestep_gate = LinearTimestepGate()


class EntropyGatedMeanFeedbackFlow(MeanFeedbackFlow):
    def __init__(self):
        super().__init__()
        self.timestep_gate = TimestepResidualGate(hidden_size=4)


class FeedbackFlowTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "model"):
            clear_feedback_flow(self.model)

    def test_feedback_ids_replace_only_still_masked_response_positions(self):
        current = torch.tensor([[1, 9, 2, 9]])
        previous = torch.tensor([[4, 5, 6, 7]])
        response_mask = torch.tensor([[False, True, True, True]])
        feedback = previous_prediction_feedback_ids(current, previous, response_mask, mask_id=9)
        self.assertTrue(torch.equal(feedback, torch.tensor([[1, 5, 2, 7]])))

    def test_hook_applies_flow_only_on_response_mask(self):
        self.model = ToyModel()
        input_ids = torch.tensor([[0, 1, 2]])
        baseline = self.model(input_ids).logits
        attach_feedback_flow(self.model, flow_block=MeanFeedbackFlow(), layer_idx=0, feedback_layer_idx=0, scale=2.0)
        controller = FeedbackFlowController(self.model)
        response_mask = torch.tensor([[False, True, True]])
        controller.prepare_feedback(
            torch.tensor([[0, 0, 0]]),
            feedback_attention_mask=torch.ones(1, 3, dtype=torch.long),
            response_mask=response_mask,
            conditional_batch_size=1,
        )
        steered = self.model(input_ids).logits
        self.assertEqual(steered[0, 0, 1].item(), baseline[0, 0, 1].item())
        self.assertAlmostEqual(steered[0, 1, 1].item() - baseline[0, 1, 1].item(), 2.0)
        self.assertAlmostEqual(steered[0, 2, 1].item() - baseline[0, 2, 1].item(), 2.0)

    def test_timestep_gate_controls_residual_strength(self):
        self.model = ToyModel()
        input_ids = torch.tensor([[0, 1, 2]])
        baseline = self.model(input_ids).logits
        attach_feedback_flow(self.model, flow_block=GatedMeanFeedbackFlow(), layer_idx=0, feedback_layer_idx=0, scale=9.0)
        controller = FeedbackFlowController(self.model)
        response_mask = torch.tensor([[False, True, True]])

        controller.prepare_feedback(
            torch.tensor([[0, 0, 0]]),
            feedback_attention_mask=torch.ones(1, 3, dtype=torch.long),
            response_mask=response_mask,
            timestep=torch.tensor([2.0]),
            conditional_batch_size=1,
        )
        steered_t2 = self.model(input_ids).logits
        state = self.model._feedback_flow_state
        self.assertAlmostEqual(state.last_gate.item(), 1.8, places=6)
        self.assertAlmostEqual(steered_t2[0, 1, 1].item() - baseline[0, 1, 1].item(), 1.8, places=6)

        controller.prepare_feedback(
            torch.tensor([[0, 0, 0]]),
            feedback_attention_mask=torch.ones(1, 3, dtype=torch.long),
            response_mask=response_mask,
            timestep=torch.tensor([7.0]),
            conditional_batch_size=1,
        )
        steered_t7 = self.model(input_ids).logits
        self.assertAlmostEqual(state.last_gate.item(), 6.3, places=6)
        self.assertAlmostEqual(steered_t7[0, 1, 1].item() - baseline[0, 1, 1].item(), 6.3, places=6)

    def test_feedback_flow_loss_combines_generation_and_stability(self):
        self.model = ToyModel()
        attach_feedback_flow(self.model, flow_block=MeanFeedbackFlow(), layer_idx=0, feedback_layer_idx=0, scale=1.0)
        controller = FeedbackFlowController(self.model)
        input_ids = torch.tensor([[0, 1, 9]])
        response_mask = torch.tensor([[False, False, True]])
        controller.prepare_feedback(
            torch.tensor([[0, 0, 0]]),
            feedback_attention_mask=torch.ones(1, 3, dtype=torch.long),
            response_mask=response_mask,
            conditional_batch_size=1,
        )
        output = self.model(input_ids)
        labels = torch.tensor([[-100, -100, 1]])
        loss, metrics = feedback_flow_loss(
            output.logits,
            labels,
            state=self.model._feedback_flow_state,
            response_mask=response_mask,
            weights=FeedbackFlowLossWeights(stability=0.1),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("generation_loss", metrics)
        self.assertIn("stability_loss", metrics)

    def test_version_one_artifact_loads_with_default_timestep_gate(self):
        block = FeedbackFlowBlock(hidden_size=4, num_heads=1, steps=1, ff_multiplier=1.0)
        old_state_dict = {
            key: value
            for key, value in block.state_dict().items()
            if not key.startswith("timestep_gate.")
        }
        payload = {
            "artifact_version": 1,
            "kind": FEEDBACK_FLOW_ARTIFACT_KIND,
            "flow_config": {
                "hidden_size": 4,
                "num_heads": 1,
                "steps": 1,
                "ff_multiplier": 1.0,
                "dropout": 0.0,
            },
            "flow_state_dict": old_state_dict,
            "metadata": {},
            "train_history": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_feedback_flow.pt"
            torch.save(payload, path)
            loaded, _ = load_feedback_flow_block(path)

        gate = loaded.timestep_gate(torch.tensor([1.0]), batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        self.assertAlmostEqual(gate.item(), 0.5, places=6)

    def test_training_timestep_uses_zero_based_dllm_steps(self):
        args = SimpleNamespace(steps=4)
        generator = torch.Generator().manual_seed(0)
        samples = [sample_timestep(args, generator) for _ in range(64)]
        self.assertGreaterEqual(min(samples), 0)
        self.assertLess(max(samples), args.steps)

    def test_timestep_gate_matches_own_parameter_dtype(self):
        gate = TimestepResidualGate(hidden_size=4).to(dtype=torch.float64)
        output = gate(
            torch.tensor([1.0], dtype=torch.float32),
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(output.dtype, torch.float32)
        self.assertAlmostEqual(output.item(), 0.5, places=6)

    def test_entropy_gate_matches_formula_and_increases_with_entropy(self):
        gate = TimestepResidualGate(hidden_size=4).to(dtype=torch.float64)
        with torch.no_grad():
            for parameter in gate.net.parameters():
                parameter.fill_(0.2)
            for parameter in gate.entropy_net.parameters():
                parameter.fill_(0.3)
        timestep = torch.tensor([1.0, 3.0])
        entropy = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        result = gate(timestep, 2, torch.device("cpu"), torch.float32, previous_entropy=entropy)
        gate_input = timestep.to(torch.float64).view(2, 1)
        expected = torch.sigmoid(
            gate.net(gate_input) + F.softplus(gate.entropy_net(gate_input)) * (entropy - 0.5)
        ).float().unsqueeze(-1)
        self.assertEqual(result.shape, (2, 3, 1))
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, expected)
        self.assertTrue(torch.all(result[:, 1:] > result[:, :-1]))

        neutral = gate(timestep, 2, torch.device("cpu"), torch.float32)
        self.assertEqual(neutral.shape, (2, 1, 1))
        torch.testing.assert_close(neutral[:, 0], result[:, 1])

    def test_entropy_gate_backpropagates_to_both_networks(self):
        gate = TimestepResidualGate(hidden_size=4)
        entropy = torch.tensor([[0.0, 0.2], [0.1, 0.3]])
        result = gate(torch.tensor([1.0, 2.0]), 2, torch.device("cpu"), torch.float32, previous_entropy=entropy)
        result.sum().backward()
        for network in (gate.net, gate.entropy_net):
            with self.subTest(network=network):
                gradients = [parameter.grad for parameter in network.parameters()]
                self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)

    def test_normalized_entropy_is_detached_and_handles_extreme_logits(self):
        logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1000.0, -1000.0, -1000.0, -1000.0]]], requires_grad=True)
        entropy = normalized_token_entropy(logits.to(torch.float16))
        self.assertEqual(entropy.shape, (1, 2))
        self.assertEqual(entropy.dtype, torch.float32)
        self.assertFalse(entropy.requires_grad)
        torch.testing.assert_close(entropy, torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(normalized_token_entropy(torch.zeros(2, 3, 1)), torch.zeros(2, 3))
        for invalid in (torch.zeros(2, 3), torch.zeros(2, 3, 0)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                normalized_token_entropy(invalid)

    def test_entropy_gate_steers_each_response_token_only_in_conditional_batch(self):
        self.model = ToyModel()
        input_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
        baseline = self.model(input_ids).logits
        state = attach_feedback_flow(
            self.model, flow_block=EntropyGatedMeanFeedbackFlow(), layer_idx=0, feedback_layer_idx=0
        )
        controller = FeedbackFlowController(self.model)
        entropy = torch.tensor([[0.5, 0.0, 1.0]], requires_grad=True)
        controller.prepare_feedback(
            torch.zeros(1, 3, dtype=torch.long),
            response_mask=torch.tensor([[False, True, True]]),
            timestep=2,
            conditional_batch_size=1,
            previous_entropy=entropy,
        )
        self.assertFalse(state.previous_entropy.requires_grad)
        output = self.model(input_ids).logits
        expected_gate = torch.sigmoid(F.softplus(torch.tensor(0.0)) * (entropy.detach() - 0.5))
        torch.testing.assert_close(state.last_gate.squeeze(-1), expected_gate)
        torch.testing.assert_close(output[0, 1:, 1] - baseline[0, 1:, 1], expected_gate[0, 1:])
        torch.testing.assert_close(output[0, 0], baseline[0, 0])
        torch.testing.assert_close(output[1], baseline[1])

    def test_entropy_cache_restores_across_contexts_and_clears(self):
        self.model = ToyModel()
        state = attach_feedback_flow(
            self.model, flow_block=EntropyGatedMeanFeedbackFlow(), layer_idx=0, feedback_layer_idx=0
        )
        controller = FeedbackFlowController(self.model)
        ids = torch.zeros(1, 3, dtype=torch.long)
        outer_entropy = torch.zeros(1, 3)
        inner_entropy = torch.ones(1, 3, requires_grad=True)
        controller.prepare_feedback(ids, timestep=1, previous_entropy=outer_entropy)
        saved_entropy = state.previous_entropy
        with self.assertRaisesRegex(RuntimeError, "test context"):
            with controller.feedback_from_ids(ids, timestep=2, previous_entropy=inner_entropy):
                torch.testing.assert_close(state.previous_entropy, inner_entropy)
                self.assertFalse(state.previous_entropy.requires_grad)
                raise RuntimeError("test context")
        self.assertIs(state.previous_entropy, saved_entropy)

        with feedback_flow_context(self.model, feedback_hidden=state.feedback_hidden, previous_entropy=inner_entropy):
            torch.testing.assert_close(state.previous_entropy, inner_entropy)
            self.assertFalse(state.previous_entropy.requires_grad)
        self.assertIs(state.previous_entropy, saved_entropy)

        output = forward_with_feedback(
            self.model, controller, ids, None, ids, timestep=3, previous_entropy=inner_entropy
        )
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertIs(state.previous_entropy, saved_entropy)
        controller.clear_feedback()
        self.assertIsNone(state.previous_entropy)
        self.assertIsNone(state.last_gate)

    def test_training_feedback_uses_prediction_logits_entropy(self):
        self.model = ToyModel()
        state = attach_feedback_flow(
            self.model, flow_block=EntropyGatedMeanFeedbackFlow(), layer_idx=0, feedback_layer_idx=0
        )
        controller = FeedbackFlowController(self.model)
        batch = {
            "input_ids": torch.tensor([[0, 8, 2]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "response_mask": torch.tensor([[False, True, True]]),
            "timestep": torch.tensor([3.0]),
        }
        prediction_logits = self.model(batch["input_ids"]).logits
        expected_entropy = normalized_token_entropy(prediction_logits)
        feedback_ids = prepare_feedback(controller, self.model, batch, mask_id=8)
        torch.testing.assert_close(feedback_ids, torch.tensor([[0, 1, 2]]))
        torch.testing.assert_close(state.previous_entropy, expected_entropy)
        self.assertFalse(state.previous_entropy.requires_grad)

    def test_generation_passes_previous_step_entropy_including_cfg(self):
        class RecordingModel(ToyModel):
            def __init__(self):
                super().__init__()
                self.prediction_logits = []

            def forward(self, input_ids, attention_mask=None, **kwargs):
                output = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
                output.logits[..., 1] += input_ids.eq(9).float() + input_ids[:, :1].eq(0).float()
                self.prediction_logits.append(output.logits.clone())
                return output

        class RecordingController:
            def __init__(self):
                self.entropies = []

            def prepare_feedback(self, _feedback_ids, **kwargs):
                self.entropies.append(kwargs["previous_entropy"])

        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(map(str, ids.tolist())))
        for cfg_scale in (0.0, 1.5):
            with self.subTest(cfg_scale=cfg_scale):
                model = RecordingModel()
                controller = RecordingController()
                output, trace = generate_with_feedback(
                    model,
                    tokenizer,
                    torch.tensor([[0]]),
                    torch.ones(1, 1, dtype=torch.long),
                    steps=2,
                    gen_length=2,
                    block_length=2,
                    cfg_scale=cfg_scale,
                    controller=controller,
                )
                self.assertEqual(output.shape, (1, 3))
                self.assertEqual(len(trace), 2)
                self.assertIsNone(controller.entropies[0])
                first_logits = model.prediction_logits[0]
                if cfg_scale > 0:
                    conditional, unconditional = first_logits.chunk(2, dim=0)
                    first_logits = unconditional + (cfg_scale + 1.0) * (conditional - unconditional)
                torch.testing.assert_close(controller.entropies[1], normalized_token_entropy(first_logits))

    def test_version_two_artifact_preserves_old_gate_for_all_entropies(self):
        block = FeedbackFlowBlock(hidden_size=4, num_heads=1, steps=1, ff_multiplier=1.0)
        with torch.no_grad():
            block.timestep_gate.net[-1].bias.fill_(0.8)
        old_state_dict = {
            key: value for key, value in block.state_dict().items() if not key.startswith("timestep_gate.entropy_net.")
        }
        payload = {
            "artifact_version": 2,
            "kind": FEEDBACK_FLOW_ARTIFACT_KIND,
            "flow_config": dict(block.config.__dict__),
            "flow_state_dict": old_state_dict,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "version_two.pt"
            torch.save(payload, path)
            loaded, _ = load_feedback_flow_block(path)
        output = loaded.timestep_gate(
            1, 1, torch.device("cpu"), torch.float32, previous_entropy=torch.tensor([[0.0, 0.5, 1.0]])
        )
        torch.testing.assert_close(output, torch.sigmoid(torch.full((1, 3, 1), 0.8)))
        for key, value in block.timestep_gate.net.state_dict().items():
            torch.testing.assert_close(loaded.timestep_gate.net.state_dict()[key], value)

    def test_version_three_artifact_roundtrip_is_strict(self):
        block = FeedbackFlowBlock(hidden_size=4, num_heads=1, steps=1, ff_multiplier=1.0)
        with torch.no_grad():
            block.timestep_gate.entropy_net[-1].bias.fill_(1.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "version_three.pt"
            save_feedback_flow_artifact(path, block)
            loaded, payload = load_feedback_flow_block(path)
            self.assertEqual(payload["artifact_version"], 3)
            for key, value in block.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value)
            del payload["flow_state_dict"]["timestep_gate.entropy_net.2.bias"]
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                load_feedback_flow_block(path)


if __name__ == "__main__":
    unittest.main()
