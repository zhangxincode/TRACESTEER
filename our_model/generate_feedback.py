from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model.backbone import (  # noqa: E402
    backbone_name,
    forward_model,
    load_backbone,
    prediction_mask,
    resolve_backbone_args,
)

try:  # pragma: no cover - supports both package and direct-script execution.
    from .feedback_flow import (
        FeedbackFlowController,
        attach_feedback_flow,
        attach_feedback_flow_from_artifact,
        normalized_token_entropy,
    )
except ImportError:  # pragma: no cover
    from feedback_flow import (
        FeedbackFlowController,
        attach_feedback_flow,
        attach_feedback_flow_from_artifact,
        normalized_token_entropy,
    )


MODEL_DIR = "./checkpoints/llada-Instruct"


def feedback_model_logits(
    model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return logits aligned to the token positions used by the feedback loop."""
    native_logits = forward_model(model, input_ids, attention_mask).logits
    if backbone_name(model) == "dream":
        token_logits = torch.cat([native_logits[:, :1], native_logits[:, :-1]], dim=1)
    else:
        token_logits = native_logits
    return token_logits


def gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(-1)
    noise = -torch.log(torch.rand_like(logits, dtype=torch.float64))
    return (logits.to(torch.float64).exp() / (noise**temperature)).argmax(-1)


def transfer_counts(mask: torch.Tensor, steps: int) -> torch.Tensor:
    total = mask.sum(1, keepdim=True)
    counts = total // steps
    counts = counts.repeat(1, steps)
    for i, remainder in enumerate((total % steps).squeeze(1)):
        counts[i, : int(remainder)] += 1
    return counts


def previous_prediction_feedback_ids(
    current_ids: torch.Tensor,
    previous_prediction: Optional[torch.Tensor],
    response_mask: torch.Tensor,
    mask_id: int,
) -> torch.Tensor:
    """Fill still-masked response slots with the previous raw prediction."""

    feedback_ids = current_ids.clone()
    if previous_prediction is None:
        return feedback_ids
    fill_mask = response_mask & current_ids.eq(mask_id)
    feedback_ids[fill_mask] = previous_prediction.to(current_ids.device)[fill_mask]
    return feedback_ids


def _decode_feedback(tokenizer, ids: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=False)


@torch.no_grad()
def generate_with_feedback(
    model,
    tokenizer,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    dream_alg: str = "origin",
    controller: Optional[FeedbackFlowController] = None,
    trace_feedback: bool = True,
) -> tuple[torch.Tensor, list[dict]]:
    """Generate with previous-prediction feedback for the selected backbone."""

    if backbone_name(model) == "dream":
        return _generate_dream_with_feedback(
            model,
            tokenizer,
            prompt,
            attention_mask,
            steps=steps,
            gen_length=gen_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            dream_alg=dream_alg,
            controller=controller,
            trace_feedback=trace_feedback,
        )

    mask_id = int(model.config.mask_token_id)
    prompt_len = prompt.size(1)
    trace = []
    x = torch.full(
        (prompt.size(0), prompt.size(1) + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.size(1)] = prompt
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones(prompt.size(0), gen_length)], dim=1)
    response_mask = torch.zeros_like(x, dtype=torch.bool)
    response_mask[:, prompt_len:] = attention_mask[:, prompt_len:].bool()

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length.")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by gen_length // block_length.")

    if controller is None and getattr(model, "_feedback_flow_state", None) is not None:
        controller = FeedbackFlowController(model)

    previous_prediction = None
    previous_entropy = None
    steps_per_block = steps // num_blocks
    for block_idx in range(num_blocks):
        start = prompt.size(1) + block_idx * block_length
        end = prompt.size(1) + (block_idx + 1) * block_length
        counts = transfer_counts(x[:, start:end] == mask_id, steps_per_block)
        for block_step in range(steps_per_block):
            global_step = block_idx * steps_per_block + block_step
            mask = x == mask_id
            feedback_ids = previous_prediction_feedback_ids(x, previous_prediction, response_mask, mask_id)

            if controller is not None:
                timestep = torch.full((x.size(0),), global_step, device=x.device, dtype=torch.float32)
                controller.prepare_feedback(
                    feedback_ids,
                    feedback_attention_mask=attention_mask,
                    response_mask=prediction_mask(model, response_mask),
                    timestep=timestep,
                    conditional_batch_size=x.size(0),
                    previous_entropy=previous_entropy,
                )

            if cfg_scale > 0.0:
                unconditional = x.clone()
                unconditional[:, :prompt_len] = mask_id
                model_input = torch.cat([x, unconditional], dim=0)
                model_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
                model_logits = feedback_model_logits(model, model_input, model_attention_mask)
                conditional_logits, unconditional_logits = torch.chunk(model_logits, 2, dim=0)
                logits = unconditional_logits + (cfg_scale + 1.0) * (conditional_logits - unconditional_logits)
                entropy_logits = logits
            else:
                logits = feedback_model_logits(model, x, attention_mask)
                entropy_logits = logits

            pred = gumbel_sample(logits, temperature)
            previous_prediction = torch.where(response_mask, pred, x)
            if controller is not None:
                previous_entropy = normalized_token_entropy(entropy_logits)
            conf = F.softmax(logits, -1).gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            conf[:, end:] = -float("inf")
            conf = torch.where(mask, conf, -float("inf"))
            filled = []
            for row_idx, count in enumerate(counts[:, block_step]):
                idx = conf[row_idx].topk(int(count)).indices
                x[row_idx, idx] = pred[row_idx, idx]
                filled.append({"batch": row_idx, "count": int(count)})

            item = {
                "step": global_step,
                "block": block_idx,
                "block_step": block_step,
                "timestep": global_step,
                "filled": filled,
                "text": tokenizer.decode(x[0, prompt_len:], skip_special_tokens=False),
            }
            if trace_feedback:
                item["feedback_text"] = _decode_feedback(tokenizer, feedback_ids, prompt_len)
            trace.append(item)
    return x, trace


@torch.no_grad()
def _generate_dream_with_feedback(
    model,
    tokenizer,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor,
    steps: int,
    gen_length: int,
    temperature: float,
    cfg_scale: float,
    dream_alg: str,
    controller: Optional[FeedbackFlowController],
    trace_feedback: bool,
) -> tuple[torch.Tensor, list[dict]]:
    """Use Dream's native scheduler while refreshing feedback before each model call."""
    if cfg_scale != 0.0:
        raise ValueError("Dream native diffusion generation requires --cfg-scale 0.")
    if dream_alg not in {"origin", "maskgit_plus", "topk_margin", "entropy"}:
        raise ValueError(f"Unsupported Dream generation algorithm: {dream_alg!r}.")

    mask_id = int(model.config.mask_token_id)
    prompt_len = prompt.size(1)
    sequence_length = prompt_len + int(gen_length)
    response_mask = torch.zeros(
        (prompt.size(0), sequence_length), dtype=torch.bool, device=prompt.device
    )
    response_mask[:, prompt_len:] = True
    feedback_attention_mask = torch.ones_like(response_mask, dtype=torch.long)
    traces: list[dict] = []
    previous_prediction: Optional[torch.Tensor] = None
    previous_entropy: Optional[torch.Tensor] = None

    if controller is None and getattr(model, "_feedback_flow_state", None) is not None:
        controller = FeedbackFlowController(model)

    def prepare_feedback(x: torch.Tensor, step: int) -> torch.Tensor:
        feedback_ids = previous_prediction_feedback_ids(
            x, previous_prediction, response_mask, mask_id
        )
        if controller is not None:
            controller.prepare_feedback(
                feedback_ids,
                feedback_attention_mask=feedback_attention_mask,
                response_mask=prediction_mask(model, response_mask),
                timestep=torch.full(
                    (x.size(0),), float(step), device=x.device, dtype=torch.float32
                ),
                conditional_batch_size=x.size(0),
                previous_entropy=previous_entropy,
            )
        return feedback_ids

    def tokens_hook(step, x, _logits):
        feedback_ids = prepare_feedback(x, 0 if step is None else int(step))
        if trace_feedback:
            traces.append({
                "step": 0 if step is None else int(step),
                "text": tokenizer.decode(x[0, prompt_len:], skip_special_tokens=False),
                "feedback_text": tokenizer.decode(
                    feedback_ids[0, prompt_len:], skip_special_tokens=False
                ),
            })
        return x

    def logits_hook(_step, _x, logits):
        nonlocal previous_prediction, previous_entropy
        previous_prediction = logits.argmax(dim=-1)
        previous_entropy = normalized_token_entropy(logits)
        return logits

    output = model.diffusion_generate(
        prompt,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        steps=steps,
        temperature=temperature,
        alg=dream_alg,
        mask_token_id=mask_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_history=False,
        generation_tokens_hook_func=tokens_hook,
        generation_logits_hook_func=logits_hook,
    )
    return output.sequences, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DLLM generation with previous-prediction Feedback Flow.")
    parser.add_argument("prompt", nargs="?", default="Where did fortune cookies originate?")
    parser.add_argument("--backbone", choices=["llada", "dream", "auto"], default="llada")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument(
        "--dream-alg",
        choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
        default="origin",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=3)
    parser.add_argument("--flow-scale", type=float, default=None)
    parser.add_argument("--flow-path")
    parser.add_argument("--json-path", default="./logs/feedback_flow_outputs.json")
    args = resolve_backbone_args(parser.parse_args())

    model, tokenizer = load_backbone(args.model_dir, args.backbone)
    device = model.device
    if args.flow_path:
        attach_feedback_flow_from_artifact(model, args.flow_path, scale=args.flow_scale)
    else:
        attach_feedback_flow(model, num_heads=args.num_heads, steps=args.flow_steps, scale=1.0 if args.flow_scale is None else args.flow_scale)

    text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
    output_ids, trace = generate_with_feedback(
        model,
        tokenizer,
        inputs.input_ids,
        inputs.attention_mask,
        args.steps,
        args.gen_length,
        args.block_length,
        args.temperature,
        args.cfg_scale,
        dream_alg=args.dream_alg,
    )
    final_text = tokenizer.decode(output_ids[0, inputs.input_ids.size(1) :], skip_special_tokens=True)

    json_path = Path(args.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "final_text": final_text,
                "gen_length": args.gen_length,
                "steps": args.steps,
                "block_length": args.block_length,
                "temperature": args.temperature,
                "cfg_scale": args.cfg_scale,
                "flow_steps": args.flow_steps,
                "flow_scale": args.flow_scale,
                "trace": trace,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(final_text)
    print(f"\nSaved inference outputs to: {json_path}")


if __name__ == "__main__":
    main()
