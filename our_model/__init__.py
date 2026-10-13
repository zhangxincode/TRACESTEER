"""Feedback-flow steering modules for LLaDA/DLLM generation."""

from .feedback_flow import (
    FeedbackFlowBlock,
    FeedbackFlowConfig,
    FeedbackFlowController,
    FeedbackFlowState,
    LayerHiddenCollector,
    TimestepResidualGate,
    attach_feedback_flow,
    attach_feedback_flow_from_artifact,
    clear_feedback_flow,
    load_feedback_flow_artifact,
    load_feedback_flow_block,
    normalized_token_entropy,
    save_feedback_flow_artifact,
    feedback_flow_context,
    forward_with_feedback,
)
from .losses import (
    FeedbackFlowLossWeights,
    feedback_flow_loss,
    masked_hidden_mse,
    masked_token_cross_entropy,
)

__all__ = [
    "FeedbackFlowBlock",
    "FeedbackFlowConfig",
    "FeedbackFlowController",
    "FeedbackFlowLossWeights",
    "FeedbackFlowState",
    "LayerHiddenCollector",
    "TimestepResidualGate",
    "attach_feedback_flow",
    "attach_feedback_flow_from_artifact",
    "clear_feedback_flow",
    "load_feedback_flow_artifact",
    "load_feedback_flow_block",
    "save_feedback_flow_artifact",
    "feedback_flow_context",
    "feedback_flow_loss",
    "forward_with_feedback",
    "generate_with_feedback",
    "masked_hidden_mse",
    "masked_token_cross_entropy",
    "normalized_token_entropy",
]


def __getattr__(name):
    if name == "generate_with_feedback":
        from .generate_feedback import generate_with_feedback

        return generate_with_feedback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
