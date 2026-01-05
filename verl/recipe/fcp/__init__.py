"""
FCP (Feedback-Conditional Policy) package.

This implementation is based on the paper:
"Language Models Can Learn from Verbal Feedback Without Scalar Rewards"
Paper: https://arxiv.org/pdf/2509.22638
"""

from .fcp_ray_trainer import RayFCPTrainer

__all__ = ["RayFCPTrainer"]
