"""
Extra-bonus filters: SVSSM-specific differentiable filter + neural-OT
adapter + training infrastructure. Sub-package of ``src.filters.bonus``.

Grouped here (separate from the Kitagawa-centric filters in the parent
``bonus`` package) so the SVSSM headline target for Section 3 and the
neural-operator OT work for Section 2 are easy to distinguish.

Public symbols:

  - ``DifferentiableLEDHLogLikelihoodSVSSM`` -- the Sinkhorn-based SVSSM
    LEDH filter (Section 3 target).
  - ``DifferentiableLEDHNeuralOTSVSSM`` -- the neural-OT adapter
    (Section 2 target).
  - ``build_svssm_context_scalars`` -- 7-D context-vector builder used
    by both the adapter and the trainer.
  - ``SVSSMNeuralOTTrainer``, ``SVSSMTrainingDataset``,
    ``generate_svssm_training_data``, ``TrainingHistory`` --
    Phase-2 training infrastructure.
"""

from .differentiable_ledh_svssm import DifferentiableLEDHLogLikelihoodSVSSM
from .differentiable_ledh_neural_ot_svssm import (
    DifferentiableLEDHNeuralOTSVSSM,
    build_svssm_context_scalars,
)
from .svssm_neural_ot_training import (
    SVSSMNeuralOTTrainer,
    SVSSMTrainingDataset,
    TrainingHistory,
    generate_svssm_training_data,
)

__all__ = [
    "DifferentiableLEDHLogLikelihoodSVSSM",
    "DifferentiableLEDHNeuralOTSVSSM",
    "build_svssm_context_scalars",
    "SVSSMNeuralOTTrainer",
    "SVSSMTrainingDataset",
    "TrainingHistory",
    "generate_svssm_training_data",
]
