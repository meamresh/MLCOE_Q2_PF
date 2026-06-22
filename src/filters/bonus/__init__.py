"""
Optional / experimental differentiable filter implementations.

These modules are not part of the core public API, but can be imported
for research experiments.
"""

from .differentiable_ledh import DifferentiableLEDHLogLikelihood
# SVSSM-specific filters live in the `extra_bonus` sub-package
# (Section 2 / Section 3 work). Re-exported here for backward-compatible
# import paths via `src.filters.bonus`.
from .extra_bonus import (
    DifferentiableLEDHLogLikelihoodSVSSM,
    DifferentiableLEDHNeuralOTSVSSM,
    build_svssm_context_scalars,
)
from .differentiable_pfpf_ledh import (
    DifferentiablePFPFLEDHLogLikelihood,
    KitagawaPFPFLEDHLogLikelihood,
)
from .lhnn_hmc_pf import LHNNHMCDiagnostics, run_lhnn_hmc
from .hmc_tfp import run_hmc_tfp, run_hmc_tfp_multi_chain
from .ssl_particle_gibbs import PGResult, particle_gibbs_ssl

__all__ = [
    "DifferentiableLEDHLogLikelihood",
    "DifferentiableLEDHLogLikelihoodSVSSM",
    "DifferentiableLEDHNeuralOTSVSSM",
    "build_svssm_context_scalars",
    "DifferentiablePFPFLEDHLogLikelihood",
    "KitagawaPFPFLEDHLogLikelihood",
    "LHNNHMCDiagnostics",
    "run_lhnn_hmc",
    "run_hmc_tfp",
    "run_hmc_tfp_multi_chain",
    "PGResult",
    "particle_gibbs_ssl",
]

