"""
Data Assimilation Models Package

This package contains various filtering methods for data assimilation:
- Bootstrap Particle Filter (BPF)
- Flow-based filtering (EnFF)
- Score-based filtering (EnSF)
- Proposal distributions for particle filters
"""

from .base_pf import FilteringMethod
from .bpf import BootstrapParticleFilter
from .flow_pf import FlowFilter
from .score_pf import ScoreFilter
from .proposals import (
    ProposalDistribution,
    TransitionProposal,
    LearnedNeuralProposal,
    GaussianMixtureProposal,
    RectifiedFlowProposal,
)

__all__ = [
    "FilteringMethod",
    "BootstrapParticleFilter",
    "FlowFilter",
    "ScoreFilter",
    "ProposalDistribution",
    "TransitionProposal",
    "LearnedNeuralProposal",
    "GaussianMixtureProposal",
    "RectifiedFlowProposal",
]
