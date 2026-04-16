"""
Data Assimilation Models Package

This package contains various filtering methods for data assimilation:
- Bootstrap Particle Filter (BPF)
- Ensemble Kalman Filter (EnKF)
- Local Ensemble Transform Kalman Filter (LETKF)
- Flow-based filtering (EnFF)
- Score-based filtering (EnSF)
- Proposal distributions for particle filters
"""

from .base_pf import FilteringMethod
from .bpf import BootstrapParticleFilter
from .enkf import EnsembleKalmanFilter, LocalEnsembleTransformKalmanFilter
from .localized_pf import LocalizedParticleFilter
from .localization import LocalizedPFConfig, LocalizationGeometry
from .flow_pf import FlowFilter
from .score_pf import ScoreFilter
from .proposals import (
    ProposalDistribution,
    TransitionProposal,
    LearnedNeuralProposal,
    GaussianMixtureProposal,
    RectifiedFlowProposal,
    LocalizedRFProposalWrapper,
    ShortcutF2D2Proposal,
)

__all__ = [
    "FilteringMethod",
    "BootstrapParticleFilter",
    "EnsembleKalmanFilter",
    "LocalEnsembleTransformKalmanFilter",
    "LocalizedParticleFilter",
    "LocalizedPFConfig",
    "LocalizationGeometry",
    "FlowFilter",
    "ScoreFilter",
    "ProposalDistribution",
    "TransitionProposal",
    "LearnedNeuralProposal",
    "GaussianMixtureProposal",
    "RectifiedFlowProposal",
    "LocalizedRFProposalWrapper",
    "ShortcutF2D2Proposal",
]
