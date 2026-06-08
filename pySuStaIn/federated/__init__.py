"""Federated Z-score SuStaIn.

A modular, network-agnostic federation layer for the cross-sectional Z-score
SuStaIn model. Each centre keeps its data local and returns only *aggregate*
statistics (responsibility sums, log-likelihood scalars, candidate-sequence
scores); the server runs the same EM / greedy sequence search as pooled
SuStaIn, driven by those aggregates.

Because every model parameter in ``ZscoreSustain`` is derived from ``Z_vals`` /
``Z_max`` (data-independent), and the data enters only through the per-subject
``_calculate_likelihood_stage`` whose results SuStaIn always reduces as a *sum
over subjects*, exact federation reproduces pooled SuStaIn: the federated EM
makes identical f-updates and sequence moves when driven with the same RNG.

Public API:
    ZscoreFederatedClient   - one centre (holds local data, returns aggregates)
    FederatedZscoreSustain  - the server/orchestrator (subclasses ZscoreSustain)
"""
from .client import ZscoreFederatedClient
from .server import FederatedZscoreSustain

__all__ = ["ZscoreFederatedClient", "FederatedZscoreSustain"]
