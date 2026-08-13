"""Assumption-explicit implementations of PG-SFD Eq. (6) and Eq. (7).

The paper does not publish enough detail to recover these losses uniquely.
The exact assumptions used by the main training entry point are documented in
``EQ67_IMPLEMENTATION.md`` at the repository root.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyProjection(nn.Module):
    """Affine phi for Eq. (7), initialized to the identity without RNG use."""

    def __init__(self, dimension):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, features):
        return F.linear(features, self.weight, self.bias)


def prototype_separation_loss(class_prototypes, normal_prototypes, tau=1.0):
    """Eq. (6): mean exp[-tau * (1 - cosine(p_class, p_normal))]."""

    if class_prototypes.ndim != 3 or normal_prototypes.ndim != 3:
        raise ValueError("Eq. (6) expects [B,C,D] and [B,K,D] prototypes")
    if (
        class_prototypes.shape[0] != normal_prototypes.shape[0]
        or class_prototypes.shape[2] != normal_prototypes.shape[2]
    ):
        raise ValueError("Eq. (6) prototype batch/feature dimensions must match")

    class_prototypes = F.normalize(class_prototypes, dim=-1)
    normal_prototypes = F.normalize(normal_prototypes, dim=-1)
    cosine = torch.einsum(
        "bcd,bkd->bck", class_prototypes, normal_prototypes
    )
    return torch.exp(-float(tau) * (1.0 - cosine)).mean()


def task_consistency_loss(
    classification_feature, reconstruction_feature, projection
):
    """Eq. (7): mean L1 distance between f_cls and phi(f_rec)."""

    if classification_feature.shape != reconstruction_feature.shape:
        raise ValueError("Eq. (7) task feature shapes must match")
    return F.l1_loss(
        classification_feature, projection(reconstruction_feature)
    )
