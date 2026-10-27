"""Small, testable math functions used in the first CLIP experiment."""

import torch
import torch.nn.functional as F


def normalized_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Return row-wise L2-normalized embeddings of shape [num_images, features]."""
    if embeddings.ndim != 2:
        raise ValueError("Expected embedding matrix of shape [num_images, features].")
    return F.normalize(embeddings.float(), p=2, dim=-1)


def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute all pairwise cosine similarities by matrix multiplication."""
    unit = normalized_embeddings(embeddings)
    return unit @ unit.T


def husky_wolf_margin(similarity: torch.Tensor) -> torch.Tensor:
    """Delta = similarity(husky, wolf) - similarity(husky, retriever).

    The caller's image order must be [husky, wolf, retriever].
    """
    if similarity.shape != (3, 3):
        raise ValueError("Expected a 3x3 similarity matrix in [husky, wolf, retriever] order.")
    return similarity[0, 1] - similarity[0, 2]
