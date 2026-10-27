"""No model download, network connection, or animal photos needed."""

import pytest
import torch

from probe1.similarity import (
    cosine_similarity_matrix,
    husky_wolf_margin,
    normalized_embeddings,
)


def test_normalization_gives_unit_length():
    x = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    z = normalized_embeddings(x)
    assert torch.allclose(torch.linalg.vector_norm(z, dim=-1), torch.ones(2))


def test_cosine_similarity_matrix_is_symmetric_with_unit_diagonal():
    x = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    scores = cosine_similarity_matrix(x)
    assert scores.shape == (3, 3)
    assert torch.allclose(scores, scores.T, atol=1e-6)
    assert torch.allclose(scores.diag(), torch.ones(3), atol=1e-6)
    assert scores[0, 2].item() == pytest.approx(0.0, abs=1e-6)


def test_margin_is_wolf_minus_retriever():
    scores = torch.tensor([[1.0, 0.8, 0.6], [0.8, 1.0, 0.4], [0.6, 0.4, 1.0]])
    assert husky_wolf_margin(scores).item() == pytest.approx(0.2, abs=1e-6)


def test_invalid_embedding_shape_is_rejected():
    with pytest.raises(ValueError):
        normalized_embeddings(torch.randn(512))
