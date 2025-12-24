"""Tests for evaluation metrics."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import (
    hit_at_k,
    ndcg_at_k,
    mrr,
    compute_metrics,
    precision_at_k,
    auc
)


class TestMetrics:
    """Test evaluation metrics."""
    
    @pytest.fixture
    def perfect_predictions(self):
        """Create perfect predictions (ground truth always ranked first)."""
        predictions = np.random.randn(100, 100)
        predictions[:, 0] = 10.0  # Ground truth has highest score
        return predictions
        
    @pytest.fixture
    def worst_predictions(self):
        """Create worst predictions (ground truth always ranked last)."""
        predictions = np.random.randn(100, 100)
        predictions[:, 0] = -10.0  # Ground truth has lowest score
        return predictions
        
    @pytest.fixture
    def random_predictions(self):
        """Create random predictions."""
        np.random.seed(42)
        return np.random.randn(100, 100)
        
    def test_hit_at_k_perfect(self, perfect_predictions):
        """Test Hit@K with perfect predictions."""
        for k in [1, 5, 10]:
            assert hit_at_k(perfect_predictions, k) == 1.0
            
    def test_hit_at_k_worst(self, worst_predictions):
        """Test Hit@K with worst predictions."""
        for k in [1, 5, 10]:
            assert hit_at_k(worst_predictions, k) == 0.0
            
    def test_hit_at_k_range(self, random_predictions):
        """Test Hit@K is in [0, 1]."""
        for k in [1, 5, 10, 20]:
            score = hit_at_k(random_predictions, k)
            assert 0 <= score <= 1
            
    def test_hit_at_k_monotonic(self, random_predictions):
        """Test that Hit@K increases with K."""
        scores = [hit_at_k(random_predictions, k) for k in [1, 5, 10, 20]]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]
            
    def test_ndcg_at_k_perfect(self, perfect_predictions):
        """Test NDCG@K with perfect predictions."""
        # Perfect ranking: NDCG = 1.0
        for k in [1, 5, 10]:
            assert ndcg_at_k(perfect_predictions, k) == 1.0
            
    def test_ndcg_at_k_worst(self, worst_predictions):
        """Test NDCG@K with worst predictions."""
        for k in [1, 5, 10]:
            assert ndcg_at_k(worst_predictions, k) == 0.0
            
    def test_ndcg_at_k_range(self, random_predictions):
        """Test NDCG@K is in [0, 1]."""
        for k in [1, 5, 10, 20]:
            score = ndcg_at_k(random_predictions, k)
            assert 0 <= score <= 1
            
    def test_mrr_perfect(self, perfect_predictions):
        """Test MRR with perfect predictions."""
        assert mrr(perfect_predictions) == 1.0
        
    def test_mrr_worst(self, worst_predictions):
        """Test MRR with worst predictions."""
        # Worst case: item at position 99 (0-indexed)
        # MRR = 1/100 = 0.01
        assert mrr(worst_predictions) == pytest.approx(0.01, abs=0.001)
        
    def test_mrr_range(self, random_predictions):
        """Test MRR is in (0, 1]."""
        score = mrr(random_predictions)
        assert 0 < score <= 1
        
    def test_compute_metrics(self, random_predictions):
        """Test compute_metrics returns all expected metrics."""
        metrics = compute_metrics(random_predictions, ks=[1, 5, 10])
        
        expected_keys = ['hit@1', 'hit@5', 'hit@10', 'ndcg@1', 'ndcg@5', 'ndcg@10', 'mrr']
        for key in expected_keys:
            assert key in metrics
            assert 0 <= metrics[key] <= 1
            
    def test_precision_at_k(self, perfect_predictions):
        """Test Precision@K."""
        # For single relevant item, Precision@K = Hit@K / K
        for k in [1, 5, 10]:
            hit = hit_at_k(perfect_predictions, k)
            precision = precision_at_k(perfect_predictions, k)
            assert precision == pytest.approx(hit / k)
            
    def test_auc_perfect(self, perfect_predictions):
        """Test AUC with perfect predictions."""
        assert auc(perfect_predictions) == 1.0
        
    def test_auc_worst(self, worst_predictions):
        """Test AUC with worst predictions."""
        assert auc(worst_predictions) == 0.0
        
    def test_auc_range(self, random_predictions):
        """Test AUC is in [0, 1]."""
        score = auc(random_predictions)
        assert 0 <= score <= 1


class TestEdgeCases:
    """Test edge cases for metrics."""
    
    def test_single_user(self):
        """Test with single user."""
        predictions = np.array([[5.0, 3.0, 2.0, 1.0]])  # GT has highest score
        
        assert hit_at_k(predictions, 1) == 1.0
        assert ndcg_at_k(predictions, 1) == 1.0
        assert mrr(predictions) == 1.0
        
    def test_single_item(self):
        """Test with single item (just ground truth)."""
        predictions = np.array([[5.0]])
        
        assert hit_at_k(predictions, 1) == 1.0
        assert ndcg_at_k(predictions, 1) == 1.0
        assert mrr(predictions) == 1.0
        
    def test_tied_scores(self):
        """Test with tied prediction scores."""
        predictions = np.array([[5.0, 5.0, 5.0, 5.0]])
        
        # All items have same score, GT might be considered as rank 0 or higher
        metrics = compute_metrics(predictions, ks=[1, 10])
        
        # Just verify no errors and values are valid
        for v in metrics.values():
            assert 0 <= v <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

