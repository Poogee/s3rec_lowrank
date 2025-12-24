"""Tests for Low-rank AAP module."""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lowrank_aap import LowRankAAP, FullRankAAP, compare_aap_modules


class TestLowRankAAP:
    """Tests for LowRankAAP module."""
    
    @pytest.fixture
    def setup(self):
        """Setup common test fixtures."""
        self.hidden_size = 64
        self.rank = 16
        self.batch_size = 4
        self.seq_len = 50
        self.num_attrs = 1221
        
    def test_init(self, setup):
        """Test module initialization."""
        module = LowRankAAP(self.hidden_size, self.rank)
        
        assert module.hidden_size == self.hidden_size
        assert module.rank == self.rank
        assert hasattr(module, 'U')
        assert hasattr(module, 'V')
        
    def test_invalid_rank(self, setup):
        """Test that rank >= hidden_size raises error."""
        with pytest.raises(AssertionError):
            LowRankAAP(self.hidden_size, self.hidden_size)
            
        with pytest.raises(AssertionError):
            LowRankAAP(self.hidden_size, self.hidden_size + 10)
            
    def test_forward_shape(self, setup):
        """Test forward pass output shape."""
        module = LowRankAAP(self.hidden_size, self.rank)
        
        sequence = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        attributes = torch.randn(self.num_attrs, self.hidden_size)
        
        output = module(sequence, attributes)
        
        expected_shape = (self.batch_size * self.seq_len, self.num_attrs)
        assert output.shape == expected_shape
        
    def test_output_range(self, setup):
        """Test that output is in [0, 1] after sigmoid."""
        module = LowRankAAP(self.hidden_size, self.rank)
        
        sequence = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        attributes = torch.randn(self.num_attrs, self.hidden_size)
        
        output = module(sequence, attributes)
        
        assert output.min() >= 0
        assert output.max() <= 1
        
    def test_parameter_count(self, setup):
        """Test parameter count calculation."""
        module = LowRankAAP(self.hidden_size, self.rank)
        stats = module.get_parameter_count()
        
        assert 'lowrank_params' in stats
        assert 'fullrank_params' in stats
        assert 'reduction_ratio' in stats
        
        # Verify parameter count
        expected_lowrank = 2 * self.hidden_size * self.rank
        assert stats['lowrank_params'] == expected_lowrank
        
        expected_fullrank = self.hidden_size * self.hidden_size
        assert stats['fullrank_params'] == expected_fullrank
        
    def test_reconstruction(self, setup):
        """Test weight matrix reconstruction."""
        module = LowRankAAP(self.hidden_size, self.rank)
        
        W = module.reconstruct_weight_matrix()
        
        assert W.shape == (self.hidden_size, self.hidden_size)
        
    def test_gradient_flow(self, setup):
        """Test that gradients flow through the module."""
        module = LowRankAAP(self.hidden_size, self.rank)
        
        sequence = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        attributes = torch.randn(self.num_attrs, self.hidden_size)
        
        output = module(sequence, attributes)
        loss = output.sum()
        loss.backward()
        
        assert sequence.grad is not None
        assert module.U.weight.grad is not None
        assert module.V.weight.grad is not None


class TestFullRankAAP:
    """Tests for FullRankAAP baseline module."""
    
    @pytest.fixture
    def setup(self):
        """Setup common test fixtures."""
        self.hidden_size = 64
        self.batch_size = 4
        self.seq_len = 50
        self.num_attrs = 1221
        
    def test_forward_shape(self, setup):
        """Test forward pass output shape."""
        module = FullRankAAP(self.hidden_size)
        
        sequence = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        attributes = torch.randn(self.num_attrs, self.hidden_size)
        
        output = module(sequence, attributes)
        
        expected_shape = (self.batch_size * self.seq_len, self.num_attrs)
        assert output.shape == expected_shape
        
    def test_parameter_count(self, setup):
        """Test parameter count."""
        module = FullRankAAP(self.hidden_size)
        stats = module.get_parameter_count()
        
        # Full rank: hidden_size^2 + hidden_size (bias)
        expected = self.hidden_size * self.hidden_size + self.hidden_size
        assert stats['fullrank_params'] == expected


class TestCompareModules:
    """Tests for module comparison utility."""
    
    def test_comparison(self):
        """Test comparison function."""
        results = compare_aap_modules(hidden_size=64, ranks=[8, 16, 32])
        
        assert 'hidden_size' in results
        assert 'full_rank_params' in results
        assert 'comparisons' in results
        
        assert len(results['comparisons']) == 3
        
        for comp in results['comparisons']:
            assert 'rank' in comp
            assert 'lowrank_params' in comp
            assert 'reduction' in comp
            
    def test_reduction_increases_with_lower_rank(self):
        """Test that lower rank gives more reduction."""
        results = compare_aap_modules(hidden_size=64, ranks=[8, 16, 32])
        
        # Extract reduction percentages
        reductions = []
        for comp in results['comparisons']:
            reduction_str = comp['reduction']
            reduction_val = float(reduction_str.replace('%', ''))
            reductions.append((comp['rank'], reduction_val))
        
        # Sort by rank
        reductions.sort(key=lambda x: x[0])
        
        # Lower rank should have higher reduction
        assert reductions[0][1] > reductions[1][1]
        assert reductions[1][1] > reductions[2][1]


class TestLowRankVsFullRank:
    """Compare low-rank and full-rank outputs."""
    
    def test_output_similarity_with_high_rank(self):
        """Test that high rank approximates full rank better."""
        hidden_size = 32
        batch_size = 4
        seq_len = 10
        num_attrs = 50
        
        # Same random input
        torch.manual_seed(42)
        sequence = torch.randn(batch_size, seq_len, hidden_size)
        attributes = torch.randn(num_attrs, hidden_size)
        
        full = FullRankAAP(hidden_size)
        low_r8 = LowRankAAP(hidden_size, rank=8)
        low_r16 = LowRankAAP(hidden_size, rank=16)
        
        out_full = full(sequence, attributes)
        out_r8 = low_r8(sequence, attributes)
        out_r16 = low_r16(sequence, attributes)
        
        # All outputs should have same shape
        assert out_full.shape == out_r8.shape == out_r16.shape
        
        # All outputs should be in [0, 1]
        assert out_full.min() >= 0 and out_full.max() <= 1
        assert out_r8.min() >= 0 and out_r8.max() <= 1
        assert out_r16.min() >= 0 and out_r16.max() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

