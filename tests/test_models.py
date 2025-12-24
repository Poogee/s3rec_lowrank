"""Tests for S3Rec model implementations."""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.s3rec import S3RecModel, S3RecLowRankModel, create_model
from models.modules import TransformerEncoder, LayerNorm


class TestS3RecModel:
    """Tests for baseline S3RecModel."""
    
    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.num_items = 12102
        self.num_attributes = 1221
        self.hidden_size = 64
        self.batch_size = 4
        self.seq_len = 50
        
    @pytest.fixture
    def model(self, setup):
        """Create model instance."""
        return S3RecModel(
            num_items=self.num_items,
            num_attributes=self.num_attributes,
            hidden_size=self.hidden_size,
            num_layers=2,
            num_heads=2,
            max_seq_length=self.seq_len
        )
        
    def test_init(self, model, setup):
        """Test model initialization."""
        assert model.num_items == self.num_items
        assert model.num_attributes == self.num_attributes
        assert model.hidden_size == self.hidden_size
        
    def test_embeddings(self, model, setup):
        """Test embedding layers."""
        # Item embeddings should include mask token
        assert model.item_embeddings.num_embeddings == self.num_items + 1
        assert model.item_embeddings.embedding_dim == self.hidden_size
        
        # Attribute embeddings
        assert model.attribute_embeddings.num_embeddings == self.num_attributes
        
    def test_encode(self, model, setup):
        """Test sequence encoding."""
        input_ids = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        
        output = model.encode(input_ids)
        
        assert output.shape == (self.batch_size, self.seq_len, self.hidden_size)
        
    def test_pretrain_forward(self, model, setup):
        """Test pre-training forward pass."""
        attributes = torch.randint(0, 2, (self.batch_size, self.seq_len, self.num_attributes))
        masked_seq = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        pos_items = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        neg_items = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        masked_segment = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        pos_segment = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        neg_segment = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        
        aap_loss, mip_loss, map_loss, sp_loss = model.pretrain(
            attributes, masked_seq, pos_items, neg_items,
            masked_segment, pos_segment, neg_segment
        )
        
        # All losses should be scalars
        assert aap_loss.dim() == 0
        assert mip_loss.dim() == 0
        assert map_loss.dim() == 0
        assert sp_loss.dim() == 0
        
        # Losses should be non-negative
        assert aap_loss >= 0
        assert mip_loss >= 0
        assert map_loss >= 0
        assert sp_loss >= 0
        
    def test_finetune_forward(self, model, setup):
        """Test fine-tuning forward pass with causal attention."""
        input_ids = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        
        output = model.finetune(input_ids)
        
        assert output.shape == (self.batch_size, self.seq_len, self.hidden_size)
        
    def test_gradient_flow_pretrain(self, model, setup):
        """Test gradient flow in pre-training."""
        attributes = torch.randint(0, 2, (self.batch_size, self.seq_len, self.num_attributes))
        masked_seq = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        pos_items = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        neg_items = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        
        aap_loss, mip_loss, map_loss, sp_loss = model.pretrain(
            attributes, masked_seq, pos_items, neg_items,
            masked_seq, pos_items, neg_items
        )
        
        total_loss = aap_loss + mip_loss + map_loss + sp_loss
        total_loss.backward()
        
        # Check gradients exist
        assert model.item_embeddings.weight.grad is not None
        
    def test_parameter_count(self, model):
        """Test parameter counting."""
        counts = model.get_parameter_count()
        
        assert 'total' in counts
        assert 'item_embeddings' in counts
        assert 'encoder' in counts
        
        # Verify total matches sum
        assert counts['total'] == sum(v for k, v in counts.items() if k != 'total')


class TestS3RecLowRankModel:
    """Tests for S3RecLowRankModel."""
    
    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.num_items = 12102
        self.num_attributes = 1221
        self.hidden_size = 64
        self.rank = 16
        self.batch_size = 4
        self.seq_len = 50
        
    @pytest.fixture
    def model(self, setup):
        """Create low-rank model."""
        return S3RecLowRankModel(
            num_items=self.num_items,
            num_attributes=self.num_attributes,
            hidden_size=self.hidden_size,
            num_layers=2,
            num_heads=2,
            max_seq_length=self.seq_len,
            rank=self.rank
        )
        
    def test_init(self, model, setup):
        """Test initialization with low-rank components."""
        assert model.rank == self.rank
        
        # AAP and MAP heads should be LowRankAAP
        from models.lowrank_aap import LowRankAAP
        assert isinstance(model.aap_head, LowRankAAP)
        assert isinstance(model.map_head, LowRankAAP)
        
    def test_pretrain_forward(self, model, setup):
        """Test pre-training with low-rank AAP."""
        attributes = torch.randint(0, 2, (self.batch_size, self.seq_len, self.num_attributes))
        masked_seq = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        pos_items = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        neg_items = torch.randint(1, self.num_items, (self.batch_size, self.seq_len))
        
        aap_loss, mip_loss, map_loss, sp_loss = model.pretrain(
            attributes, masked_seq, pos_items, neg_items,
            masked_seq, pos_items, neg_items
        )
        
        # All losses should work
        assert aap_loss.dim() == 0
        assert map_loss.dim() == 0
        
    def test_parameter_reduction(self, model):
        """Test parameter reduction calculation."""
        reduction = model.get_parameter_reduction()
        
        assert 'lowrank_total' in reduction
        assert 'fullrank_total' in reduction
        assert 'reduction' in reduction
        
        # Low-rank should have fewer parameters
        assert reduction['lowrank_total'] < reduction['fullrank_total']
        
    def test_aap_analysis(self, model):
        """Test AAP analysis method."""
        analysis = model.get_aap_analysis()
        
        assert 'aap' in analysis
        assert 'map' in analysis
        assert 'rank' in analysis
        
    def test_reconstruct_weights(self, model, setup):
        """Test weight matrix reconstruction."""
        aap_weight, map_weight = model.reconstruct_aap_weights()
        
        assert aap_weight.shape == (self.hidden_size, self.hidden_size)
        assert map_weight.shape == (self.hidden_size, self.hidden_size)


class TestModelFactory:
    """Test model creation factory."""
    
    def test_create_baseline(self):
        """Test creating baseline model."""
        config = {
            'model': {'hidden_size': 64, 'num_layers': 2, 'num_heads': 2, 'dropout': 0.2},
            'sequence': {'max_length': 50}
        }
        
        model = create_model(
            num_items=1000,
            num_attributes=100,
            config=config,
            use_lowrank=False
        )
        
        assert isinstance(model, S3RecModel)
        assert not isinstance(model, S3RecLowRankModel)
        
    def test_create_lowrank(self):
        """Test creating low-rank model."""
        config = {
            'model': {'hidden_size': 64, 'num_layers': 2, 'num_heads': 2, 'dropout': 0.2},
            'lowrank': {'rank': 16},
            'sequence': {'max_length': 50}
        }
        
        model = create_model(
            num_items=1000,
            num_attributes=100,
            config=config,
            use_lowrank=True
        )
        
        assert isinstance(model, S3RecLowRankModel)


class TestTransformerEncoder:
    """Tests for Transformer encoder."""
    
    def test_forward(self):
        """Test encoder forward pass."""
        encoder = TransformerEncoder(
            num_layers=2,
            hidden_size=64,
            num_heads=2,
            dropout=0.1
        )
        
        x = torch.randn(4, 50, 64)
        outputs = encoder(x)
        
        assert len(outputs) == 2  # All layer outputs
        assert outputs[-1].shape == x.shape
        
    def test_with_mask(self):
        """Test encoder with attention mask."""
        encoder = TransformerEncoder(
            num_layers=2,
            hidden_size=64,
            num_heads=2
        )
        
        x = torch.randn(4, 50, 64)
        mask = torch.zeros(4, 1, 1, 50)
        mask[:, :, :, -10:] = -1e9  # Mask last 10 positions
        
        outputs = encoder(x, mask)
        
        assert outputs[-1].shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

