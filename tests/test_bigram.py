import unittest
from pathlib import Path

import torch
from core.src.utils.data_processor import TextProcessor
from core.src.models.bigram import BigramLanguageModel


class TestBigramModel(unittest.TestCase):
    def setUp(self):
        # Simple test vocabulary
        self.test_text = Path(__file__).resolve().parent.parent / "server/core/data/input.txt"
        self.processor = TextProcessor(self.test_text)
        self.model = BigramLanguageModel(vocab_size=self.processor.vocab_size)

    def test_model_initialization(self):
        """Test if model initializes with correct vocabulary size"""
        self.assertEqual(
            self.model.token_embedding_table.num_embeddings,
            self.processor.vocab_size
        )

    def test_forward_pass(self):
        """Test model forward pass with and without targets"""
        # Prepare sample data
        # batch_size=1, sequence_length=3
        x = torch.tensor([[0, 1, 2]], dtype=torch.long)
        y = torch.tensor([[1, 2, 3]], dtype=torch.long)

        # Test without targets
        logits, loss = self.model(x)
        self.assertEqual(logits.shape, (1, 3, self.processor.vocab_size))
        self.assertIsNone(loss)

        # Test with targets
        logits, loss = self.model(x, y)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Loss should be a scalar

    def test_generation(self):
        """Test if model can generate tokens"""
        idx = torch.zeros((1, 1), dtype=torch.long)
        logits, _ = self.model(idx)
        probs = torch.softmax(logits, dim=-1)
        self.assertEqual(probs.shape, (1, 1, self.processor.vocab_size))

    def test_device_placement(self):
        """Test if model can be moved to available device"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        x = torch.zeros((1, 1), dtype=torch.long, device=device)

        logits, _ = self.model(x)
        self.assertEqual(logits.device.type, device)


if __name__ == '__main__':
    unittest.main()
