import unittest
import torch
from core.src.utils.data_processor import TextProcessor
from core.src.models.gpt import GPTLanguageModel

block_size = 256


class TestGPTLanguageModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.model = GPTLanguageModel(vocab_size=self.vocab_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

    def test_basic_generation(self):
        """Test if generate produces correct shape output and valid tokens"""
        idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        max_new_tokens = 10

        generated = self.model.generate(idx, max_new_tokens)

        self.assertEqual(generated.shape, (1, max_new_tokens + 1))
        self.assertTrue(torch.all(generated >= 0))
        self.assertTrue(torch.all(generated < self.vocab_size))

    def test_different_max_tokens(self):
        """Test generation with different numbers of max_new_tokens"""
        idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        for tokens in [1, 5, 20]:
            generated = self.model.generate(idx, tokens)
            self.assertEqual(generated.shape, (1, tokens + 1))

    def test_block_size_limit(self):
        """Test if generation respects block_size limit"""
        # Create input sequence at block_size limit
        idx = torch.zeros((1, block_size), dtype=torch.long,
                          device=self.device)
        max_new_tokens = 5

        generated = self.model.generate(idx, max_new_tokens)

        self.assertEqual(generated.shape, (1, block_size + max_new_tokens))

    def test_batch_generation(self):
        """Test generation with different batch sizes"""
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            idx = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=self.device)
            max_new_tokens = 10

            generated = self.model.generate(idx, max_new_tokens)

            self.assertEqual(generated.shape, (batch_size, max_new_tokens + 1))

    def test_device_consistency(self):
        """Test if generation maintains device placement"""
        idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        max_new_tokens = 5

        generated = self.model.generate(idx, max_new_tokens)

        self.assertEqual(generated.device, idx.device)


if __name__ == '__main__':
    unittest.main()
