import torch
import torch.nn as nn
import torch.nn.functional as F

# Model parameters
n_embedding = 384  # Dimension of token embeddings
n_head = 6  # Number of attention heads
n_layer = 6  # Number of transformer blocks
dropout = 0.2  # Dropout probability
block_size = 256  # Context length for predictions


class Head(nn.Module):
    """
    A single self-attention head that performs scaled dot-product attention.

    The attention mechanism allows the model to weigh the importance of different
    positions in the input sequence when computing the representation for the
    current position.
    """

    def __init__(self, head_size):
        """
        Initialize attention head components.

        Args:
            head_size (int): Dimensionality of the attention head's key/query/value projections
        """
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Compute the output of the attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, block_size, n_embedding)
        """
        _, T, _ = x.shape
        key = self.key(x)
        query = self.query(x)

        # Compute attention scores
        attn = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        value = self.value(x)
        out = attn @ value
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    The model uses multiple attention heads to allow the model to focus on
    different parts of the input sequence when computing the representation
    for the current position.
    """

    def __init__(self, n_head, head_size):
        """
        Initialize the multi-head attention mechanism.

        Args:
            n_head (int): Number of attention heads
            head_size (int): Dimensionality of each attention head's key/query/value projections
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.fc = nn.Linear(n_head * head_size, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Compute the output of the multi-head attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, block_size, n_embedding)
        """
        return self.dropout(self.fc(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedFoward(nn.Module):
    """
    Feed-forward neural network with ReLU activation and dropout.

    The feed-forward network is applied after the self-attention mechanism
    to compute the final output of the transformer block.
    """

    def __init__(self, n_embedding):
        """
        Initialize the feed-forward network.

        Args:
            n_embedding (int): Dimensionality of token embeddings
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.ReLU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Compute the output of the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, block_size, n_embedding)
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block consisting of a multi-head self-attention mechanism
    followed by a feed-forward neural network.
    """

    def __init__(self, n_embedding, n_head):
        """
        Initialize the transformer block.

        Args:
            n_embedding (int): Dimensionality of token embeddings
            n_head (int): Number of attention heads
        """
        super().__init__()
        head_size = n_embedding // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedFoward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        """
        Compute the output of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, block_size, n_embedding)
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    GPT language model with a transformer decoder.

    The model uses a transformer decoder to autoregressive generate new tokens
    based on an input sequence of tokens.
    """

    def __init__(self, vocab_size):
        """
        Initialize the GPT language model.

        Args:
            vocab_size (int): Size of the vocabulary - number of unique tokens
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        self.blocks = nn.Sequential(
            *[Block(n_embedding, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embedding)
        self.fc = nn.Linear(n_embedding, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights of the model.

        Args:
            module (nn.Module): Module to initialize

            The weights of the model are initialized using a normal distribution
            with mean 0 and standard deviation 0.02.

            The bias terms are initialized to zero.

            The weights of the token embedding and position embedding tables
            are initialized using a normal distribution with mean 0 and standard
            deviation 0.02.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Compute the output of the GPT language model.

        Args:
            idx (torch.Tensor): Input tensor of token indices, shape (batch_size, sequence_length)
            targets (torch.Tensor, optional): Target tensor of next tokens, shape (batch_size, sequence_length)

        Returns:
            tuple: (logits, loss) if targets provided, else logits
        """
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding_table(pos)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.fc(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens by sampling from the model's probability distribution.

        Takes an input context of tokens and autoregressively generates max_new_tokens
        additional tokens by sampling from the predicted probability distribution.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=-1)

        return idx
