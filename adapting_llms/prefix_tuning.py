import fnmatch
import logger
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl


class CausalSelfAttention(torch.nn.Module):
    """A Causal Self-Attention module.

    This module is based on the paper "Attention is all you need" by Vaswani et
    al. and the paper "Language Models are Unsupervised Multitask Learners" by
    Brown et al. The module is a causal self-attention module, meaning that the
    attention is only applied to the left of the current token. This is done by
    masking out the attention to the right of the current token. Ultimately,
    this module is a vanilla multi-head masked self-attention layer with a
    projection at the end. This could likely have been implemented with
    `torch.nn.MultiheadAttention`, but the documentation is sparse and code is
    not clear, so as an exercise we'll implement it ourself here.
    """

    def __init__(
        self,
        n_embedding_dims: int,
        n_attention_heads: int,
        self_attention_drop_probability: float,
        residual_drop_probability: float,
        block_size: int,
    ):
        super().__init__()
        assert n_embedding_dims % n_attention_heads == 0, (
            "Number of embedding dimensions should be divisible by the number "
            "of attention heads, this means each head gets an equal share of "
            "the embedding dimensions."
        )
        self.n_head_dims: int = n_embedding_dims // n_attention_heads
        # The key, query, value projections for all heads
        self.key_projection = torch.nn.Linear(n_embedding_dims, n_embedding_dims)
        self.query_projection = torch.nn.Linear(n_embedding_dims, n_embedding_dims)
        self.value_projection = torch.nn.Linear(n_embedding_dims, n_embedding_dims)
        # Regularization
        self.self_attention_dropout = torch.nn.Dropout(self_attention_drop_probability)
        self.output_dropout = torch.nn.Dropout(residual_drop_probability)
        # Output projection
        self.output_projection = torch.nn.Linear(n_embedding_dims, n_embedding_dims)
        # Causal mask to ensure that attention is only applied to the left in
        # the input sequence. Basically we don't want to use the future to
        # predict the present. Top triangle will be True, and be converted to
        # -infinity and thus zero when passed through a softmax function.
        triangle_matrix: torch.Tensor = torch.tril(
            torch.ones(block_size, block_size)
        ).view(1, 1, block_size, block_size)
        self.register_buffer(
            "is_future_token_mask", torch.isclose(triangle_matrix, torch.tensor(0.0))
        )
        # Number of heads.
        self.n_attention_heads = n_attention_heads
        # Used for visualisation
        self.attention: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the self-attention."""
        batch_size, block_size, n_embedding_dims = x.shape
        # The intermediate shape to use for the multi-head attention. A
        # transposition will be required to get the final shape. This splits
        # the embedding dimensions into multiple attention heads of equal size.
        to_shape: Tuple[int, int, int, int] = (
            batch_size,
            block_size,
            self.n_attention_heads,
            self.n_head_dims,
        )
        # Calculate { query, key, values } for all heads in batch and move head
        # forward to be the batch dim. After the reshape and transpose of dims
        # `1` and `2`, each of the projetions will have shape:
        #   (batch_size, n_attention_heads, block_size, n_head_dims)
        key: torch.Tensor = self.key_projection(x).view(to_shape).transpose(1, 2)
        query: torch.Tensor = self.query_projection(x).view(to_shape).transpose(1, 2)
        value: torch.Tensor = self.value_projection(x).view(to_shape).transpose(1, 2)
        # Scale by the square root of the head dimension.
        # As per the original paper, attention is all you need in sections
        # 3.2.1, this is to prevent the dot product from growing too large: "We
        # suspect that for large values of `n_head_dims`, the dot products grow
        # large in magnitude, pushing the softmax function into regions where
        # it has extremely small gradients. To counteract this effect, we scale
        # the dot products by 1.0/sqrt(`n_head_dims`)."
        # https://arxiv.org/pdf/1706.03762.pdf
        scaling_factor: float = 1.0 / math.sqrt(self.n_head_dims)
        # Causal self-attention, or in other words, self-attend:
        #   (batch_size, n_attention_heads, block_size, n_head_dims)
        # @ (batch_size, n_attention_heads, n_head_dims, block_size)
        # = (batch_size, n_attention_heads, block_size, block_size)
        #
        # We now have the attention scores for each head, but we still need to
        # apply the mask and normalize with softmax.
        attention: torch.Tensor = (query @ key.transpose(2, 3)) * scaling_factor
        # Apply the mask to prevent attending to the future.
        mask: torch.Tensor = self.is_future_token_mask[:, :, :block_size, :block_size]
        attention = attention.masked_fill(mask=mask, value=-torch.inf)
        # Normalize with softmax. All `-inf` values will be converted to 0.0,
        # so there is no attention being applied to the future tokens.
        attention = torch.nn.functional.softmax(attention, dim=-1)
        # Apply the dropout regularization.
        self.attention = attention = self.self_attention_dropout(attention)
        # Attend to the values to get the readout at each position.
        #   (batch_size, n_attention_heads, block_size, block_size)
        # @ (batch_size, n_attention_heads, block_size, n_head_dims)
        # = (batch_size, n_attention_heads, block_size, n_head_dims)
        #
        # Recall that the attention is softmaxed, so the sum of the attention
        # weights across all positions is 1.0, and this will mean the model
        # will focus on the most relevant tokens when making a prediction.
        y = attention @ value
        # Re-assemble all head outputs side by side, e.g if we had 4 heads of
        # size 64, the output will be of size 256.
        self.y = y = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch_size, block_size, n_embedding_dims)
        )
        # Output projection.
        return self.output_dropout(self.output_projection(y))


class GptTransformerBlock(torch.nn.Module):
    """GPT Transformer block."""

    def __init__(
        self,
        n_embedding_dims: int,
        n_attention_heads: int,
        self_attention_drop_probability: float,
        residual_drop_probability: float,
        block_size: int,
    ):
        super().__init__()
        self.layer_norm_0 = torch.nn.LayerNorm(n_embedding_dims)
        self.layer_norm_1 = torch.nn.LayerNorm(n_embedding_dims)
        self.self_attention = CausalSelfAttention(
            n_embedding_dims=n_embedding_dims,
            n_attention_heads=n_attention_heads,
            self_attention_drop_probability=self_attention_drop_probability,
            residual_drop_probability=residual_drop_probability,
            block_size=block_size,
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_embedding_dims, 4 * n_embedding_dims),
            torch.nn.GELU(),
            torch.nn.Linear(4 * n_embedding_dims, n_embedding_dims),
            torch.nn.Dropout(residual_drop_probability),
        )
        # The prefix MLP is used to project the soft prompt of shape:
        #   (batch_size, n_soft_prompt_tokens, n_embedding_dims).
        self.prefix_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_embedding_dims, n_embedding_dims),
            torch.nn.GELU(),
            torch.nn.Linear(n_embedding_dims, n_embedding_dims),
        )

    def forward(self, x: torch.Tensor, soft_prompt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass the transformer block."""
        # Collapse the soft_prompt tokens into the batch dimension.
        #    (batch_size, n_soft_prompt_tokens, n_embedding_dims)
        # -> (batch_size * n_soft_prompt_tokens, n_embedding_dims)
        soft_prompt = soft_prompt.view(-1, soft_prompt.size(-1))
        projected_soft_prompt: torch.Tensor = self.prefix_mlp(soft_prompt)
        # Reshape the soft prompt back to its original shape.
        #    (batch_size * n_soft_prompt_tokens, n_embedding_dims)
        # -> (batch_size, n_soft_prompt_tokens, n_embedding_dims)
        projected_soft_prompt = projected_soft_prompt.view(
            x.size(0), soft_prompt.size(1), soft_prompt.size(2)
        )
        # Concat the soft prompt to the input.
        x = torch.cat([projected_soft_prompt, x], dim=1)
        # Norm `x` before feeding to self-attention (see GPT-2 for this)
        layer_norm_0_x: torch.Tensor = self.layer_norm_0(x)
        # Self attention with residual connection.
        x = x + self.self_attention(layer_norm_0_x)
        # Norm `x` before feeding to mlp.
        layer_norm_1_x: torch.Tensor = self.layer_norm_1(x)
        # MLP with residual connection.
        x = x + self.mlp(layer_norm_1_x)
        # Split the soft prompt from the output otherwise it will be fed to the
        # next block and the context will grow with each block.
        n_soft_prompt_tokens: int = soft_prompt.shape[1]
        return x[:, n_soft_prompt_tokens:], soft_prompt


class GptTransfomer(pl.LightningModule):
    """The GptTransfomer language model, with a context window-size of `block_size`."""

    def __init__(
        self,
        vocab_size: int,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        learning_rate: float = 3e-4,
        n_embedding_dims: int = 768,
        block_size: int = 128,
        embedding_drop_probability: float = 0.1,
        n_layers: int = 12,
        n_attention_heads: int = 4,
        residual_drop_probability: float = 0.1,
        self_attention_drop_probability: float = 0.1,
        n_soft_prompt_tokens: int = 2,
    ):
        super().__init__()
        self.hparams: Any  # Adding this line to please mypy.
        # Saves all of the arguments passed to __init__ to self.hparams
        self.save_hyperparameters()
        # Create the soft prompt trainable tensor.
        self.soft_prompt = torch.nn.Parameter(
            torch.randn(1, n_soft_prompt_tokens, n_embedding_dims),
            requires_grad=True,
        )
        # Input embedding stem.
        self.char_token_embeddings = torch.nn.Embedding(vocab_size, n_embedding_dims)
        self.position_embeddings = torch.nn.Parameter(
            torch.zeros(1, block_size, n_embedding_dims)
        )
        self.embedding_dropout = torch.nn.Dropout(embedding_drop_probability)
        # Transformer blocks.
        blocks: List[GptTransformerBlock] = [
            GptTransformerBlock(
                n_embedding_dims=self.hparams.n_embedding_dims,
                n_attention_heads=self.hparams.n_attention_heads,
                self_attention_drop_probability=self.hparams.self_attention_drop_probability,
                residual_drop_probability=self.hparams.residual_drop_probability,
                block_size=self.hparams.block_size,
            )
            for _ in range(self.hparams.n_layers)
        ]
        self.blocks = torch.nn.Sequential(*blocks)
        # Decoder head (layer norm the final output and project to vocab size).
        self.layer_norm = torch.nn.LayerNorm(self.hparams.n_embedding_dims)
        self.head = torch.nn.Linear(
            self.hparams.n_embedding_dims, self.hparams.vocab_size, bias=False
        )
        # Initialise parameters.
        self.apply(self._init_weights)
        # Apply a special scaled init to the residual projections, per GPT-2 paper.
        for name, parameters in self.named_parameters():
            if fnmatch.fnmatch(name, "*.mlp.2.weight"):
                torch.nn.init.normal_(
                    parameters,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.hparams.n_layers),
                )
        # Log the parameters.
        n_parameters: int = sum(p.numel() for p in self.parameters())
        logger.info(f"Number of parameters: {n_parameters}")

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self):
        """Create the optimizer.

        Creates two lists of parmeters - those with, and those without weight
        decay (L2 regularization). The parameters without weight decay are the
        bias and LayerNorm parameters, and the parameters with weight decay are
        the rest.
        """
        verbose: bool = True
        # Don't apply weight decay to bias terms - we only need to regularise
        # the weights, the biases will work as intercepts.
        no_weight_decay_glob_patterns: List[str] = [
            "*.bias",
            "*layer_norm.weight",
            "char_token_embeddings.weight",
        ]
        # Loop over the parameters, and sort them into two groups, one with
        # weight decay, and one without.
        params_weight_decay: List[torch.nn.Parameter] = []
        params_no_weight_decay: List[torch.nn.Parameter] = []
        for name, parameters in self.named_parameters():
            # If the parameter name matches any of the names in the list of
            # names without weight decay, add it to the list of parameters
            # without weight decay. Otherwise, add it to the list of parameters
            # with weight decay.
            is_no_weight_decay_parameter: bool = any(
                fnmatch.fnmatch(name, pattern)
                for pattern in no_weight_decay_glob_patterns
            )
            if is_no_weight_decay_parameter:
                params_no_weight_decay.append(parameters)
            else:
                params_weight_decay.append(parameters)
            if verbose:
                print(f"{name} - weight decay: {not is_no_weight_decay_parameter}")
        # Create the AdamW optimizer, with the two groups of parameters.
        param_groups: List[Dict[str, Any]] = [
            {"params": params_weight_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_no_weight_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
        )
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, block_size). Here `x` is a
            sequence of tokens, where each token is an integer in the range
            [0, vocab_size).

        Returns
        -------
        logits : torch.Tensor
            output tensor of shape (batch_size, block_size, vocab_size).
        """
        batch_size, block_size = x.shape
        assert (
            block_size <= self.hparams.block_size - self.hparams.n_soft_prompt_tokens
        ), "Cannot forward, model block size is exhausted."
        # Each index is mapped to a learnable vector. If we had a different
        # modality to text, such as audio, we could skip the learnable
        # embeddings and feed the raw data directly to the transformer.
        token_embeddings: torch.Tensor = self.char_token_embeddings(x)
        # Each position maps to a learnable vector. Other models use sinusoidal
        # embeddings, but this is a simple way to get the job done. This has a
        # shape of (1, block_size, n_embedding_dims).
        position_embeddings: torch.Tensor = self.position_embeddings[:, :block_size, :]
        # Add the token and position embeddings together, and apply dropout.
        x = self.embedding_dropout(token_embeddings + position_embeddings)
        # Forward pass the transformer blocks. We feed in a shape of
        # (batch_size, block_size, n_embedding_dims) and get back a shape of
        # the same shape: (batch_size, block_size, n_embedding_dims).
        x = self.blocks(x=x, soft_prompt=self.soft_prompt)
        # Apply layer norm after last block (GPT-2 introduced this)
        x = self.layer_norm(x)
        # Project the `n_embedding_dims dimension (dim=2) of the transformer
        # blocks to the vocab size.
        logits: torch.Tensor = self.head(x)
        return logits

    def training_step(self, batch, batch_idx: int):
        """Forward pass and loss calculation for training."""
        # unpack the batch, which is `x` and `y`. `y` is the same as `x`, but
        # shifted by one token to the right. Both `x` and `y` are of shape
        # (batch_size, block_size).
        x, y = batch
        # Forward pass the data through the model. Get the logits, which has a
        # shape of (batch_size, block_size, vocab_size).
        logits: torch.Tensor = self(x)
        # Collapse the logits into a shape of
        # (batch_size * block_size, vocab_size)
        collapsed_logits: torch.Tensor = logits.view(-1, self.hparams["vocab_size"])
        # Collapse the labels into a shape of (batch_size * block_size,).
        collapsed_y: torch.Tensor = y.view(-1)
        # For each token in the sequence, calculate the cross entropy loss.
        loss: torch.Tensor = torch.nn.functional.cross_entropy(
            collapsed_logits, collapsed_y
        )
        self.log("train_loss", loss)
        return loss