"""Char generation tutorial with transfomers."""
import fnmatch
import logging
import math
import multiprocessing
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import click
import joblib
import pytorch_lightning as pl
import numpy as np
import requests
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ===============================================================================
#
# Dataset code
#
# ===============================================================================


# Feel free to add more data sources here, this should be a link to one or more
# text files that we can download and read as a string. Check github/google
# around - there is a lot of context available out there.
AVAILABLE_DATA_SOURCES: Dict[str, List[str]] = {
    "shakespeare": [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ],
    "wikipedia": [
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    ],
    "philosophy": ["https://s3.amazonaws.com/text-datasets/nietzsche.txt"],
    "linux": [
        "https://raw.githubusercontent.com/cedricdeboom/character-level-rnn-datasets/master/datasets/linux.txt"
    ],
    "midi": [
        "https://raw.githubusercontent.com/cedricdeboom/character-level-rnn-datasets/master/datasets/music.txt"
    ],
    "game-of-thrones": [
        "https://raw.githubusercontent.com/nihitx/game-of-thrones-/master/gameofthrones.txt"
    ],
    "harry-potter": [
        "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20-%20The%20Philosopher's%20Stone.txt",
        "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%202%20-%20The%20Chamber%20of%20Secrets.txt",
        "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%203%20-%20The%20Prisoner%20of%20Azkaban.txt",
        "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%204%20-%20The%20Goblet%20of%20Fire.txt",
        "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%205%20-%20The%20Order%20of%20the%20Phoenix.txt",
        "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%206%20-%20The%20Half%20Blood%20Prince.txt",
        "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%207%20-%20The%20Deathly%20Hallows.txt",
    ],
}


class CharDataset(Dataset):
    """Character level dataset."""

    def __init__(self, data: str, block_size: int) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        data : str
            The data to use, should be a large body of text.
        block_size : int
            The number of characters to use in each block, that is fed into the
            model.
        """
        chars: List[str] = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size} characters, {vocab_size} unique.")
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.block_size: int = block_size
        self.vocab_size: int = vocab_size
        self.data: str = data

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single item from the dataset."""
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i : i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        # Input to the model is the first block_size characters.
        x = torch.tensor(dix[:-1], dtype=torch.long)
        # Target is the next character, or the future character after the block.
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def get_text(data_source: str) -> str:
    """Download text from url or frome file path and read it as a string."""
    if os.path.isfile(data_source):
        # If it's a file, just read it.
        with open(data_source, "r") as f:
            return f.read()
    elif data_source in AVAILABLE_DATA_SOURCES:
        # If it's url(s), download it/them. We'll use the requests library for
        # this.
        urls: List[str] = AVAILABLE_DATA_SOURCES[data_source]
        assert isinstance(urls, list), f"Urls should be list but was type: {type(urls)}"
        text: str = ""
        for url in urls:
            with tempfile.NamedTemporaryFile() as f:
                logger.info(f"Downloading {data_source} data from {url}")
                r = requests.get(url)
                f.write(r.content)
                f.flush()
                with open(f.name, "r") as f:  # type: ignore
                    text += f.read()  # type: ignore
        return text
    else:
        raise ValueError(
            f"Unknown data source {data_source}, please either provide a file "
            f"path or one of {list(AVAILABLE_DATA_SOURCES.keys())}"
        )


# ===============================================================================
#
# Model code
#
# ===============================================================================


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass the transformer block."""
        # Norm `x` before feeding to self-attention (see GPT-2 for this)
        layer_norm_0_x: torch.Tensor = self.layer_norm_0(x)
        # Self attention with residual connection.
        x = x + self.self_attention(layer_norm_0_x)
        # Norm `x` before feeding to mlp.
        layer_norm_1_x: torch.Tensor = self.layer_norm_1(x)
        # MLP with residual connection.
        x = x + self.mlp(layer_norm_1_x)
        return x


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
    ):
        super().__init__()
        self.hparams: Any  # Adding this line to please mypy.
        # Saves all of the arguments passed to __init__ to self.hparams
        self.save_hyperparameters()
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
            block_size <= self.hparams.block_size
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
        x = self.blocks(x)
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


# ==============================================================================
#
# Utility and learning rate stuff.
#
# ==============================================================================


class LearningRateDecayCallback(pl.Callback):
    def __init__(
        self, learning_rate, warmup_tokens=375e6, final_tokens=260e9, lr_decay=True
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.tokens = 0
        self.final_tokens = final_tokens
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        optimizer = trainer.optimizers[0]
        _, y = batch
        if self.lr_decay:
            # Number of tokens processed this step (i.e. label is not -100).
            self.tokens += (y >= 0).sum()
            if self.tokens < self.warmup_tokens:
                # Linear warmup of learning rate.
                lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
            else:
                # Cosine learning rate decay learning rate.
                progress = float(self.tokens - self.warmup_tokens) / float(
                    max(1, self.final_tokens - self.warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


class SampleCallback(pl.Callback):
    def __init__(
        self,
        context: str,
        itos: Dict[int, str],
        stoi: Dict[str, int],
        steps: int,
        temperature: float = 1.0,
        sample: bool = False,
        top_k: Optional[int] = None,
    ):
        super().__init__()
        assert len(context) > 0, "context must be non-empty"
        self.context = context
        self.itos = itos
        self.stoi = stoi
        self.steps = steps
        self.temperature = temperature
        self.sample = sample
        self.top_k = top_k

    def on_train_epoch_end(self, trainer, pl_module):
        print("=========================================")
        print(f"Sample from epoch {trainer.current_epoch}")
        print("=========================================")
        self.inference(model=pl_module)
        print("\n")

    def on_train_start(self, trainer, pl_module):
        print("=========================================")
        print("Sample from start of training")
        print("=========================================")
        self.inference(model=pl_module)
        print("\n")

    @staticmethod
    def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Set all tokens but the k first most likely tokens to -infinity (1e10)."""
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float("Inf")
        return out

    @torch.no_grad()
    def inference(self, model: GptTransfomer) -> str:
        """Generate text from the model.

        Here we will take a conditioning sequence of indices in x (of shape
        (batch_size, block_size)) and predict the next token in the sequence,
        feeding the predictions back into the model each time. Clearly the
        sampling has quadratic complexity unlike an RNN that is only linear,
        and has a finite context window of block_size, unlike an RNN that has
        an infinite context window.
        """
        assert isinstance(
            model, GptTransfomer
        ), "model must be an instance of GptTransfomer"
        # Convert the context to a tensor ready to be fed to the model.
        x: torch.Tensor = torch.tensor(
            [self.stoi[s] for s in self.context], dtype=torch.long
        )[None, ...].to(model.device)
        # We will print as we go (e.g no newline at the end of each print
        # statement).
        print(self.context, end="", flush=True)
        # Inference time.
        model.eval()
        for k in range(self.steps):
            # Crop the context window if needed.
            cropped_x: torch.Tensor = (
                x
                if x.size(1) <= model.hparams.block_size
                else x[:, -model.hparams.block_size :]
            )
            # Pluck the logits at the final step and scale by temperature
            logits: torch.Tensor = model(cropped_x)
            logits = logits[:, -1, :] / self.temperature
            # Optionally crop probabilities to only the top k options. This
            # will set the non-top-k logits to negative infinity, meaning they
            # will be ingnored after softmaxing.
            if self.top_k is not None:
                logits = self._top_k_logits(logits, self.top_k)
            # Apply softmax to convert to probabilities.
            likelihoods: torch.Tensor = torch.nn.functional.softmax(logits, dim=-1)
            # Sample from the distribution or take the most likely/argmax.
            if self.sample:
                predicted_class_i: torch.Tensor = torch.multinomial(
                    likelihoods, num_samples=1
                )
            else:
                _, predicted_class_i = torch.topk(likelihoods, k=1, dim=-1)
            # Push the new index onto the context. This will be cropped out
            # above in a future iteration if `x` has a size greater than
            # `block_size`.
            x = torch.cat((x, predicted_class_i), dim=1)
            print(self.itos[int(predicted_class_i.item())], end="", flush=True)
        print("\r", end="")
        # Covert the indices to text.
        return "".join([self.itos[int(i)] for i in x[0].tolist()])


# ==============================================================================
#
# Training and inference.
#
# ==============================================================================


@click.command()
@click.option(
    "--data-source",
    help=(
        f"Either a valid path to a text file locally that can be read for the"
        f"chars, or one of the following IDs, which link to URLs that will be"
        f"downloaded and cached locally: {list(AVAILABLE_DATA_SOURCES)}"
    ),
)
@click.option("--context", help="Context to use for sampling")
@click.option("--n-epochs", default=50, help="Number of epochs to train for")
@click.option("--batch-size", default=256, help="Batch size")
@click.option("--block-size", default=128, help="Block size")
@click.option("--seed", default=42, help="Seed")
def pipeline(
    data_source: str,
    context: str,
    n_epochs: int,
    batch_size: int,
    block_size: int,
    seed: int,
) -> None:
    """Train and perform inference."""
    pl.seed_everything(seed)
    data: str = get_text(data_source)
    train_dataset: CharDataset = CharDataset(data=data, block_size=block_size)
    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()
    )
    model: GptTransfomer = GptTransfomer(
        # Number of tokens in the vocabulary.
        vocab_size=train_dataset.vocab_size,
        # Size of the sequence.
        block_size=train_dataset.block_size,
        # Number of layers in the transformer.
        n_layers=8,
        # Number of attention heads in the transformer.
        n_attention_heads=8,
        # Embedding dimension.
        n_embedding_dims=512,
        # Learning rate.
        learning_rate=6e-4,
    )
    # Callbacks.
    lr_decay: LearningRateDecayCallback = LearningRateDecayCallback(
        learning_rate=6e-4,
        warmup_tokens=512 * 20,
        final_tokens=00 * len(train_dataset) * block_size,
    )
    sample: SampleCallback = SampleCallback(
        context=context,
        itos=train_dataset.itos,
        stoi=train_dataset.stoi,
        steps=1000,
        temperature=0.9,
        sample=True,
        top_k=5,
    )
    # Model path.
    model_md5: str = joblib.hash([seed, data_source, dict(model.hparams), n_epochs])
    model_path: str = f"{model_md5}.pt"
    if os.path.isfile(model_path):
        print(f"Found pre-trained model, loading model from: {model_path}")
        # Load the model.
        model = GptTransfomer.load_from_checkpoint(model_path)
    else:
        print(f"No model found, so training model, saving to: {model_path}")
        # Train the model.
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision=16,
            max_epochs=n_epochs,
            gradient_clip_val=1.0,
            callbacks=[lr_decay, sample],
            log_every_n_steps=5,
        )
        # Overfit to the corpus.
        trainer.fit(model, train_loader)
        # Save the model.
        trainer.save_checkpoint(model_path)
    # Now let's generate some text.
    sample.inference(model=model)


if __name__ == "__main__":
    pipeline()
