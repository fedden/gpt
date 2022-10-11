"""Char generation tutorial with transfomers."""
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


class CharDataset(Dataset):
    def __init__(self, data: str, block_size: int) -> None:
        chars: List[str] = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"data has {data_size} characters, {vocab_size} unique.")
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.block_size: int = block_size
        self.vocab_size: int = vocab_size
        self.data: str = data

    def __len__(self) -> int:
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i : i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def get_text(data_source: str) -> str:
    """Download text from url and read it as a string."""
    available_data_sources: Dict[str, str] = {
        "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "wikipedia": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "philosophy": "https://s3.amazonaws.com/text-datasets/nietzsche.txt",
        "linux": "https://raw.githubusercontent.com/cedricdeboom/character-level-rnn-datasets/master/datasets/linux.txt",
        "midi": "https://raw.githubusercontent.com/cedricdeboom/character-level-rnn-datasets/master/datasets/music.txt",
        "game-of-thrones": "https://raw.githubusercontent.com/nihitx/game-of-thrones-/master/gameofthrones.txt",
    }
    if data_source not in available_data_sources:
        raise ValueError(
            f"Unknown data source. Available data sources: {available_data_sources.keys()}"
        )
    url = available_data_sources[data_source]
    with tempfile.NamedTemporaryFile() as f:
        logger.info(f"Downloading {data_source} data from {url}")
        r = requests.get(url)
        f.write(r.content)
        f.flush()
        with open(f.name, "r") as f:  # type: ignore
            text: str = f.read()  # type: ignore
    return text


# ===============================================================================
#
# Model code
#
# ===============================================================================


class CausalSelfAttention(torch.nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
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
        assert n_embedding_dims % n_attention_heads == 0
        # key, query, value projections for all heads
        self.key_projection = torch.nn.Linear(n_embedding_dims, n_embedding_dims)
        self.query_projection = torch.nn.Linear(n_embedding_dims, n_embedding_dims)
        self.value_projection = torch.nn.Linear(n_embedding_dims, n_embedding_dims)
        # regularization
        self.self_attention_drop = torch.nn.Dropout(self_attention_drop_probability)
        self.residual_drop = torch.nn.Dropout(residual_drop_probability)
        # output projection
        self.output_projection = torch.nn.Linear(n_embedding_dims, n_embedding_dims)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_attention_heads = n_attention_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the self-attention."""
        batch_size, block_size, n_embedding_dims = x.shape
        # Calculate { query, key, values } for all heads in batch and move head
        # forward to be the batch dim.
        key: torch.Tensor = (
            self.key_projection(x)
            .view(
                batch_size,
                block_size,
                self.n_attention_heads,
                n_embedding_dims // self.n_attention_heads,
            )
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        query: torch.Tensor = (
            self.query_projection(x)
            .view(
                batch_size,
                block_size,
                self.n_attention_heads,
                n_embedding_dims // self.n_attention_heads,
            )
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        value: torch.Tensor = (
            self.value_projection(x)
            .view(
                batch_size,
                block_size,
                self.n_attention_heads,
                n_embedding_dims // self.n_attention_heads,
            )
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        # causal self-attention; Self-attend:
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        att = att.masked_fill(
            self.mask[:, :, :block_size, :block_size] == 0, float("-inf")
        )
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.self_attention_drop(att)
        y = att @ value  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch_size, block_size, n_embedding_dims)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.residual_drop(self.output_projection(y))
        return y


class Block(torch.nn.Module):
    """ an unassuming Transformer block """

    def __init__(
        self,
        n_embedding_dims: int,
        n_attention_heads: int,
        self_attention_drop_probability: float,
        residual_drop_probability: float,
        block_size: int,
    ):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(n_embedding_dims)
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
        layer_norm_x: torch.Tensor = self.layer_norm(x)
        # Self attention with residual connection.
        x = x + self.self_attention(layer_norm_x)
        # MLP with residual connection.
        x = x + self.mlp(layer_norm_x)
        return x


class GPT(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """

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
        # input embedding stem
        self.tok_emb = torch.nn.Embedding(vocab_size, n_embedding_dims)
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embedding_dims))
        self.drop = torch.nn.Dropout(embedding_drop_probability)
        # transformer
        blocks: List[Block] = [
            Block(
                n_embedding_dims=self.hparams.n_embedding_dims,
                n_attention_heads=self.hparams.n_attention_heads,
                self_attention_drop_probability=self.hparams.self_attention_drop_probability,
                residual_drop_probability=self.hparams.residual_drop_probability,
                block_size=self.hparams.block_size,
            )
            for _ in range(self.hparams.n_layers)
        ]
        self.blocks = torch.nn.Sequential(*blocks)
        # decoder head
        self.layer_norm = torch.nn.LayerNorm(self.hparams.n_embedding_dims)
        self.head = torch.nn.Linear(
            self.hparams.n_embedding_dims, self.hparams.vocab_size, bias=False
        )
        # Initialise parameters.
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
        )
        return optimizer

    def forward(self, idx):
        b, t = idx.size()
        assert (
            t <= self.hparams.block_size
        ), "Cannot forward, model block size is exhausted."
        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        # same action as inference
        logits = self(x)
        # if we are given some desired targets also calculate the loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
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
            self.tokens += (
                y >= 0
            ).sum()  # number of tokens processed this step (i.e. label is not -100)
            if self.tokens < self.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
            else:
                # cosine learning rate decay
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

    @staticmethod
    def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Set all tokens but the k first most likely tokens to -infinity (1e10)."""
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float("Inf")
        return out

    @torch.no_grad()
    def inference(self, model: GPT) -> str:
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        """
        assert isinstance(model, GPT), "model must be an instance of GPT"
        x: torch.Tensor = torch.tensor(
            [self.stoi[s] for s in self.context], dtype=torch.long
        )[None, ...].to(model.device)
        print(self.context, end="", flush=True)
        model.eval()
        for k in range(self.steps):
            x_cond = (
                x
                if x.size(1) <= model.hparams.block_size
                else x[:, -model.hparams.block_size :]
            )  # crop context if needed
            logits = model(x_cond)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / self.temperature
            # optionally crop probabilities to only the top k options
            if self.top_k is not None:
                logits = self._top_k_logits(logits, self.top_k)
            # apply softmax to convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if self.sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # Push the new index onto the context and pop the first index if we're
            # overflowing.
            x = torch.cat((x, ix), dim=1)
            print(self.itos[ix.item()], end="", flush=True)
        print("\r", end="")
        # Covert the indices to text.
        return "".join([self.itos[int(i)] for i in x[0].tolist()])


# ==============================================================================
#
# Training and inference.
#
# ==============================================================================


@click.command()
@click.option("--data-source", default="game-of-thrones", help="ID of the dataset")
@click.option("--n-epochs", default=50, help="Number of epochs to train for")
@click.option("--batch-size", default=256, help="Batch size")
@click.option("--block-size", default=128, help="Block size")
@click.option("--seed", default=42, help="Seed")
def pipeline(
    data_source: str, n_epochs: int, batch_size: int, block_size: int, seed: int
) -> None:
    """Train and perform inference."""
    pl.seed_everything(seed)
    data: str = get_text(data_source)
    train_dataset: CharDataset = CharDataset(data=data, block_size=block_size)
    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()
    )
    model: GPT = GPT(
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
        context="The king said, ",
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
        model = GPT.load_from_checkpoint(model_path)
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
