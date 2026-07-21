"""SASRec — self-attentive sequential recommendation (Kang & McAuley 2018).

A causally-masked transformer reads a session's item sequence like a sentence and
predicts the next item. Where the two-tower model pools a *bag* of history, SASRec
uses *order* — "what did you just look at" rather than "what do you like overall",
which is the right question for session data.

Two details carry the weight:

**Causal masking is correctness-critical.** Position t may attend only to positions
<= t. A bug there lets the model peek at the future token it is trying to predict and
produces spectacular, meaningless accuracy. It has a dedicated test that perturbs a
later position and asserts earlier outputs do not move.

**The loss is the finding.** The original SASRec trains with binary cross-entropy
against a single sampled negative per position (``loss="sampled_bce"``). Klenitskiy &
Vasilev (2023) showed that the *same architecture* trained with full-softmax
cross-entropy over the catalogue (``loss="full_softmax"``) improves substantially —
enough to erase BERT4Rec's reported advantage. The loss function mattered more than
the architecture, and a line of published comparisons was confounded by it. With
~14k items, full softmax is entirely affordable here, so both are offered and both
reported.

Item indices in sequences are 0..n_items-1; internally they are shifted by +1 so that
0 is a padding index for the left-padded fixed-length windows.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from reclab.models.base import HistoryBatch, as_matrix, check_fitted

PAD = 0


class _SASRecNet(nn.Module):
    def __init__(self, n_items: int, emb_dim: int, max_len: int,
                 n_blocks: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.max_len = max_len
        # +1 for the padding index at 0; real items occupy 1..n_items.
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[PAD].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) right-padded item ids (0..n_items, 0=pad). Returns (B, L, d).

        Sequences are **right**-padded (real items first, padding last) and only the
        causal mask is used — no key-padding mask. With right-padding every position
        t can causally attend to the real tokens at positions 0..t, so no query is
        ever left with an all-masked context. That is what avoids the NaN a
        key-padding mask produces for fully-padded query rows; padding positions
        still produce outputs, but they sit past the real tokens and are never read.
        """
        b, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(b, length)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(positions))
        # Causal mask: position t attends only to <= t (True = blocked).
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=seq.device), diagonal=1
        )
        return self.encoder(x, mask=causal)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Score all real items from a hidden state via the tied embedding matrix."""
        return hidden @ self.item_emb.weight[1:].T  # drop the padding row


class SASRec:
    name = "sasrec"

    def __init__(
        self,
        n_items: int,
        emb_dim: int = 64,
        max_len: int = 50,
        n_blocks: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
        loss: str = "full_softmax",
        lr: float = 1e-3,
        epochs: int = 12,
        batch_size: int = 256,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if loss not in ("full_softmax", "sampled_bce"):
            raise ValueError(f"loss must be full_softmax or sampled_bce, got {loss}")
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.loss = loss
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.net_: _SASRecNet | None = None

    def _pad_right(self, seq: np.ndarray) -> tuple[np.ndarray, int]:
        """Most-recent max_len items, shifted +1, right-padded with PAD.

        Returns (padded, length). Real items occupy positions 0..length-1; the last
        real position (length-1) summarises the sequence for next-item scoring.
        """
        seq = seq[-self.max_len :] + 1  # +1: reserve 0 for padding
        out = np.zeros(self.max_len, dtype=np.int64)
        out[: len(seq)] = seq
        return out, len(seq)

    def fit(self, split) -> "SASRec":
        torch.manual_seed(self.seed)
        sequences = split.train_sequences
        if sequences is None:
            raise ValueError("SASRec.fit needs a SessionSplit carrying train_sequences")
        train = [s for s in sequences if len(s) >= 2]
        if not train:
            raise ValueError("no training sessions with >=2 items")

        net = _SASRecNet(self.n_items, self.emb_dim, self.max_len,
                         self.n_blocks, self.n_heads, self.dropout).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        n = len(train)

        for _ in range(self.epochs):
            order = rng.permutation(n)
            net.train()
            for start in range(0, n, self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                # Input = sequence[:-1], target = sequence[1:] (next-item at each pos).
                inputs = np.stack([self._pad_right(train[i][:-1])[0] for i in batch_idx])
                targets = np.stack([self._pad_right(train[i][1:])[0] for i in batch_idx])
                inp = torch.as_tensor(inputs, device=self.device)
                tgt = torch.as_tensor(targets, device=self.device)

                hidden = net(inp)  # (B, L, d)
                mask = tgt != PAD  # positions with a real next-item
                loss = self._loss(net, hidden, tgt, mask, rng)

                opt.zero_grad()
                loss.backward()
                opt.step()

        net.eval()
        self.net_ = net
        return self

    def _loss(self, net, hidden, tgt, mask, rng):
        h = hidden[mask]  # (M, d) real positions
        pos = tgt[mask] - 1  # back to 0-based real item ids
        if self.loss == "full_softmax":
            return nn.functional.cross_entropy(net.logits(h), pos)
        # sampled_bce: one uniform negative per position (original SASRec).
        neg = torch.as_tensor(
            rng.integers(0, self.n_items, size=len(pos)), device=self.device
        )
        item_w = net.item_emb.weight[1:]
        pos_score = (h * item_w[pos]).sum(-1)
        neg_score = (h * item_w[neg]).sum(-1)
        bce = nn.functional.binary_cross_entropy_with_logits
        return bce(pos_score, torch.ones_like(pos_score)) + bce(
            neg_score, torch.zeros_like(neg_score)
        )

    def score(self, history) -> np.ndarray:
        check_fitted(self, "net_")
        if isinstance(history, HistoryBatch) and history.sequences is not None:
            sequences = history.sequences
        else:
            # Fallback: no order available (a bare matrix). Reconstruct an
            # arbitrary-order sequence from the bag — degraded, but never reached
            # in the normal harness, which always carries sequences.
            mat = as_matrix(history).tocsr()
            sequences = [mat.indices[mat.indptr[r] : mat.indptr[r + 1]]
                         for r in range(mat.shape[0])]

        padded, lengths = [], []
        for s in sequences:
            s = np.asarray(s, dtype=np.int64)
            if len(s) == 0:
                padded.append(np.zeros(self.max_len, dtype=np.int64))
                lengths.append(1)  # read position 0; an all-pad row scores ~uniformly
            else:
                p, length = self._pad_right(s)
                padded.append(p)
                lengths.append(length)
        with torch.no_grad():
            hidden = self.net_(torch.as_tensor(np.stack(padded), device=self.device))
            # Gather the last *real* position per row (right-padding puts it at len-1).
            last_idx = torch.as_tensor(np.array(lengths) - 1, device=self.device)
            last = hidden[torch.arange(len(sequences)), last_idx, :]
            scores = self.net_.logits(last)
        return scores.cpu().numpy()
