"""Recommender models, from the trivial baseline upward."""

from reclab.models.als import ALS
from reclab.models.base import HistoryBatch, Recommender, as_matrix
from reclab.models.ease import EASE, estimate_memory_gb
from reclab.models.itemknn import ItemKNN
from reclab.models.popularity import Popularity

# Neural models (Stage 2). Imported lazily-friendly: they need torch, which the
# classical path and its tests do not, so guard the import so a torch-less
# environment can still use the classical models.
try:
    from reclab.models.sasrec import SASRec
    from reclab.models.two_tower import TwoTower

    _NEURAL_AVAILABLE = True
except ImportError:  # pragma: no cover - torch not installed
    SASRec = TwoTower = None  # type: ignore
    _NEURAL_AVAILABLE = False

#: Registry of the classical models used by the CLI's tuned-defaults path. The
#: neural models take n_items / item_features at construction, so they are wired
#: in explicitly by the CLI rather than through this name->class map.
MODELS = {
    Popularity.name: Popularity,
    ItemKNN.name: ItemKNN,
    EASE.name: EASE,
    ALS.name: ALS,
}

__all__ = [
    "ALS",
    "EASE",
    "HistoryBatch",
    "ItemKNN",
    "MODELS",
    "Popularity",
    "Recommender",
    "SASRec",
    "TwoTower",
    "as_matrix",
    "estimate_memory_gb",
]
