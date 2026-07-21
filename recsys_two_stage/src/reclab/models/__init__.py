"""Recommender models, from the trivial baseline upward."""

from reclab.models.als import ALS
from reclab.models.base import Recommender
from reclab.models.ease import EASE, estimate_memory_gb
from reclab.models.itemknn import ItemKNN
from reclab.models.popularity import Popularity

#: Registry used by the CLI and evaluation commands. Stage 2 and 3 models
#: register here so the evaluation code never needs editing to accommodate them.
MODELS = {
    Popularity.name: Popularity,
    ItemKNN.name: ItemKNN,
    EASE.name: EASE,
    ALS.name: ALS,
}

__all__ = [
    "ALS",
    "EASE",
    "ItemKNN",
    "MODELS",
    "Popularity",
    "Recommender",
    "estimate_memory_gb",
]
