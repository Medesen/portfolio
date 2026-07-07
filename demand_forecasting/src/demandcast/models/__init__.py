from demandcast.models.baselines import Naive, SeasonalNaive
from demandcast.models.lgbm import LgbmForecaster
from demandcast.models.sarimax import Sarimax, select_skus

#: CLI registry: name -> factory taking the full long frame (models that need
#: the known-in-advance promo/calendar schedule take it from here; sales
#: history still only ever reaches models via the per-fold train slice).
MODELS = {
    Naive.name: lambda long: Naive(),
    SeasonalNaive.name: lambda long: SeasonalNaive(),
    Sarimax.name: lambda long: Sarimax(promo_schedule=long),
    LgbmForecaster.name: lambda long: LgbmForecaster(full_long=long),
}

__all__ = [
    "MODELS",
    "LgbmForecaster",
    "Naive",
    "SeasonalNaive",
    "Sarimax",
    "select_skus",
]
