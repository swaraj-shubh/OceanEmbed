"""Warm app state + domain errors. Stdlib only — holds loaded artifacts as opaque objects
so this module imports without numpy/torch. Populated once at startup by services.build_state.
"""
from dataclasses import dataclass, field
from typing import Any


class NotReady(Exception):
    """A required artifact (store / model / argo / metrics) was not loaded -> 503."""


class NotFound(Exception):
    """Unknown date / model, or a date the model cannot predict -> 404."""


class BadInput(Exception):
    """Valid request shape, invalid value (e.g. depth not in the 15 levels) -> 422."""


@dataclass
class Bundle:
    """One served model: net + its NIODataset (window/anomaly bound) + land mask + cache."""
    net: Any
    ds: Any
    kind: str
    window: int
    anomaly: bool
    land: Any                 # [15,96,176] bool, True = land/never-supervised
    dates: list               # dates this model can predict (window shrinks the head)
    cache: dict = field(default_factory=dict)   # date -> [15,96,176] reconstruction


@dataclass
class AppState:
    store: Any = None                 # xr.Dataset sliced to the split (raw physical units)
    dates: list = field(default_factory=list)
    date_index: dict = field(default_factory=dict)   # "YYYY-MM-DD" -> np.datetime64
    models: dict = field(default_factory=dict)        # run name -> Bundle
    argo: Any = None                  # DataFrame (tz-naive, region-filtered)
    metrics: dict = field(default_factory=dict)       # csv stem -> DataFrame
    device: str = "cpu"
    default_model: str = ""
    errors: dict = field(default_factory=dict)        # component -> error string

    def ready(self):
        return self.store is not None and len(self.models) > 0
