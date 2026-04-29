"""
src.backtesting
---------------
Module de backtesting de la stratégie momentum long-only sur le MSCI World.

Exports principaux :
  DataLoader      — chargement et accès aux données
  SignalType      — types de signaux (4 variantes momentum)
  SelectionMode   — best_in_universe / best_in_class
  QuantileMode    — décile / quintile / quartile
  SignalCalculator— calcul des signaux
  AllocationMethod— méthodes d'allocation (3 variantes)
  AllocationEngine— moteur d'allocation + contrainte UCITS
  BacktestConfig  — configuration complète d'une stratégie
  BacktestResult  — résultats du backtesting
  BacktestEngine  — moteur de backtesting
  ReportingEngine — reporting institutionnel
"""

from .data_loader import DataLoader, RF_ANNUAL, RF_DAILY
from .signals import (
    SignalType,
    SelectionMode,
    QuantileMode,
    SignalCalculator,
    SIGNAL_LABELS,
    SELECTION_LABELS,
    QUANTILE_LABELS,
)
from .allocation import AllocationMethod, AllocationEngine, METHOD_LABELS
from .engine import BacktestConfig, BacktestResult, BacktestEngine
from .reporting import ReportingEngine

__all__ = [
    "DataLoader",
    "RF_ANNUAL",
    "RF_DAILY",
    "SignalType",
    "SelectionMode",
    "QuantileMode",
    "SignalCalculator",
    "SIGNAL_LABELS",
    "SELECTION_LABELS",
    "QUANTILE_LABELS",
    "AllocationMethod",
    "AllocationEngine",
    "METHOD_LABELS",
    "BacktestConfig",
    "BacktestResult",
    "BacktestEngine",
    "ReportingEngine",
]
