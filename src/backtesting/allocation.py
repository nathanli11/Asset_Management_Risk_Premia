"""
allocation.py
-------------
Méthodes de pondération du portefeuille pour la stratégie momentum long-only.

Méthodes implémentées :
  (a) Risk Parity (ERC): Equal Risk Contribution via optimisation scipy
  (b) Min Variance     : variance minimale via optimisation scipy
  (c) Signal Weight    : pondération proportionnelle au score du signal

Contrainte UCITS 5/10/40 :
  Implémentée sous forme d'optimisation sous contrainte (scipy SLSQP) :
    - Toute position individuelle ≤ 10 %
    - Somme des positions > 5 % ≤ 40 %
  La contrainte est appliquée en deux temps :
    1. Les bornes [0, 10%] sont intégrées dans l'optimisation principale
    2. La règle des 40 % est imposée via une deuxième passe d'optimisation QP
       (minimisation de ||w - w0||² sous la contrainte UCITS 40%)
"""

import logging
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds

logger = logging.getLogger(__name__)

# Paramètres UCITS 5/10/40
_UCITS_MAX_SINGLE = 0.10          # position maximale par titre
_UCITS_THRESHOLD_LARGE = 0.05     # seuil "grosse position"
_UCITS_MAX_SUM_LARGE = 0.40       # somme maximale des grosses positions

# Paramètres numériques
_EPS = 1e-8
_MAX_ITER_SCIPY = 1000


class AllocationMethod(Enum):
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    SIGNAL_WEIGHT = "signal_weight"


# Libellés courts pour les noms de fichiers / dossiers
METHOD_LABELS = {
    AllocationMethod.RISK_PARITY: "ERC",
    AllocationMethod.MIN_VARIANCE: "MinVar",
    AllocationMethod.SIGNAL_WEIGHT: "SigW",
}


class AllocationEngine:
    """
    Calcule les poids du portefeuille selon la méthode choisie.

    Paramètres
    ----------
    cov_lookback_days : nombre de jours historiques pour l'estimation de la
                        matrice de covariance (ERC, MinVar). Défaut : 252 (1 an).
    min_obs           : minimum d'observations pour estimer la covariance.
    """

    def __init__(self, cov_lookback_days: int = 252, min_obs: int = 60):
        self._cov_lookback = cov_lookback_days
        self._min_obs = min_obs

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def compute(
        self,
        method: AllocationMethod,
        tickers: List[str],
        returns: Optional[pd.DataFrame] = None,
        signal_scores: Optional[pd.Series] = None,
        apply_ucits: bool = True,
    ) -> pd.Series:
        """
        Calcule les poids du portefeuille.

        Paramètres
        ----------
        method        : méthode de pondération
        tickers       : liste des tickers sélectionnés
        returns       : DataFrame de rendements journaliers (requis pour ERC, MinVar)
        signal_scores : pd.Series {ticker: score} (requis pour SignalWeight)
        apply_ucits   : appliquer la contrainte UCITS 5/10/40

        Retourne
        --------
        pd.Series {ticker: poids}, normalisé à 1.0
        """
        if not tickers:
            return pd.Series(dtype=float)

        if method == AllocationMethod.RISK_PARITY:
            cov, valid_tickers = self._estimate_covariance(tickers, returns)
            if cov is None or len(valid_tickers) < 2:
                logger.warning("ERC : covariance insuffisante, repli uniforme")
                weights = self._uniform_weights(tickers)
            else:
                weights = self._risk_parity(valid_tickers, cov, apply_ucits)
                weights = weights.reindex(tickers).fillna(0.0)

        elif method == AllocationMethod.MIN_VARIANCE:
            cov, valid_tickers = self._estimate_covariance(tickers, returns)
            if cov is None or len(valid_tickers) < 2:
                logger.warning("MinVar : covariance insuffisante, repli uniforme")
                weights = self._uniform_weights(tickers)
            else:
                weights = self._min_variance(valid_tickers, cov, apply_ucits)
                weights = weights.reindex(tickers).fillna(0.0)

        elif method == AllocationMethod.SIGNAL_WEIGHT:
            weights = self._signal_weight(tickers, signal_scores)

        else:
            raise ValueError(f"Méthode d'allocation inconnue : {method}")

        weights = weights.clip(lower=0.0)
        total = weights.sum()
        if total < _EPS:
            # Repli uniforme si somme nulle
            weights = self._uniform_weights(tickers)
            total = weights.sum()
        weights = weights / total

        # Contrainte UCITS 5/10/40 via optimisation (2e passe)
        if apply_ucits:
            weights = self._ucits_projection(weights)

        # Normalisation finale
        s = weights.sum()
        if s > _EPS:
            weights = weights / s

        return weights

    # ------------------------------------------------------------------
    # Méthodes de pondération
    # ------------------------------------------------------------------

    def _uniform_weights(self, tickers: List[str]) -> pd.Series:
        """Répartition uniforme entre les titres."""
        n = len(tickers)
        return pd.Series(1.0 / n, index=tickers)

    def _risk_parity(
        self,
        tickers: List[str],
        cov: np.ndarray,
        apply_ucits_bounds: bool = True,
    ) -> pd.Series:
        """
        Equal Risk Contribution (ERC) via optimisation SLSQP.

        Objectif : minimiser Σi Σj (RCi - RCj)²
        où RCi = wi × (Σw)i = contribution au risque du titre i

        Contraintes :
          - Σwi = 1
          - 0 ≤ wi ≤ 10% (si UCITS bounds actifs)
        """
        n = len(tickers)
        w0 = np.ones(n) / n  # point de départ uniforme

        upper = _UCITS_MAX_SINGLE if apply_ucits_bounds else 1.0
        bounds = Bounds(lb=np.zeros(n), ub=np.full(n, upper))

        def erc_objective(w: np.ndarray) -> float:
            sigma_p_sq = float(w @ cov @ w)
            if sigma_p_sq < _EPS:
                return 0.0
            # Contributions marginales au risque × poids
            rc = w * (cov @ w)
            # Objectif : égaliser les contributions → minimiser variance des RC
            rc_mean = rc.mean()
            return float(np.sum((rc - rc_mean) ** 2))

        def erc_gradient(w: np.ndarray) -> np.ndarray:
            sigma_p_sq = float(w @ cov @ w)
            if sigma_p_sq < _EPS:
                return np.zeros(n)
            rc = w * (cov @ w)
            rc_mean = rc.mean()
            # Gradient numérique pour robustesse
            return 2.0 * (rc - rc_mean) * (cov @ w + cov.T @ np.diag(w) @ np.ones(n))

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        res = minimize(
            erc_objective,
            w0,
            jac="2-point",
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": _MAX_ITER_SCIPY, "disp": False},
        )

        if res.success or res.fun < 1e-6:
            w = np.maximum(res.x, 0.0)
        else:
            logger.debug("ERC : optimisation non convergée, fallback 1/vol")
            # Fallback : pondération inverse de la volatilité
            vols = np.sqrt(np.diag(cov))
            inv_vol = np.where(vols > _EPS, 1.0 / vols, 0.0)
            w = inv_vol / inv_vol.sum() if inv_vol.sum() > _EPS else np.ones(n) / n

        return pd.Series(w, index=tickers)

    def _min_variance(
        self,
        tickers: List[str],
        cov: np.ndarray,
        apply_ucits_bounds: bool = True,
    ) -> pd.Series:
        """
        Minimum Variance (Markowitz) via optimisation SLSQP.

        Objectif : minimiser w^T Σ w
        Contraintes :
          - Σwi = 1
          - 0 ≤ wi ≤ 10% (si UCITS bounds actifs)
        La contrainte UCITS 40% est appliquée ensuite via _ucits_projection.
        """
        n = len(tickers)
        w0 = np.ones(n) / n

        upper = _UCITS_MAX_SINGLE if apply_ucits_bounds else 1.0
        bounds = Bounds(lb=np.zeros(n), ub=np.full(n, upper))

        def variance(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        def grad_variance(w: np.ndarray) -> np.ndarray:
            return 2.0 * cov @ w

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        res = minimize(
            variance,
            w0,
            jac=grad_variance,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": _MAX_ITER_SCIPY, "disp": False},
        )

        if res.success or res.fun < variance(w0):
            w = np.maximum(res.x, 0.0)
        else:
            logger.debug("MinVar : optimisation non convergée, fallback 1/vol²")
            variances = np.diag(cov)
            inv_var = np.where(variances > _EPS, 1.0 / variances, 0.0)
            w = inv_var / inv_var.sum() if inv_var.sum() > _EPS else np.ones(n) / n

        return pd.Series(w, index=tickers)

    def _signal_weight(
        self, tickers: List[str], signal_scores: Optional[pd.Series]
    ) -> pd.Series:
        """
        Pondération proportionnelle au score du signal (scores positifs uniquement).
        Les scores négatifs sont mis à 0 (long-only).
        """
        if signal_scores is None or signal_scores.empty:
            return self._uniform_weights(tickers)

        scores = signal_scores.reindex(tickers).fillna(0.0).clip(lower=0.0)
        total = scores.sum()
        if total < _EPS:
            return self._uniform_weights(tickers)
        return scores / total

    # ------------------------------------------------------------------
    # Contrainte UCITS 5/10/40 via optimisation sous contrainte
    # ------------------------------------------------------------------

    def _ucits_projection(self, weights: pd.Series) -> pd.Series:
        """
        Projette les poids sur l'ensemble admissible UCITS 5/10/40
        via optimisation quadratique (SLSQP) :

          min   ||w - w0||²
          s.t.  Σwi = 1
                0 ≤ wi ≤ 10%
                Σ{i ∈ I} wi ≤ 40%
                où I = {i : w0_i > 5%}  (ensemble des grosses positions)

        L'ensemble I est fixé depuis la solution initiale w0, ce qui rend
        le problème quadratique avec contrainte linéaire, soluble par SLSQP.
        """
        w0 = weights.values.copy().astype(float)
        n = len(w0)
        tickers = weights.index.tolist()

        # Vérification rapide : si déjà conforme, on retourne directement
        if self._is_ucits_compliant(w0):
            return weights

        # Ensemble I des grosses positions (fixé depuis w0)
        large_idx = np.where(w0 > _UCITS_THRESHOLD_LARGE)[0]

        def objective(w: np.ndarray) -> float:
            return float(np.sum((w - w0) ** 2))

        def grad_objective(w: np.ndarray) -> np.ndarray:
            return 2.0 * (w - w0)

        constraints = [
            # Contrainte d'égalité : somme = 1
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        ]

        # Contrainte UCITS 40% : somme des grosses positions ≤ 40%
        if len(large_idx) > 0:
            def ucits_40(w: np.ndarray) -> float:
                return _UCITS_MAX_SUM_LARGE - w[large_idx].sum()

            constraints.append({
                "type": "ineq",
                "fun": ucits_40,
            })

        bounds = Bounds(lb=np.zeros(n), ub=np.full(n, _UCITS_MAX_SINGLE))

        res = minimize(
            objective,
            w0,
            jac=grad_objective,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": _MAX_ITER_SCIPY, "disp": False},
        )

        if res.success or res.fun < 1.0:
            w = np.maximum(res.x, 0.0)
        else:
            # Fallback itératif si l'optimisation échoue
            logger.debug("UCITS projection : SLSQP non convergé, fallback itératif")
            w = self._ucits_iterative_cap(w0)

        # Normalisation finale
        s = w.sum()
        w = w / s if s > _EPS else np.ones(n) / n

        return pd.Series(w, index=tickers)

    def _ucits_iterative_cap(self, w: np.ndarray) -> np.ndarray:
        """
        Fallback itératif pour le respect de la règle UCITS 5/10/40.
        Utilisé si l'optimisation SLSQP ne converge pas.

        Étape 1 : plafonnement à 10%
        Étape 2 : réduction itérative des grosses positions jusqu'à Σ ≤ 40%
        """
        w = w.copy()
        # Étape 1 : plafonnement à 10%
        excess = np.maximum(w - _UCITS_MAX_SINGLE, 0.0)
        w = np.minimum(w, _UCITS_MAX_SINGLE)
        total_excess = excess.sum()
        if total_excess > _EPS:
            small_mask = w < _UCITS_MAX_SINGLE
            if small_mask.sum() > 0:
                w[small_mask] += total_excess * w[small_mask] / w[small_mask].sum()
        s = w.sum()
        w = w / s if s > _EPS else np.ones(len(w)) / len(w)

        # Étape 2 : règle des 40%
        for _ in range(200):
            large_mask = w > _UCITS_THRESHOLD_LARGE
            if w[large_mask].sum() <= _UCITS_MAX_SUM_LARGE + _EPS:
                break
            # Réduction proportionnelle des grosses positions
            factor = _UCITS_MAX_SUM_LARGE / w[large_mask].sum()
            freed = (w[large_mask] * (1.0 - factor)).sum()
            w[large_mask] *= factor
            # Redistribution sur les petites positions
            small_mask = ~large_mask
            if small_mask.sum() > 0 and w[small_mask].sum() > _EPS:
                w[small_mask] += freed * w[small_mask] / w[small_mask].sum()
            else:
                w[large_mask] += freed / large_mask.sum()
            w = np.maximum(w, 0.0)
            s = w.sum()
            w = w / s if s > _EPS else np.ones(len(w)) / len(w)

        return w

    @staticmethod
    def _is_ucits_compliant(w: np.ndarray) -> bool:
        """Vérifie si les poids respectent déjà la règle UCITS 5/10/40."""
        if np.any(w > _UCITS_MAX_SINGLE + _EPS):
            return False
        large = w[w > _UCITS_THRESHOLD_LARGE]
        if large.sum() > _UCITS_MAX_SUM_LARGE + _EPS:
            return False
        return True

    # ------------------------------------------------------------------
    # Estimation de la matrice de covariance
    # ------------------------------------------------------------------

    def _estimate_covariance(
        self,
        tickers: List[str],
        returns: Optional[pd.DataFrame],
    ):
        """
        Estime la matrice de covariance annualisée à partir des rendements journaliers.

        Retourne (cov_matrix, valid_tickers) ou (None, []) si données insuffisantes.
        Exclut les tickers avec moins de min_obs observations valides.
        """
        if returns is None or returns.empty:
            return None, []

        # Filtrer les tickers disponibles dans le DataFrame de rendements
        available = [t for t in tickers if t in returns.columns]
        if not available:
            return None, []

        sub = returns[available].copy()

        # Garder uniquement les tickers avec suffisamment d'observations
        valid_mask = sub.notna().sum() >= self._min_obs
        valid_tickers = [t for t in available if valid_mask.get(t, False)]

        if len(valid_tickers) < 2:
            return None, []

        sub = sub[valid_tickers].fillna(0.0)

        # Utilisation de la fenêtre de lookback
        if len(sub) > self._cov_lookback:
            sub = sub.iloc[-self._cov_lookback:]

        # Covariance annualisée (252 jours de bourse)
        cov_matrix = sub.cov().values * 252.0

        # Vérification de la positivité semi-définie
        # Légère régularisation (nugget) pour garantir l'inversibilité
        cov_matrix += np.eye(len(valid_tickers)) * 1e-8

        return cov_matrix, valid_tickers
