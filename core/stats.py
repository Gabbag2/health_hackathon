"""
stats.py
--------
Utilitaires statistiques non-paramétriques pondérés pour l'analyse LFP.

Fonctions
---------
weighted_permutation_test : test de permutation sur la différence de
                            moyennes pondérées entre deux groupes.
sig_label                 : étoiles de significativité (p-value → string).
eta_sq_label              : label qualitatif pour η² (Cohen 1988).
"""

import numpy as np


def weighted_permutation_test(v1, w1, v2, w2, n_perm=10000, seed=42):
    """
    Test de permutation bilatéral sur la différence de moyennes pondérées.

    Statistique observée :
        Δ_obs = moyenne_pondérée(v1, w1) − moyenne_pondérée(v2, w2)

    Distribution nulle : les étiquettes de groupe sont réassignées
    aléatoirement à chaque permutation en conservant les tailles de groupe.
    Les poids sont réassignés avec les valeurs.

    Paramètres
    ----------
    v1, v2 : array-like, shape (n1,) et (n2,)
        Valeurs des deux groupes (NaN déjà exclus avant appel).
    w1, w2 : array-like, mêmes shapes
        Poids positifs (durées en secondes). Doivent être > 0.
    n_perm : int
        Nombre de permutations. Défaut : 10 000.
    seed : int
        Graine aléatoire pour la reproductibilité. Défaut : 42.

    Retourne
    --------
    obs_delta : float
        Δ observé = wmean_1 − wmean_2.
    p_two_sided : float
        p-value bilatérale : P(|Δ_perm| ≥ |Δ_obs|).
    p_one_sided : float
        p-value unilatérale (H1 : groupe 1 > groupe 2) : P(Δ_perm ≥ Δ_obs).
    perm_dist : np.ndarray, shape (n_perm,)
        Distribution nulle complète (utile pour visualisation).

    Notes
    -----
    Si l'un des groupes contient moins de 2 éléments, retourne
    (np.nan, np.nan, np.nan, np.array([])).
    """
    v1 = np.asarray(v1, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    w2 = np.asarray(w2, dtype=float)

    n1, n2 = len(v1), len(v2)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan, np.array([])

    def _wmean(v, w):
        wn = w / w.sum()
        return float(np.dot(wn, v))

    obs_delta = _wmean(v1, w1) - _wmean(v2, w2)

    pool_v = np.concatenate([v1, v2])
    pool_w = np.concatenate([w1, w2])
    rng = np.random.default_rng(seed)
    perm_dist = np.empty(n_perm)

    for i in range(n_perm):
        idx = rng.permutation(n1 + n2)
        a_v, a_w = pool_v[idx[:n1]], pool_w[idx[:n1]]
        b_v, b_w = pool_v[idx[n1:]], pool_w[idx[n1:]]
        perm_dist[i] = _wmean(a_v, a_w) - _wmean(b_v, b_w)

    p_two = float(np.mean(np.abs(perm_dist) >= np.abs(obs_delta)))
    p_one = float(np.mean(perm_dist >= obs_delta))

    return obs_delta, p_two, p_one, perm_dist


def sig_label(p):
    """
    Convertit une p-value en étoiles de significativité.

    Seuils conventionnels :
      p < 0.001 → '***'
      p < 0.01  → '**'
      p < 0.05  → '*'
      sinon     → 'ns'
      NaN       → 'n.a.'

    Paramètres
    ----------
    p : float

    Retourne
    --------
    str
    """
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.a."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def eta_sq_label(eta2):
    """
    Label qualitatif pour la taille d'effet η² (Cohen 1988).

    Seuils :
      η² < 0.01  → 'négligeable'
      η² < 0.06  → 'petit'
      η² < 0.14  → 'moyen'
      η² ≥ 0.14  → 'grand'
      NaN        → 'n.a.'

    Paramètres
    ----------
    eta2 : float

    Retourne
    --------
    str
    """
    if eta2 is None or (isinstance(eta2, float) and np.isnan(eta2)):
        return "n.a."
    if eta2 < 0.01:
        return "négligeable"
    if eta2 < 0.06:
        return "petit"
    if eta2 < 0.14:
        return "moyen"
    return "grand"
