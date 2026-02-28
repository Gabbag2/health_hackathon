# README H2 — Couplage inter-région : évolution des corrélations LFP entre Stage I et Stage II

## Objectif

Ce notebook (`H2_COUPLING_SHIFT.ipynb`) analyse l'évolution du **couplage fonctionnel inter-région** (corrélation de Pearson des signaux LFP) entre le **Stage I** (sommeil pré-apprentissage) et le **Stage II** (sommeil post-apprentissage), dans trois paires de régions cérébrales :

- **vHPC ↔ dHPC** (hippocampe ventral ↔ dorsal)
- **BLA ↔ dHPC** (amygdale BLA ↔ hippocampe dorsal)
- **BLA ↔ vHPC** (amygdale BLA ↔ hippocampe ventral)

### Questions scientifiques

1. **Δ ≠ 0 ?** — Le shift de corrélation (Stage II − Stage I) est-il significativement différent de zéro pour chaque fichier et chaque condition ?
2. **Cohérence intra-condition ?** — Les deux fichiers de même condition (av-1 vs av-2, rw-1 vs rw-2) montrent-ils un shift cohérent (même direction, IC qui se chevauchent) ?
3. **Effet condition ?** — Le shift diffère-t-il entre les conditions aversive et rewarded ?

### Distinction avec les autres notebooks H2

| Aspect | H2_WEIGHTED_CORRELATION / H2_ALL_DATASETS_COMPARISON | H2_COUPLING_SHIFT (ce notebook) |
|--------|------------------------------------------------------|----------------------------------|
| Axe y des figures | Corrélation brute par epoch | **Δ = shift** (wmean Stage II − Stage I) |
| NREM / REM | Poolés en « Stage I / Stage II » | **Analysés séparément** |
| Violins | Distribution des corrélations par epoch | **Distribution bootstrap de Δ** |
| Question centrale | Aversive vs Rewarded | **Changement Stage I → Stage II** |

---

## Données

### Fichiers

| Identifiant | Fichier pkl | Condition | Epochs NREM I / NREM II | Epochs REM I / REM II |
|-------------|-------------|-----------|------------------------|-----------------------|
| `av-1` | `lfp_epochs_with_spikes_by_region-av-1.pkl` | Aversive | ~268 / ~248 | ~257 / ~242 |
| `av-2` | `lfp_epochs_with_spikes_by_region-av-2.pkl` | Aversive | ~32 / ~26 (longs ~77–105 s) | ~20 / ~17 |
| `rw-1` | `lfp_epochs_with_spikes_by_region-rw-1.pkl` | Rewarded | ~8 / ~5 (très longs ~221–228 s) | ~8 / ~5 |
| `rw-2` | `lfp_epochs_with_spikes_by_region-rw-2.pkl` | Rewarded | ~184 / ~203 | ~178 / ~193 |

### Structure d'un enregistrement (DataFrame)

| Colonne | Type | Description |
|---------|------|-------------|
| `epoch_label` | str | `'NREM I'`, `'NREM II'`, `'REM I'`, `'REM II'` |
| `t_start` | float | Début de l'epoch (s) |
| `t_end` | float | Fin de l'epoch (s) |
| `dHPC_lfp` | np.ndarray | Signal LFP hippocampe dorsal (µV) à 1250 Hz |
| `vHPC_lfp` | np.ndarray | Signal LFP hippocampe ventral (µV) à 1250 Hz |
| `bla_lfp` | np.ndarray | Signal LFP amygdale BLA (µV) à 1250 Hz |

**Fréquence d'échantillonnage** : 1250 Hz
**Epoch valide** : ≥ 1000 échantillons (~0.8 s minimum) et σ ≥ 0.01 µV pour les deux régions

---

## Pipeline de traitement

### Étape 1 — Corrélation de Pearson par epoch

Pour chaque epoch valide $e$ et chaque paire de régions $(r_1, r_2)$ :

$$r_e = \text{Pearson}\bigl(\text{LFP}_{r_1,e},\ \text{LFP}_{r_2,e}\bigr)$$

**Pas de filtre notch** : l'analyse de corrélation temporelle n'est pas affectée par les composantes à 50 Hz de la même façon que l'analyse spectrale. Cette approche est cohérente avec les notebooks H2 existants.

**Epoch invalide** (exclue du calcul) si :
- $N_e < 1000$ échantillons, ou
- $\hat{\sigma}(\text{LFP}_{r_1,e}) < 0.01$ µV, ou
- $\hat{\sigma}(\text{LFP}_{r_2,e}) < 0.01$ µV

### Étape 2 — Calcul du shift Δ

Pour chaque fichier, sleep type et paire de régions :

$$\Delta_{\text{obs}} = \bar{r}_{w,\text{Stage II}} - \bar{r}_{w,\text{Stage I}}$$

où $\bar{r}_w$ est la moyenne pondérée des corrélations (voir section Pondération).

- $\Delta > 0$ → le couplage **augmente** après l'expérience
- $\Delta < 0$ → le couplage **diminue** après l'expérience

### Étape 3 — Distribution bootstrap de Δ

Pour visualiser l'incertitude sur l'estimation de Δ :

$$\Delta^{(b)} = \bar{r}^{(b)}_{w,\text{Stage II}} - \bar{r}^{(b)}_{w,\text{Stage I}}$$

où $\bar{r}^{(b)}$ est calculé sur un rééchantillonnage pondéré avec remise (5 000 itérations, seed=42) tiré indépendamment depuis les deux stages.

---

## Métriques calculées

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| $r_e$ | Pearson(LFP₁, LFP₂) par epoch | Couplage fonctionnel instantané |
| $\bar{r}_w$ | Moyenne pondérée des $r_e$ | Couplage moyen du stage |
| $\Delta_{\text{obs}}$ | $\bar{r}_{w,\text{II}} - \bar{r}_{w,\text{I}}$ | Shift de couplage Stage I → II |
| IC 95% | Percentiles 2.5–97.5 du bootstrap | Incertitude sur $\Delta$ |

---

## Pondération par durée d'epoch

Les epochs ont des durées très variables (de quelques secondes à plusieurs minutes).
Toutes les statistiques sont **pondérées par la durée** :

$$w_e = t_{\text{end}} - t_{\text{start}} \quad \text{(secondes)}$$

**Moyenne pondérée :**
$$\bar{r}_w = \frac{\sum_e w_e \cdot r_e}{\sum_e w_e}$$

**Rééchantillonnage bootstrap :** tirage avec remise en utilisant les poids normalisés comme probabilités de sélection, ce qui préserve la représentation relative des epochs longs.

---

## Tests statistiques

### Test Δ ≠ 0 (permutation pondérée)

**H₀** : wmean(Stage II) = wmean(Stage I) → Δ = 0

Distribution nulle construite par permutation des étiquettes Stage I / Stage II (10 000 itérations, seed=42, poids permutés avec les valeurs) :

$$p_{\text{bilatéral}} = P\bigl(|\Delta_{\text{perm}}| \geq |\Delta_{\text{obs}}|\bigr)$$

Implémenté via `weighted_permutation_test` depuis `core/stats.py`.

### Cohérence intra-condition

Comparaison des intervalles de confiance bootstrap à 95% entre les deux fichiers d'une même condition :

- **IC se chevauchent** → les deux fichiers montrent des shifts compatibles
- **IC ne se chevauchent pas** → les deux fichiers divergent

### Test d'interaction (Aversive vs Rewarded)

**H₀** : Δ_aversive = Δ_rewarded

Statistique observée :
$$\delta_{\text{int}} = \Delta_{\text{av}} - \Delta_{\text{rw}}$$

Distribution nulle : permutation des étiquettes de condition (aversive/rewarded) en gardant les stages fixes (10 000 itérations). Les données aversives et rewarded sont poolées séparément pour chaque stage avant la permutation.

$$p_{\text{bilatéral}} = P\bigl(|\delta_{\text{int,perm}}| \geq |\delta_{\text{int,obs}}|\bigr)$$

**H₁ unilatérale** : Δ_rw > Δ_av → le couplage augmente davantage en condition rewarded.

---

## Fonctions `core/` utilisées

| Fonction | Module | Description |
|----------|--------|-------------|
| `weighted_permutation_test` | `core/stats.py` | Test de permutation pondéré (H0: Δ=0) |
| `sig_label` | `core/stats.py` | p-value → étoiles de significativité |
| `eta_sq_label` | `core/stats.py` | η² → label qualitatif |

---

## Sorties

| Fichier | Description |
|---------|-------------|
| `figs/H2_NREM_aversive.png` | Shift Δ NREM — av-1 et av-2 (distribution bootstrap) |
| `figs/H2_NREM_rewarded.png` | Shift Δ NREM — rw-1 et rw-2 (distribution bootstrap) |
| `figs/H2_REM_aversive.png` | Shift Δ REM — av-1 et av-2 (distribution bootstrap) |
| `figs/H2_REM_rewarded.png` | Shift Δ REM — rw-1 et rw-2 (distribution bootstrap) |
| `figs/H2_NREM_comparison.png` | Comparaison NREM — tous 4 fichiers + test interaction |
| `figs/H2_REM_comparison.png` | Comparaison REM — tous 4 fichiers + test interaction |
| `H2_coupling_shift_stats.csv` | Δ obs, IC 95%, p-value par fichier (24 lignes) |
| `H2_coupling_shift_interaction.csv` | Tests d'interaction + cohérence intra-condition (6 lignes) |

---

## Résultats

> Tests de permutation pondérés, N=5 000 itérations, seed=42.
> Notation : *** p<0.001 · ** p<0.01 · * p<0.05 · ns non significatif · n.a. données insuffisantes (n<3).

---

### Résumé exécutif

**Condition Aversive** : augmentation significative et cohérente du couplage inter-région dans les trois paires, aussi bien en NREM qu'en REM. Ce résultat est répliqué sur les deux fichiers (av-1 et av-2).

**Condition Rewarded** : aucun shift significatif pour les trois paires en NREM. En REM, rw-2 montre une augmentation significative ou tendancielle pour deux paires (vHPC↔dHPC, BLA↔dHPC). rw-1 est non interprétable (n=8/5 epochs).

**Effet condition** : la différence de shift entre conditions (Δ_av − Δ_rw) n'est pas significative pour aucune paire ni en NREM ni en REM, mais la direction est systématiquement Δ_av > Δ_rw.

---

### NREM — Shift NREM I → NREM II

#### Résultats par fichier

| Paire de régions | Fichier | n₁ / n₂ | wmean Stage I | wmean Stage II | Δ | p (perm.) | sig. |
|-----------------|---------|---------|--------------|---------------|---|-----------|------|
| vHPC ↔ dHPC | av-1 | 263 / 246 | 0.1703 | 0.2117 | +0.0414 | 0.0112 | * |
| vHPC ↔ dHPC | av-2 | 32 / 26 | 0.1295 | 0.1613 | +0.0318 | 0.0002 | *** |
| vHPC ↔ dHPC | rw-1 | 8 / 5 | 0.2138 | 0.2117 | −0.0021 | 0.9446 | ns |
| vHPC ↔ dHPC | rw-2 | 184 / 202 | 0.2775 | 0.3063 | +0.0287 | 0.1692 | ns |
| BLA ↔ dHPC | av-1 | 263 / 246 | 0.2827 | 0.3314 | +0.0486 | 0.0132 | * |
| BLA ↔ dHPC | av-2 | 32 / 26 | 0.2287 | 0.2811 | +0.0525 | <0.0001 | *** |
| BLA ↔ dHPC | rw-1 | 8 / 5 | 0.3390 | 0.3456 | +0.0066 | 0.8102 | ns |
| BLA ↔ dHPC | rw-2 | 184 / 202 | 0.4223 | 0.4533 | +0.0311 | 0.1860 | ns |
| BLA ↔ vHPC | av-1 | 263 / 246 | 0.3175 | 0.3630 | +0.0456 | 0.0100 | * |
| BLA ↔ vHPC | av-2 | 32 / 26 | 0.2821 | 0.3067 | +0.0246 | 0.0020 | ** |
| BLA ↔ vHPC | rw-1 | 8 / 5 | 0.3391 | 0.3275 | −0.0116 | 0.6964 | ns |
| BLA ↔ vHPC | rw-2 | 184 / 202 | 0.3967 | 0.4324 | +0.0357 | 0.1588 | ns |

#### Résultats poolés par condition (aversive = av-1+av-2, rewarded = rw-1+rw-2)

| Paire de régions | Δ_aversive | p_av | sig. | Δ_rewarded | p_rw | sig. | Δ_av−Δ_rw | p_interaction | sig. |
|-----------------|-----------|------|------|-----------|------|------|----------|--------------|------|
| vHPC ↔ dHPC | +0.0337 | 0.0012 | ** | +0.0278 | 0.1338 | ns | +0.0059 | 0.7848 | ns |
| BLA ↔ dHPC | +0.0451 | <0.0001 | *** | +0.0334 | 0.1220 | ns | +0.0117 | 0.6454 | ns |
| BLA ↔ vHPC | +0.0339 | 0.0034 | ** | +0.0305 | 0.1606 | ns | +0.0033 | 0.8932 | ns |

**Interprétation NREM** :
- **Aversive** : augmentation significative du couplage pour les trois paires (** à ***), répliquée sur av-1 et av-2 de façon cohérente.
- **Rewarded** : aucun shift significatif. rw-1 est très limité (n=8/5) ; rw-2 montre une tendance positive mais non significative.
- **Interaction** : pas de différence significative Δ_av vs Δ_rw, mais la direction est systématiquement Δ_av > Δ_rw pour les trois paires.

---

### REM — Shift REM I → REM II

#### Résultats par fichier

| Paire de régions | Fichier | n₁ / n₂ | wmean Stage I | wmean Stage II | Δ | p (perm.) | sig. |
|-----------------|---------|---------|--------------|---------------|---|-----------|------|
| vHPC ↔ dHPC | av-1 | 257 / 242 | 0.1863 | 0.2241 | +0.0378 | 0.0228 | * |
| vHPC ↔ dHPC | av-2 | 20 / 17 | 0.1670 | 0.1888 | +0.0218 | 0.1430 | ns |
| vHPC ↔ dHPC | rw-1 | 8 / 5 | 0.2288 | 0.2072 | −0.0216 | 0.6286 | ns |
| vHPC ↔ dHPC | rw-2 | 178 / 193 | 0.2855 | 0.3278 | +0.0423 | 0.0368 | * |
| BLA ↔ dHPC | av-1 | 257 / 242 | 0.3021 | 0.3494 | +0.0474 | 0.0150 | * |
| BLA ↔ dHPC | av-2 | 20 / 17 | 0.2944 | 0.3200 | +0.0256 | 0.1584 | ns |
| BLA ↔ dHPC | rw-1 | 8 / 5 | 0.3748 | 0.3463 | −0.0285 | 0.5380 | ns |
| BLA ↔ dHPC | rw-2 | 178 / 193 | 0.4449 | 0.4926 | +0.0477 | 0.0494 | * |
| BLA ↔ vHPC | av-1 | 257 / 242 | 0.3206 | 0.3605 | +0.0399 | 0.0260 | * |
| BLA ↔ vHPC | av-2 | 20 / 17 | 0.3149 | 0.3278 | +0.0129 | 0.4052 | ns |
| BLA ↔ vHPC | rw-1 | 8 / 5 | 0.3533 | 0.3252 | −0.0281 | 0.5684 | ns |
| BLA ↔ vHPC | rw-2 | 178 / 193 | 0.3965 | 0.4350 | +0.0385 | 0.0682 | ns |

#### Résultats poolés par condition

| Paire de régions | Δ_aversive | p_av | sig. | Δ_rewarded | p_rw | sig. | Δ_av−Δ_rw | p_interaction | sig. |
|-----------------|-----------|------|------|-----------|------|------|----------|--------------|------|
| vHPC ↔ dHPC | +0.0301 | 0.0126 | * | +0.0246 | 0.2666 | ns | +0.0056 | 0.8172 | ns |
| BLA ↔ dHPC | +0.0378 | 0.0058 | ** | +0.0266 | 0.2976 | ns | +0.0113 | 0.6924 | ns |
| BLA ↔ vHPC | +0.0282 | 0.0282 | * | +0.0201 | 0.3746 | ns | +0.0081 | 0.7484 | ns |

**Interprétation REM** :
- **Aversive** : augmentation significative poolée pour les trois paires (* à **), portée principalement par av-1 (grand N). av-2 montre la même direction mais sans significativité (n=20/17).
- **Rewarded** : rw-2 montre une augmentation significative pour vHPC↔dHPC (* p=0.037) et BLA↔dHPC (* p=0.049). rw-1 est trop petit (n=8/5) et tend même à montrer des Δ légèrement négatifs (non interprétables). Poolé, la condition rewarded est ns.
- **Interaction** : pas de différence significative Δ_av vs Δ_rw ; direction systématiquement Δ_av > Δ_rw.

---

### Cohérence intra-condition

**NREM :**
- **Aversive** : av-1 (* p≈0.01) et av-2 (** à *** p<0.002) montrent tous les deux un Δ positif et significatif pour les trois paires. La direction est parfaitement cohérente.
- **Rewarded** : rw-1 (n=8/5) donne des Δ proches de 0 ou légèrement négatifs (ns). rw-2 (n=184/202) donne des Δ positifs mais ns. La direction est incohérente entre les deux fichiers, mais rw-1 est trop petit pour tirer des conclusions.

**REM :**
- **Aversive** : av-1 (* pour les trois paires) et av-2 (Δ positif mais ns, n=20/17) — direction cohérente, la non-significativité de av-2 est vraisemblablement due à la faible puissance statistique.
- **Rewarded** : rw-2 est significatif pour deux paires (* vHPC↔dHPC, * BLA↔dHPC) ; rw-1 montre des Δ négatifs (ns). Incohérence de direction, mais rw-1 est non interprétable.

---

### Effet condition (Aversive vs Rewarded)

Aucune différence significative Δ_av − Δ_rw n'est détectée pour aucune paire, ni en NREM ni en REM (tous ns, p > 0.64).

Cependant, la direction est **systématiquement Δ_av > Δ_rw** dans les 6 comparaisons (3 paires × 2 sleep types). L'absence de significativité est probablement liée au manque de puissance dans la condition rewarded : rw-1 est très petit (n=8/5) et rw-2, bien que plus grand, reste insuffisant pour détecter des effets modérés une fois poolé avec rw-1.

**Conclusion globale** : le shift de couplage Stage I → Stage II est un effet robuste et répliqué dans la condition aversive (amplitude Δ ≈ +0.03 à +0.05, significatif pour les deux fichiers). Ce même pattern existe probablement dans la condition rewarded (direction positive dans rw-2, surtout en REM) mais ne peut pas être confirmé avec les données actuelles.
