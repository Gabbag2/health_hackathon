# README H1 — Analyse spectrale LFP : Activité broadband et puissance par bande

## Objectif

Ce notebook (`H1_ALL_DATASETS_COMPARISON.ipynb`) compare l'activité spectrale LFP enregistrée
dans trois régions cérébrales (hippocampe dorsal, hippocampe ventral, amygdale BLA) sur
quatre enregistrements couvrant deux conditions comportementales :

- **Condition Aversive** : `av-1` et `av-2`
- **Condition Rewarded (non-aversive)** : `rw-1` et `rw-2`

La question principale est : l'activité spectrale LFP diffère-t-elle entre les stades de
sommeil (NREM I vs NREM II, REM I vs REM II) et entre les deux conditions ?

---

## Données

### Fichiers

| Identifiant | Fichier pkl | Condition | Nombre d'epochs (approx.) |
|-------------|-------------|-----------|--------------------------|
| `av-1` | `lfp_epochs_with_spikes_by_region-av-1.pkl` | Aversive | ~525 NREM I, ~490 NREM II |
| `av-2` | `lfp_epochs_with_spikes_by_region-av-2.pkl` | Aversive | ~52 NREM I, ~43 NREM II (epochs longs ~77–105 s) |
| `rw-1` | `lfp_epochs_with_spikes_by_region-rw-1.pkl` | Rewarded | ~16 NREM I, ~10 NREM II (très longs ~221–228 s) |
| `rw-2` | `lfp_epochs_with_spikes_by_region-rw-2.pkl` | Rewarded | ~362 NREM I, ~396 NREM II |

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
**Durée d'un epoch** : $\Delta t_e = t_{\text{end}} - t_{\text{start}}$ (secondes)
**Epoch valide** : ≥ 1000 échantillons (~0.8 s minimum)

---

## Pipeline de traitement

### Étape 1 — Filtre notch 50 Hz

Le signal brut contient une composante à 50 Hz due au bruit secteur. On applique un
filtre **coupe-bande (notch)** de Butterworth via `scipy.signal.iirnotch` avec filtrage
zero-phase (`filtfilt`) :

$$H_{\text{notch}}(f) \approx 0 \quad \text{pour} \quad f \approx f_{\text{notch}} \pm \frac{f_{\text{notch}}}{2Q}$$

**Paramètres :**
- Fréquence notch : $f_{\text{notch}} = 50$ Hz
- Facteur de qualité : $Q = 30$ → bande d'atténuation ≈ 1.67 Hz

```python
from scipy.signal import iirnotch, filtfilt
b, a = iirnotch(50.0, Q=30, fs=1250)
lfp_filtered = filtfilt(b, a, lfp)
```

Le filtre est appliqué **indépendamment** sur chaque epoch et chaque région.

### Étape 2 — Puissance spectrale (méthode de Welch)

La densité spectrale de puissance (PSD) est estimée par la méthode de Welch :

$$\hat{S}(f) = \frac{1}{K} \sum_{k=1}^{K} |X_k(f)|^2$$

où $X_k(f)$ est la TFD du $k$-ième segment fenêtré (fenêtre de Hann), $K$ le nombre de segments.

**Paramètres :**
- Longueur du segment : $n_{\text{seg}} = \min(4 \times f_s,\ N_{\text{epoch}})$ échantillons
- Chevauchement : 50% ($n_{\text{overlap}} = n_{\text{seg}} / 2$)

```python
from scipy.signal import welch
freqs, psd = welch(lfp, fs=1250, nperseg=n_seg, noverlap=n_seg//2)
```

---

## Métriques calculées

### Puissance broadband

$$P_{\text{broadband}} = \sum_{f=0.5}^{50} \hat{S}(f) \cdot \Delta f \quad (\mu V^2)$$

Représente l'énergie totale du signal filtré sur la bande 0.5–50 Hz.

### Puissance relative par bande

$$P_{\text{rel}}(\text{bande}) = \frac{\sum_{f \in [f_{\text{low}},\, f_{\text{high}}]} \hat{S}(f)}{\sum_{f=0.5}^{50} \hat{S}(f)}$$

Fraction de la puissance totale contenue dans la bande d'intérêt (adimensionnel, ∈ [0, 1]).

**Bandes de fréquence :**

| Bande | Fréquences | Contenu physiologique |
|-------|-----------|----------------------|
| `low` | 0.2–7 Hz | Ondes lentes, delta |
| `theta` | 7–12 Hz | Rythme theta hippocampique |
| `beta` | 15–30 Hz | Activité beta |
| `high` | 30–80 Hz | Gamme basse, haute fréquence |

---

## Pondération par durée d'epoch

Les epochs ont des durées très variables (de quelques secondes à plusieurs minutes),
notamment dans `av-2` et `rw-1`. Toutes les statistiques descriptives sont donc
**pondérées par la durée** de chaque epoch :

$$w_e = t_{\text{end}} - t_{\text{start}} \quad \text{(secondes)}$$

**Moyenne pondérée :**
$$\bar{x}_w = \frac{\sum_e w_e \cdot x_e}{\sum_e w_e}$$

**Écart-type pondéré :**
$$\sigma_w = \sqrt{\frac{\sum_e w_e \cdot (x_e - \bar{x}_w)^2}{\sum_e w_e}}$$

---

## Tests statistiques

### Mann-Whitney U (bilatéral)

Test non-paramétrique comparant les rangs de deux distributions indépendantes.

$$H_0 : P(X > Y) = P(Y > X) = 0.5$$

```python
from scipy import stats
U, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
```

### Taille d'effet

**r rang-bisériel :**
$$r = 1 - \frac{2U}{n_1 \cdot n_2}$$

**η² (Cohen 1988) :**
$$\eta^2 = r^2$$

| η² | Interprétation |
|----|---------------|
| < 0.01 | Négligeable |
| 0.01–0.06 | Petit |
| 0.06–0.14 | Moyen |
| ≥ 0.14 | Grand |

### Test de permutation pondéré

Complète le MWU en tenant compte des poids. La statistique est la différence de
moyennes pondérées entre les deux groupes :

$$\Delta_{\text{obs}} = \bar{x}_{w,1} - \bar{x}_{w,2}$$

Distribution nulle construite par permutation des étiquettes de groupe (10 000 itérations, seed=42) :

$$p_{\text{bilatéral}} = P\left(|\Delta_{\text{perm}}| \geq |\Delta_{\text{obs}}|\right)$$

---

## Comparaison inter-fichiers

Pour vérifier si les deux fichiers d'une même condition (av-1 vs av-2, rw-1 vs rw-2)
sont cohérents, un MWU bilatéral + test de permutation pondéré est appliqué **entre les
deux datasets** pour chaque combinaison stage × bande × région.

Un résultat non significatif (p > 0.05) indique que les deux enregistrements sont
compatibles et peuvent être considérés comme des répliques de la même condition.

---

## Fonctions `core/` utilisées

| Fonction | Module | Description |
|----------|--------|-------------|
| `apply_notch_filter` | `core/band_detection.py` | Filtre notch iirnotch + filtfilt |
| `compute_broadband_power` | `core/band_detection.py` | Puissance totale Welch 0.5–50 Hz |
| `compute_relative_band_power` | `core/band_detection.py` | Puissance relative Welch par bande |
| `weighted_permutation_test` | `core/stats.py` | Test de permutation pondéré |
| `sig_label` | `core/stats.py` | p-value → étoiles de significativité |
| `eta_sq_label` | `core/stats.py` | η² → label qualitatif |

---

## Sorties

| Fichier | Description |
|---------|-------------|
| `figs/H1_broadband_all_datasets.png` | Violins broadband — tous datasets |
| `figs/H1_NREM_delta_violin.png` | Distributions des Δ NREM I→II par fichier |
| `figs/H1_NREM_delta_bars.png` | Δ moyen pondéré NREM — résumé par bande |
| `figs/H1_REM_delta_violin.png` | Distributions des Δ REM I→II par fichier |
| `figs/H1_REM_delta_bars.png` | Δ moyen pondéré REM — résumé par bande |
| `H1_comparaison_statistiques.xlsx` | Résultats statistiques (5 onglets) |

---

## Résultats

Tests : Mann-Whitney U bilatéral (pondéré par durée d'epoch). Seuils : \* p<0.05, \*\* p<0.01, \*\*\* p<0.001.
Taille d'effet η² : négligeable <0.01 · petit <0.06 · moyen <0.14 · **grand ≥0.14**.

---

### 1. Cohérence inter-fichiers (même condition)

> **Conclusion principale : les deux fichiers d'une même condition présentent des profils spectraux DISTINCTS et ne doivent pas être poolés.**

#### Aversive : av-1 vs av-2

Les deux enregistrements aversifs diffèrent fortement, en particulier dans **vHPC** (η² > 0.85 dans toutes les bandes NREM et REM, p \*\*\*).
av-2 présente une puissance broadband vHPC nettement plus élevée (NREM I : ~15 500 µV² vs ~9 800 µV² pour av-1).

| Région | NREM I | NREM II | REM I | REM II |
|--------|--------|---------|-------|--------|
| dHPC | ** (low, high) | *** (low, high, theta) | ns | ** (low) |
| vHPC | *** toutes bandes (η²≥0.30) | *** toutes bandes (η²≥0.68) | *** low/beta/high | *** low/theta/beta/high |
| BLA | *** low/theta, * beta | *** toutes bandes | ns | * low/beta |

#### Rewarded : rw-1 vs rw-2

Différences tout aussi importantes (η² grand pour la majorité des bandes en NREM), mais à interpréter avec prudence : **rw-1 n'a que 5–8 epochs par stage**, ce qui rend les tests peu fiables (faible puissance statistique).

---

### 2. Activité broadband (Welch 0.5–50 Hz)

Tendance générale : légère **augmentation de NREM I vers NREM II** dans toutes les régions (+5 à +10 % en av-1 et rw-1/rw-2), sauf **av-2 vHPC** qui montre une diminution (15 485 → 14 299 µV²).

| Dataset | Région | NREM I (µV²) | NREM II (µV²) | REM I (µV²) | REM II (µV²) |
|---------|--------|-------------|--------------|------------|-------------|
| av-1 | dHPC | 8 106 | 8 558 (+6%) | 8 219 | 8 547 (+4%) |
| av-1 | vHPC | 9 828 | 10 843 (+10%) | 10 144 | 10 423 (+3%) |
| av-1 | BLA | 5 320 | 5 252 (−1%) | 4 406 | 4 662 (+6%) |
| av-2 | vHPC | **15 485** | **14 299 (−8%)** | 13 897 | 13 073 (−6%) |
| rw-2 | dHPC | 7 678 | 8 322 (+8%) | 7 973 | 8 156 (+2%) |
| rw-2 | vHPC | 9 921 | 9 936 (+0%) | 10 278 | 9 717 (−5%) |

---

### 3. Shift NREM I → NREM II (puissance relative par bande)

#### Condition Aversive — av-1 (n=263/246, forte puissance statistique)

**Pattern clair et cohérent** : le passage NREM I → NREM II s'accompagne d'un **shift spectral vers les hautes fréquences** dans toutes les régions.

| Région | low (0.2–7 Hz) | theta (7–12 Hz) | beta (15–30 Hz) | high (30–80 Hz) |
|--------|---------------|-----------------|-----------------|-----------------|
| dHPC | ↓ −0.024 *** | ns | ns | ↑ +0.072 *** |
| vHPC | ↓ −0.023 *** | ↓ −0.017 *** | ↑ +0.028 *** | ↑ +0.040 *** |
| BLA | ↓ −0.048 *** | ↓ −0.026 *** | ↑ +0.035 *** | ↑ +0.057 ** |

#### Condition Aversive — av-2 (n=32/26)

Le shift est **concentré dans vHPC** : ↓ low \*\*\* (η²=0.44), ↑ beta \*\*\* (η²=0.35), ↑ high \*\*\* (η²=0.36). Pas de changement significatif en dHPC ni BLA. Direction cohérente avec av-1.

#### Condition Rewarded — rw-2 (n=184/202)

Un **seul effet significatif** : vHPC low ↓ −0.028 \*\*\*. Aucun autre changement. La BLA et dHPC ne montrent aucune différence inter-phases en condition rewarded.

#### Condition Rewarded — rw-1 (n=8/5)

Pas de résultats significatifs (effectif insuffisant pour atteindre la puissance statistique).

---

### 4. Shift REM I → REM II (puissance relative par bande)

#### Condition Aversive — av-1 (n=257/242)

Pattern similaire au NREM mais **moins généralisé**, principalement porté par **vHPC et BLA** :

| Région | low | theta | beta | high |
|--------|-----|-------|------|------|
| dHPC | ↓ −0.012 *** | ns | ns | ns |
| vHPC | ↓ −0.010 ** | ↓ −0.014 *** | ↑ +0.028 *** | ns |
| BLA | ↓ −0.016 *** | ↓ −0.006 * | ↑ +0.023 *** | ns |

#### Condition Aversive — av-2 (n=20/17)

Uniquement **vHPC** : ↓ low \*\*\* (η²=0.55), ↑ high \* (η²=0.21). Cohérent avec av-1.

#### Condition Rewarded — rw-2 (n=178/193)

Un seul effet : vHPC low ↓ −0.023 \*\*\*. Pas d'autres différences.

---

### 5. Comparaison Aversive vs Rewarded

> Les données ci-dessus permettent de comparer indirectement les deux conditions.

**En NREM :**
- La condition **aversive** (av-1) produit un shift NREM I→II robuste et multi-régional (dHPC, vHPC, BLA), avec réduction des basses fréquences et augmentation des hautes fréquences.
- La condition **rewarded** (rw-2) ne produit qu'un effet isolé en vHPC low.
- → **La transition NREM I→II est spectrallement plus marquée dans la condition aversive.**

**En REM :**
- Même pattern : la condition aversive (av-1) montre des changements dans vHPC et BLA, la condition rewarded ne montre qu'un effet en vHPC.
- → **L'effet de la condition sur le shift REM I→II est qualitativement similaire mais quantitativement moins fort qu'en NREM.**

**Région la plus sensible** : **vHPC** — seule région qui montre des changements significatifs dans les deux conditions et les deux comparaisons de phases.

**Région la moins affectée** : **dHPC** en condition rewarded (aucun effet).

---

### Synthèse

| Question | Réponse |
|----------|---------|
| Les deux fichiers aversifs sont-ils comparables ? | **Non** — différences massives, surtout en vHPC (η² grand) |
| Les deux fichiers rewarded sont-ils comparables ? | **Non** — mais rw-1 manque de puissance (n=5–8) |
| Y a-t-il un shift NREM I→II en aversive ? | **Oui** — shift HF (↓ low/theta, ↑ beta/high) dans dHPC, vHPC et BLA (av-1) |
| Y a-t-il un shift NREM I→II en rewarded ? | **Faible** — uniquement vHPC low ↓ (rw-2) |
| Y a-t-il un shift REM I→II en aversive ? | **Oui partiel** — vHPC et BLA (↓ low/theta, ↑ beta) |
| Y a-t-il un shift REM I→II en rewarded ? | **Faible** — uniquement vHPC low ↓ |
| Le shift diffère-t-il entre conditions ? | **Oui** — le shift aversif est plus fort et plus généralisé |
