# Soil organic matter prediction — results notes for the paper

Source: MLflow experiment `Soil_Model_Training_Paper` (id `800855796524282784`), plus one
run from `Soil_Model_Training_v2` (id `385945551149310753`). Compiled 2026-09-25.
Target throughout: `organic_matter_g_kg` (SOM, g kg⁻¹).

All numbers below are read directly from the MLflow store. Where I recomputed something
from the stored per-point predictions (`eval_results.csv`), it is labelled as such.

> **Read the caveats in §6 before writing any of this up.** Three of the five runs differ
> from each other in more than one setting at a time, and the last two runs were evaluated
> on *different* test points from the first three. The headline table (§3) is therefore not
> a clean model ranking; §5 is the defensible comparison.

---

## 1. Dataset and target

| Item | Value |
|---|---|
| Target | `organic_matter_g_kg` |
| Lab records with SOM | 6 000 |
| Mean ± SD | 27.13 ± 11.55 g kg⁻¹ |
| Median (IQR) | 25.0 (19.0 – 34.0) g kg⁻¹ |
| Range | 2.0 – 65.0 g kg⁻¹ |
| Skewness | 0.69 |
| Coefficient of variation | 42.5 % |
| Target transform used by all CNNs | `log1p` |

Static predictor set (25 features): 20 Sentinel-2 barest-earth reflectance layers
(B2–B8, B8A, B11, B12, each with its temporal standard deviation) plus `elevation`,
`slope`, `aspect`, `twi` and `aridity_index`.

Time-series modalities available to the CNNs (channels per step): Sentinel-2 = 11,
Sentinel-1 ascending = 4, Sentinel-1 descending = 4, climate = 3, agronomic = 3, soil = 3.

Split: random hold-out, seed 42, 70 / 15 / 15 (train / validation / test). The number of
eligible points differs between runs because each model family drops points with
unusable inputs, which is what makes the test sets differ (§6.1).

| Run | Eligible points | Train | Val | Test |
|---|---|---|---|---|
| M1, M2, M3 | 5 295 | 3 705 | 795 | 795 |
| M4 | 5 263 | 3 683 | 790 | 790 |
| M5 | 5 046 | 3 532 | 757 | 757 |

---

## 2. The five model configurations

`soil_residual_attention_cnn` is one switchable architecture: a per-modality dilated
temporal CNN encoder, attention fusion over the modality tokens plus a static-feature
token, and a shared head. The four CNN runs differ only in the switches listed here.
Shared hyperparameters: learning rate 9.21 × 10⁻⁴, weight decay 2.80 × 10⁻⁴, dropout 0.6,
batch size 64, AdamW, MSE loss on the `log1p` target, attention `d_model` 128 / 8 heads /
1 layer, plateau LR scheduler on `val_loss`, max 100 epochs.

| ID | Run (model-level id) | Model | Inputs | Targets | Lat/lon encoding | Residual baseline | Uncertainty | Best epoch |
|---|---|---|---|---|---|---|---|---|
| **M1** | `43f5b570` (parent `0bc34f03`) | XGBoost | 25 static only | SOM alone | – | no | 10-member bootstrap | – |
| **M2** | `2a773974` (parent `dc740933`) | soil_residual_attention_cnn | static + time series | SOM alone | **off** (`coord_dim=0`) | no | 10 members, heteroscedastic head | 50 |
| **M3** | `e6a5eb57` (parent `94ede1e5`) | soil_residual_attention_cnn | static + time series | SOM alone | **on** (`coord_dim=2`) | no | off | 20 |
| **M4** | `1c349779` (parent `6b06006c`), SOM child `09255a3f` | soil_residual_attention_cnn | static + time series | **joint, 4**: SOM, pH, CaCO₃, CEC | off (`coord_dim=0`) | no | off | 67 |
| **M5** | `6fc9f8f4` (parent `73410639`, exp. *v2*), SOM child `6fc1444d` | soil_residual_attention_cnn | static + time series | **joint, 6**: SOM, clay, sand, pH, CaCO₃, CEC | **on** (`coord_dim=2`) | **yes** | 10 members, heteroscedastic head | 16 |

XGBoost (M1): 100 trees, max depth 6, learning rate 0.05, ordinal encoding of
categoricals. Fitted on train + validation pooled (4 500 rows) because the bootstrap
uncertainty path uses the `train_val` fit pool — so M1 saw **more** labelled rows than the
CNNs (3 705), not fewer.

### 2.1 How the lat/lon encoding works (M3, M5)

Coordinates are not fed as raw numbers. Each of latitude and longitude is expanded into a
sine/cosine bank at 6 frequencies and concatenated with the two raw values, giving
2 + 2 × 2 × 6 = 26 channels into a dedicated location branch that joins the attention
fusion. The switch is `USE_HARMONIC_COORDS` in `configs/data_spec.yml`; it resolves to the
`coord_dim` hyperparameter recorded in each checkpoint (0 = branch absent, 2 = branch
present). MLflow logs only `coord_dim: auto`, so the values in the table above were read
from the saved checkpoints, not from the logged parameters.

### 2.2 Residual baselines (M5 only)

With `residual_enabled=True` the network predicts a *correction* to an existing per-target
prediction rather than the value itself:

| Target | Baseline column used |
|---|---|
| SOM | `organic_matter_g_kg_base` |
| clay | `clay_median_0_30cm` |
| sand | `sand_median_0_30cm` |
| pH | `ph_median_0_30cm` |
| CaCO₃ | `caco3_pct_total__TabICL` |
| CEC | `c_e_c_meq_100g__TabICL` |

Clay / sand / pH baselines are external gridded medians for 0–30 cm; CaCO₃ and CEC
baselines are predictions from a previously trained TabICL model. See §6.4 — the SOM
baseline column cannot be verified in the current data and is the weakest of the six.

---

## 3. Headline test metrics, as logged

Each row is on that run's own test set, which is what MLflow reports. **These rows are not
mutually comparable across the M1–M3 block and the M4 / M5 rows** (§6.1).

| ID | Configuration | n | R² | RMSE (g kg⁻¹) | MAE (g kg⁻¹) | Bias (g kg⁻¹) | RPD | RPIQ |
|---|---|---|---|---|---|---|---|---|
| M1 | XGBoost, static only | 795 | 0.269 | 9.86 | 7.57 | +0.07 | 1.17 | 1.62 |
| M2 | CNN, no lat/lon encoding | 795 | 0.335 | 9.40 | 6.94 | −1.54 | 1.23 | 1.70 |
| M3 | CNN, lat/lon encoding | 795 | **0.396** | **8.96** | **6.62** | +0.32 | **1.29** | **1.79** |
| M4 | CNN, joint 4 targets | 790 | 0.385 | 8.67 | 6.48 | −0.98 | 1.28 | 1.70 |
| M5 | CNN, joint 6 + residual (v2) | 757 | 0.501 | 7.81 | 6.01 | −0.30 | 1.42 | 1.92 |

M1 cross-validation (5-fold, on the fit pool): RMSE 9.47 ± 0.31 g kg⁻¹; resubstitution
RMSE 6.79 g kg⁻¹ and R² 0.56, i.e. the gradient-boosted trees overfit the static
predictors appreciably.

All five models under-disperse their predictions, which is the usual regression-to-the-mean
signature of a weakly informative predictor set. The ratio of predicted to observed
standard deviation on the test set is 0.64 (M2), 0.72 (M3) and 0.68 (M4, SOM head).

---

## 4. Uncertainty, where it was estimated

Intervals are ±1σ, nominal coverage 68.27 %. Available for M1, M2 and M5 only; M3 and M4
were run with the uncertainty path switched off.

| Metric | M1 XGBoost | M2 CNN no-PE | M5 CNN joint + residual |
|---|---|---|---|
| Mean σ (g kg⁻¹) | 1.96 | 8.45 | 7.86 |
| PICP (target 0.683) | **0.170** | 0.714 | 0.691 |
| Coverage error | −0.513 | +0.032 | **+0.008** |
| MPIW (g kg⁻¹) | 3.92 | 16.90 | 15.72 |
| NMPIW | 0.066 | 0.286 | 0.258 |
| Interval score | 40.46 | 27.61 | **23.28** |
| CRPS | 6.65 | 5.01 | **4.29** |
| NLL | 18.98 | 3.63 | **3.44** |
| ENCE | 4.62 | 0.105 | 0.106 |
| σ–error correlation | 0.130 | 0.252 | 0.198 |

The XGBoost bootstrap ensemble is severely over-confident: it covers 17 % of observations
where 68 % is nominal, and its ENCE is more than forty times the CNNs'. This is expected —
bootstrap spread across ten tree ensembles captures model variance only, not the noise in
the SOM measurement itself. The CNN heteroscedastic heads are close to nominal coverage
(errors of +3.2 and +0.8 percentage points) and calibrated to within ENCE ≈ 0.11. Even so,
the σ–error correlation never exceeds 0.26, so the predicted σ ranks *which* samples are
uncertain only weakly, and interval width should not be read as a per-point reliability
score in the paper.

---

## 5. Fair comparisons, recomputed on shared test points

Recomputed from the stored per-point predictions. Each comparison uses only the points
present in *both* runs, so the two models are scored on identical samples. `p` is a
two-sided Wilcoxon signed-rank test on paired absolute errors.

### 5.1 Earth observation + deep learning vs. static-only trees (identical 795 points)

| Comparison | R² | RMSE | MAE | Δ MAE | p |
|---|---|---|---|---|---|
| M1 XGBoost static | 0.269 | 9.86 | 7.57 | — | — |
| M2 CNN, no lat/lon | 0.335 | 9.40 | 6.94 | −0.63 | **2.1 × 10⁻⁶** |
| M3 CNN, lat/lon | 0.396 | 8.96 | 6.62 | −0.95 (vs M1) | **1.3 × 10⁻⁷** |

Both CNNs beat the static-only baseline on the same points, and both margins are highly
significant. This is the cleanest and best-supported result in the set: the multi-modal
time-series network improves SOM prediction over gradient-boosted trees on static
covariates by 0.46–0.90 g kg⁻¹ RMSE (5–9 % relative), *despite* XGBoost having been fitted
on 4 500 rows against the CNNs' 3 705.

### 5.2 Effect of lat/lon positional encoding (identical 795 points)

| | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| M2, encoding **off** | 0.335 | 9.40 | 6.94 | −1.54 |
| M3, encoding **on** | 0.396 | 8.96 | 6.62 | +0.32 |
| Δ | +0.061 | **−0.44** | **−0.33** | bias nearly removed |

Paired tests: Wilcoxon p = 0.124, paired t p = 0.040. So the improvement is consistent in
every aggregate metric and reaches significance under the parametric test, but not under
the rank test — the gain is **suggestive rather than firmly established** on a single
795-point hold-out. Note also that the encoding-off run carried the heteroscedastic
two-output head and the encoding-on run did not (§6.2), so this is not a single-switch
ablation. And see §6.3: a random hold-out systematically flatters a location-aware model.

### 5.3 Joint multi-target training (identical 136 points)

Comparing like with like — M4 has the lat/lon encoding **off**, so its correct single-target
counterpart is M2, not M3:

| | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| M2, SOM alone (no PE) | 0.468 | 8.21 | 6.13 | −0.11 |
| M4, joint 4 targets (no PE) | 0.463 | 8.25 | 6.27 | +0.49 |
| Δ MAE | | | **+0.14** | Wilcoxon p = 0.78 |

Against the encoding-on single-target model the joint model is also no better
(M3 MAE 5.76 vs M4 6.27 on these points, Wilcoxon p = 0.11).

**Joint training did not improve SOM prediction.** The apparently better headline RMSE for
M4 in §3 (8.67 vs 8.96) comes from an easier test draw, not from the joint objective; once
the same points are scored, the difference vanishes and the sign even reverses. The joint
model does predict the other three properties usefully in the same pass, which is a real
practical benefit, but it should not be claimed as a gain *for SOM*.

M4 per-target test metrics (790 points, for reference):

| Target | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| SOM (g kg⁻¹) | 0.385 | 8.67 | 6.48 | −0.98 |
| pH (water) | 0.430 | 0.41 | 0.31 | −0.02 |
| CaCO₃ (%) | 0.333 | 9.43 | 7.01 | −2.70 |
| CEC (meq/100 g) | 0.559 | 8.62 | 6.40 | −1.38 |

### 5.4 Residual training on baseline predictors (M5)

| Comparison | n | R² | RMSE | MAE | Δ RMSE | p (Wilcoxon / t) |
|---|---|---|---|---|---|---|
| M4 joint → **M5 joint + residual** | 106 | 0.360 → 0.452 | 7.98 → 7.38 | 6.22 → 5.65 | **−0.60** | 0.086 / 0.177 |
| M3 single + PE → **M5** | 102 | 0.484 → 0.476 | 7.88 → 7.94 | 6.28 → 6.33 | +0.06 | 0.742 / 0.880 |

Two different conclusions depending on the reference:

- **Against the joint model without baselines (M4)**, residual training improves SOM by
  0.60 g kg⁻¹ RMSE and 0.57 g kg⁻¹ MAE — a consistent gain, marginal significance
  (p = 0.086), on only 106 shared points.
- **Against the single-target model with lat/lon encoding (M3)**, residual training gives
  **no gain at all** (p = 0.74).

So M5's headline R² of 0.501 in §3 substantially overstates its advantage: most of the gap
to M3's 0.396 is test-draw difference, not method. On matched points M3 and M5 are
statistically indistinguishable for SOM.

M5 per-target test metrics (757 points) explain why:

| Target | Baseline quality | R² | RMSE | MAE |
|---|---|---|---|---|
| CEC (meq/100 g) | TabICL prediction | **0.858** | 4.77 | 3.08 |
| CaCO₃ (%) | TabICL prediction | **0.810** | 5.08 | 3.40 |
| Clay (%) | gridded median | 0.711 | 10.21 | 7.11 |
| Sand (%) | gridded median | 0.678 | 15.06 | 10.87 |
| **SOM (g kg⁻¹)** | see §6.4 — weak | **0.501** | 7.81 | 6.01 |
| pH (water) | gridded median | 0.400 | 0.42 | 0.31 |

Residual training pays off in proportion to how informative the baseline is. SOM has the
weakest baseline of the six and shows the smallest benefit. The very high CEC and CaCO₃
scores should be treated cautiously in the paper, since their baselines are themselves
model predictions from the same data (§6.5) — they are not independent of the labels in
the way a gridded external product is.

---

## 6. Caveats that must be stated in the paper

### 6.1 The test sets are not the same across runs

Points are assigned to the test set by a random draw over whichever points are eligible in
that run. Because the eligible population differs (5 295 / 5 263 / 5 046), the draw
reshuffles entirely even though the seed is fixed at 42:

| | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| M1 | — | 1.00 | 1.00 | 0.09 | 0.07 |
| M2 | | — | 1.00 | 0.09 | 0.07 |
| M3 | | | — | 0.09 | 0.07 |
| M4 | | | | — | 0.07 |

(Jaccard overlap of test-point sets.) M1, M2 and M3 share all 795 points exactly, so
comparisons within that block are clean. M4 and M5 overlap the others by only ~7–9 %,
i.e. effectively independent draws — their headline numbers in §3 carry test-draw noise
of roughly the same size as the effects being claimed. Only 16 points are common to all
five runs, which is why §5 uses pairwise overlaps instead of one shared subset.

**Recommendation:** rerun M4 and M5 with the M1–M3 population, or report only the §5
paired comparisons.

### 6.2 The positional-encoding comparison is not a single-switch ablation

M2 (encoding off) and M3 (encoding on) also differ in:

- the output head — M2 predicts mean and σ (2 outputs, heteroscedastic) and ran a
  10-member ensemble; M3 predicts the mean only (1 output);
- the selected epoch — 50 for M2, 20 for M3.

Everything else in the logged parameter set is identical. Part of the 0.44 g kg⁻¹ RMSE
difference may come from the extra σ output competing with the mean for capacity, and from
the different early-stopping point, rather than from location awareness. For a publishable
ablation, rerun M2 and M3 with the uncertainty setting matched.

### 6.3 A random hold-out flatters location-aware models

The project's own configuration documentation flags this: a model that knows *where* a
point is can exploit spatial autocorrelation — nearby points have similar SOM — rather than
learning the relationship between the remote-sensing signal and SOM. All five runs used
`split.strategy: random`, so the two encoding-on models (M3, M5) are evaluated under the
condition that most favours them. Any claim that positional encoding improves SOM
prediction should be repeated under `spatial_group` splitting before publication;
otherwise the honest framing is "improves interpolation within the sampled area", not
"improves prediction".

### 6.4 The SOM residual baseline cannot be verified

M5 used the column `organic_matter_g_kg_base`, which no longer exists in
`Morocco_soil_targets_uuid_roadsafe_6ksample_V_310826_base.csv`; the file now has
`organic_matter_g_kg_baseline`, and the rename is an uncommitted edit to
`configs/data_spec.yml`. Whether the values changed with the rename is unknown.

Scored on M5's own test points, the *current* `organic_matter_g_kg_baseline` column is a
poor standalone SOM predictor: R² = −0.75, RMSE 14.63 g kg⁻¹, MAE 11.19 g kg⁻¹, bias
−6.67 g kg⁻¹, correlation with observed SOM 0.305. It carries some signal but is badly
mis-calibrated, which the residual head can partly absorb. If this is indeed the column
M5 used, the framing is "the network corrects a strongly biased prior", and the small
SOM gain in §5.4 is unsurprising. **Confirm which column the run actually read before
describing the baseline in the paper.**

### 6.5 The CaCO₃ and CEC baselines are model predictions on the same data

`caco3_pct_total__TabICL` and `c_e_c_meq_100g__TabICL` come from a TabICL model trained on
this dataset. Unless that model's training points were excluded from M5's test set, those
two targets' R² values (0.81, 0.86) are optimistic. This does not affect the SOM result but
does affect any table that presents all six targets together.

### 6.6 Other points

- M1 was fitted on train + validation pooled (4 500 rows) while the CNNs used 3 705 train
  rows. State this, since it makes the CNN-over-XGBoost result conservative rather than
  inflated.
- M2 and M5 report ensemble-based uncertainty but their point predictions come from the
  single best checkpoint, not from the ensemble mean. Member test losses for M2 span
  0.219–0.270 (standardised `log1p` space), indicating meaningful run-to-run variability;
  a mean-of-ensemble prediction would likely score slightly better than the numbers in §3.
- M5 comes from a different experiment (`Soil_Model_Training_v2`, 2026-09-13, commit
  `6125ea2`) and a different code state from the four `Soil_Model_Training_Paper` runs
  (2026-09-24/25, commit `29b4e07`). The intervening commits include a unification of the
  `soil_cnn` architecture variants, so M5's network is not guaranteed to be identical to
  M2–M4's even where the logged switches agree. This is an additional reason to treat the
  M5-vs-M3 comparison in §5.4 as indicative only.
- SOM is never predicted better than R² ≈ 0.5 / RPIQ ≈ 1.9 in any configuration. By the
  common spectroscopy convention (RPIQ < 2.0 = poor to fair), every model here is at best
  fair for SOM, and should be described as suitable for relative mapping rather than
  quantitative site-level estimation.

---

## 7. Suggested narrative for the results section

1. SOM is the hardest of the six properties studied, with high variability
   (CV 42.5 %) and modest predictability throughout (R² 0.27–0.50).
2. Replacing static covariates and gradient-boosted trees with a multi-modal
   time-series network improves SOM prediction significantly and reproducibly
   (RMSE 9.86 → 9.40–8.96 g kg⁻¹ on identical test points, p < 10⁻⁵). **This is the
   headline result.**
3. Adding a harmonic lat/lon encoding lowers RMSE by a further 0.44 g kg⁻¹ and removes a
   −1.5 g kg⁻¹ bias, but the gain is only marginally significant and is obtained under a
   random hold-out that favours location-aware models. Present it as a promising avenue
   that needs spatial-block validation, not as an established improvement.
4. Joint prediction of SOM with pH, CaCO₃ and CEC does **not** improve SOM itself; its
   value is that four properties come from one model at one inference cost.
5. Residual training on existing baseline predictions helps where the baseline is
   informative (CEC, CaCO₃, clay, sand) and helps SOM little, because the available SOM
   baseline is weak and biased. This is a useful, reportable negative-to-neutral finding.
6. Uncertainty from the CNN heteroscedastic head is well calibrated in aggregate
   (coverage error ≤ 3 percentage points, ENCE ≈ 0.11), unlike the bootstrap intervals from
   XGBoost (17 % coverage against 68 % nominal). Per-point discrimination is weak in both
   (σ–error correlation ≤ 0.26).

## 8. Run identifiers for the methods section

| ID | Experiment | Parent run | Model run |
|---|---|---|---|
| M1 | Soil_Model_Training_Paper | `0bc34f03a1d9441c82fbb1692f4e62b2` | `43f5b570a56a4e6789de8285e4027bfc` |
| M2 | Soil_Model_Training_Paper | `dc74093394c0405eb5af05e655e822ef` | `2a773974b44a47598bc5416940b75269` |
| M3 | Soil_Model_Training_Paper | `94ede1e5e8884f93ab0aeb038003e8a4` | `e6a5eb5769f645f68e7b17450bc504a5` |
| M4 | Soil_Model_Training_Paper | `6b06006ced324dce85441d8b2dab62bc` | `1c349779320c442481eee9110a629ec4` (SOM child `09255a3f8a4549c0bfdfbd9bdeb8ce62`) |
| M5 | Soil_Model_Training_v2 | `73410639b8a4417395d5bec247b6540a` | `6fc9f8f45a0e4330b061d2e6c5f08374` (SOM child `6fc1444d4cda48579a8d6e8cdcb2e520`) |

Code state: M1–M4 ran at commit `29b4e07` (2026-09-23); M5 ran at commit `6125ea2`
(2026-09-13, "fix: tuning results reproducibility in final training"), ten days and several
commits earlier. Split seed 42, random hold-out, 70/15/15 throughout.
