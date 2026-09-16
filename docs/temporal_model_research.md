# Temporal extension of Prithvi-UNet: literature review and architecture decisions

> **2026-09-15 provenance correction:** Historical Phase-1 foundation
> provenance remains unresolved. Current ECCC/Phase-1 shape incompatibility does
> not prove random historical initialization, and weight histograms do not prove
> untrained or useless time embeddings. The general/rollout cadence difference
> is a transfer limitation, not a proof of impossibility. The original statements
> below are retained as the handoff record and superseded on these points by
> [the takeover evidence report](temporal_pretrained_transfer_takeover.md).


**Review date: 2026-09-14.** Branch: `Prithvi-UNet_temporal_model`, based on
`Prithvi-UNet-stochastic_refinement` @ `dd48c6f`.

This document separates three things, and labels every claim as one of them:

| Tag | Meaning |
|---|---|
| **[VERIFIED]** | Read from the primary source (arXiv HTML/LaTeX, official repo file, or model card) and quoted. |
| **[MEASURED]** | Measured by us on the actual data or code in this repository, with the command recorded. |
| **[OUR ADAPTATION]** | A design decision of ours. Not endorsed by any cited paper. |

Verification method for every arXiv ID: `curl` the abstract page, require HTTP 200,
then parse `citation_title` / `citation_author` / `citation_date` from the returned
HTML. A known-bad control (`2412.99999`) was fetched repeatedly and interleaved
with the real requests; it returns 404, so the test discriminates. **Note:** a
summarizing fetch tool returns plausible-looking prose for a 404 page, so status
code plus a negative control is the only trustworthy existence test. Two IDs in
the original brief were flagged internally as implausible on the "arXiv numbers
rarely exceed ~20000/month" heuristic; **that heuristic is out of date and both
IDs are real.**

---

## 1. Verified references

### 1.1 The four papers named in the brief

| ID | Real title | v1 | Venue | Code |
|---|---|---|---|---|
| [2408.06400](https://arxiv.org/abs/2408.06400) | *MetMamba: Regional Weather Forecasting with Spatial-Temporal Mamba Model* — Qin, Chen, Jiang, Sun, Ye, Lin | 12 Aug 2024 (v2 14 Aug 2024) | **none stated** | **none** |
| [2501.11238](https://arxiv.org/abs/2501.11238) | *WSSM: Geographic-enhanced hierarchical state-space model for global station weather forecast* — Yang, Liu, Shi, Zou | 20 Jan 2025 | **none stated** | **none** |
| [2605.14606](https://arxiv.org/abs/2605.14606) | *MambaRain: Multi-Scale Mamba-Attention Framework for 0-3 Hour Precipitation Nowcasting* — Shi et al. (13 authors) | **14 May 2026** | **none stated** | project page only |
| [2606.29172](https://arxiv.org/abs/2606.29172) | *CORDEX-ML-Bench: A Benchmark for Data-Driven Regional Climate Downscaling — Experiment Design and Overview* — Rampal, González-Abad, Addison, Baño-Medina, … Gutiérrez (37 authors) | **28 Jun 2026** | **none stated** | [WCRP-CORDEX/ml-benchmark](https://github.com/WCRP-CORDEX/ml-benchmark) |

All four resolve and all four match their labels. **None has a peer-reviewed
venue in its arXiv metadata; cite all four as preprints.** [VERIFIED]

### 1.2 Foundations

| ID | Title | v1 | Venue | Code |
|---|---|---|---|---|
| [1506.04214](https://arxiv.org/abs/1506.04214) | *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting* — Shi, Chen, Wang, Yeung, Wong, Woo | 13 Jun 2015 (v2 19 Sep 2015) | **NIPS 2015** (confirmed on `papers.nips.cc`) | **none official** |
| [2312.00752](https://arxiv.org/abs/2312.00752) | *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — Gu, Dao | 1 Dec 2023 (v2 31 May 2024) | none stated | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| [2405.21060](https://arxiv.org/abs/2405.21060) | *Transformers are SSMs: … Structured State Space Duality* — Dao, Gu (order flipped vs Mamba-1) | 31 May 2024 | **ICML 2024** (in arXiv `Comments`, corroborated by repo BibTeX) | same repo |
| [2409.13598](https://arxiv.org/abs/2409.13598) | *Prithvi WxC: Foundation Model for Weather and Climate* — Schmude et al. (29 authors) | 20 Sep 2024, **v1 only** | none stated | [NASA-IMPACT/Prithvi-WxC](https://github.com/NASA-IMPACT/Prithvi-WxC) |

[VERIFIED]

---

## 2. What Prithvi-WxC actually does with time, and what survives in this repo

### 2.1 Prithvi-WxC's temporal mechanisms [VERIFIED]

From `PrithviWxC/model.py`:

```python
self.input_time_embedding = nn.Linear(1, embed_dim//4, bias=True)
self.lead_time_embedding  = nn.Linear(1, embed_dim//4, bias=True)

def time_encoding(self, input_time, lead_time):
    input_time = self.input_time_embedding(input_time.view(-1, 1, 1, 1))
    lead_time  = self.lead_time_embedding(lead_time.view(-1, 1, 1, 1))
    return torch.cat((torch.cos(input_time), torch.cos(lead_time),
                      torch.sin(input_time), torch.sin(lead_time)), axis=3)
```

* Two scalars in **hours**: `δτ` = spacing between the two input states
  (`input_time`, negative), `δt` = forecast lead time (`lead_time`).
* Conditioning is **purely additive** at the token level, consumed once before
  the encoder: `tokens = x_embedded + static_embedded + time_encoding`. There is
  **no FiLM / adaLN path anywhere in the backbone.**
* Native input axis: `input_size_time: 2`, flattened before patch embedding
  (`[batch, time, parameter, lat, lon] -> [batch, time × parameter, lat, lon]`).
* Trained support is **discrete and sub-daily**: pretraining drew `input_time`
  from `[-3, -6, -9, -12]` h and `lead_time` from `[0, 6, 12, 24]` h; the rollout
  checkpoint fixes both to 6 h. MERRA-2 is 3-hourly.
* Rollout is a plain Python loop in `rollout.py`; the model itself is single-step
  and `lead_time` is *not* incremented across steps.
* 2300M config: `embed_dim: 2560`, `n_blocks_encoder: 12`, `n_blocks_decoder: 2`,
  `patch_size_px: [2,2]`, `mask_unit_size_px: [30,32]`.
  ⚠️ The paper says "25 encoder and 5 decoder blocks" because the code builds
  `2·n_blocks+1` blocks. Do not write `n_blocks_encoder: 25` into a config.

### 2.2 None of it is present in this repository [MEASURED]

```
$ python -c "import inspect, PrithviWxC.model as M; \
    print(inspect.signature(M.PrithviWxCEncoderDecoder.__init__)); \
    src=inspect.getsource(M.PrithviWxCEncoderDecoder); \
    print([k in src for k in ('input_time','lead_time','time_encoding','climate')])"
(self, embed_dim, n_blocks, mlp_multiplier, n_heads, dropout, drop_path, shifter=None, transformer_cp=[])
[False, False, False, False]
```

`granitewxc/models/model.py::get_finetune_model_UNET` builds
`PrithviWxCEncoderDecoder`, whose `forward(x)` takes **only** the token tensor.
There is no `input_time`, no `lead_time`, no `time_encoding`, no climatology path.
Both target cases set `n_input_timestamps: 1`.

Three consequences that shaped the design:

1. **There is no pretrained temporal conditioning to reuse.** A temporal pathway
   has to be added, not re-enabled.
2. **There is no axis-conflation risk.** The backbone's native multi-timestamp
   axis has extent 1 and carries no time metadata, so the outer context sequence
   we add is unambiguously a separate axis. We never merge them and never fake
   history by duplicating a snapshot into the native axis.
3. **Even if the mechanism were present it would not transfer.** Its trained
   support is ±3–12 h at 3-hourly cadence; our task is daily. A learned embedding
   of "−6 hours" says nothing about "−1 day".

---

## 3. Mamba: what the papers say about Δ, and where we depart

### 3.1 Δ is a learned gate, not physical time [VERIFIED]

Zero-order-hold discretization, Mamba-1 Eq. (4):

> **Ā** = exp(Δ**A**)   **B̄** = (Δ**A**)⁻¹(exp(Δ**A**) − **I**) · Δ**B**

Δ is produced input-dependently: `s_Δ(x) = Broadcast_D(Linear_1(x))`,
`τ_Δ = softplus`. In code (`mamba_simple.py`) it is a rank-`dt_rank` bottleneck
with the bias initialized as the inverse softplus of a log-uniform draw in
`[dt_min, dt_max] = [0.001, 0.1]`.

§3.5.2, verbatim:

> **Interpretation of Δ.** In general, Δ controls the balance between how much to
> focus or ignore the current input x_t. It generalizes RNN gates … mechanically,
> a large Δ resets the state h and focuses on the current input x, while a small Δ
> persists the state and ignores the current input.

**Δ is dimensionless and carries no calendar units. Nothing in either paper lets
you feed a physical Δt.** [VERIFIED]

### 3.2 Mamba-2 / SSD structural restriction [VERIFIED]

> • The structure on A is further simplified from diagonal to **scalar times
> identity** structure. Each A_t can also be identified with just a scalar…
> • We use a larger head dimension **P** … Typically **P = {64,128}**

Buys "**2−8× faster** than the optimized selective scan implementation of Mamba,
while … allowing for much larger recurrent state sizes". In `modules/mamba2.py`
there is **one Δ per head** and a single fused `in_proj` of width
`2*d_inner + 2*ngroups*d_state + nheads`.

### 3.3 Our departure [OUR ADAPTATION]

`granitewxc/temporal/backends.py::TemporalSSDBlock` multiplies the *learned* Δ by
the **measured** interval ratio `r = Δt_actual / cadence`:

```python
dt = F.softplus(dt_raw + self.dt_bias)          # learned, dimensionless
if self.time_scale_from_metadata and interval_ratio is not None:
    dt = dt * ratio                              # r = Δt_actual / cadence
decay = torch.exp(dt * A)                        # A = -exp(A_log), per head
```

Justification: the block is a ZOH discretization of a continuous-time linear
system, so integrating over an interval `r` times longer means using `Δ·r`. A step
across a removed 29 February therefore decays the state by `exp(2·Δ·A)` rather
than `exp(Δ·A)`. The learned timescale and the physical interval remain separate
quantities that are *multiplied*, never conflated.

**This is ours, not the papers'.** It is exactly the "you'd have to inject it
yourself" case. It is configurable (`temporal.mamba.time_scale_from_metadata`) so
it can be turned off and compared.

We adopt Mamba-2's simplified `B̄ = Δ·B` (not Mamba-1's full ZOH `B̄`), scalar-×-identity
`A` per head, and one Δ per head. `headdim: 32` is smaller than the papers' 64/128
because our latent is 256 channels wide, not 2048.

### 3.4 Kernels [VERIFIED / MEASURED]

`mamba_ssm` ships pure-PyTorch reference paths (`selective_scan_ref`,
`ssd_minimal_discrete` — "the same as Listing 1 from the paper"), but
`ssd_minimal.py` imports Triton at module top level, and current `main` installs
"without compiling the `selective_scan_cuda` extension" by default.

[MEASURED] In this environment `mamba_ssm` is not importable at all:

```
$ python -c "import mamba_ssm"
ModuleNotFoundError: No module named 'mamba_ssm'
```

and the platform is Windows/CUDA 12.8 (torch 2.11.0+cu128), where upstream's
build path is unsupported. We therefore write the SSD recurrence out explicitly
in-repo. `temporal.mamba.implementation` is `fused` (require `mamba_ssm`, raise
`MambaUnavailableError` if absent), `reference` (in-repo pure PyTorch), or `auto`
(prefer fused, fall back with a loud warning and a recorded provenance field).
**It never substitutes a different architecture** — the shipped configs set
`reference` explicitly so the run record is unambiguous.

---

## 4. Scan axis: what the weather-Mamba papers actually do

This is the single most important finding for the architecture, and it is
negative.

### 4.1 MetMamba scans a jointly-flattened (T, H, W) [VERIFIED]

> **Scan routes** To flatten a spatial-temporal block of $(T,H,W)$ for Mamba-3D to
> scan, there exists 6 permutations of this block and 6 corresponding traversal
> routes … **making a total of 12 routes**.

with `T = 2` (two initial conditions, 6 h apart), so the receptive-field work is
overwhelmingly spatial. The authors are candid that the flattening is a poor
spatial operator:

> we suspect this scan pattern and implementation is **insufficient to exploit the
> spatial relationship** but good enough to spatial temporal information

Cadence 6-hourly, East Asia, 0.25° (**inferred** — the paper never states grid
spacing; derived from "28 grid points (~780 km)"). Loss: **MSE only**, latitude-
and level-weighted. Cost: MetMamba-3D runs at 1.72 samples/s vs MetSwin's 6.02,
i.e. ~3.5× slower than the Swin baseline it beats.

### 4.2 MambaRain never states its scan axis [VERIFIED]

It *asserts* a clean division of labour —

> Mamba blocks capture global temporal evolution … while self-attention
> complements this by **explicitly modeling spatial correlations** … a capability
> **inherently absent in Mamba's sequential processing**

— but the only definition given is "output features $X_l$ with dimensions
$L\times D$, where **$L$ represents the sequence length**", and `L` is never
defined. Searches for "temporal axis", "time axis", "reshape", "flatten" return
zero hits.

Its `ViM` layer is **MambaVision** ([2407.08083](https://arxiv.org/abs/2407.08083)),
an *image* backbone that scans flattened *spatial* tokens and whose abstract says
attention was added in the final layers precisely "to capture **long-range spatial
dependencies**". MambaRain reuses MambaVision's Mamba→attention ordering while
relabelling the Mamba part "temporal" and the attention part "spatial" —
**inverting the rationale of the block it borrows.**

**Conclusion: MambaRain is not usable evidence for a time-axis SSM scan.** It is
cited here for its loss (§6) and its ConvGRU comparison (§5), not for its scan.

### 4.3 WSSM scans time, but there is no spatial axis at all [VERIFIED]

Task is Global **Station** Weather Forecasting: multivariate 1-D time series at a
100-station subset of Weather-5K, hourly, 48 h input, 48/72/120 h horizons.
"Geographic enhancement" is lon/lat/elevation pushed through three linear layers.
**There is no spatial grid.** The scan is over time, bidirectionally, at multiple
temporal patch scales (`(16,8), (8,4), (2,1)`).

The widely-quoted line —

> early investigations have revealed that Mamba **underperforms** compared to
> Transformer-based state-of-the-art methods in the context of GSWF [2]

— is (a) about per-station 1-D series, so it says nothing about spatially
organized memory, and (b) a second-hand citation to Weather-5K
([2406.14399](https://arxiv.org/abs/2406.14399)), not their own experiment.

⚠️ Two internal inconsistencies make WSSM's numbers unreliable as printed:
"up to 90% (0.21 → 2.81)" is a ~13× change, not 90%; and the results table lists
horizons "48, 72, **128**" where the text says 48/72/**120**.

### 4.4 Our choice [OUR ADAPTATION]

None of the three published weather/climate SSM designs is a clean time-axis scan
over a spatially organized latent. Ours is:

```
latent [B, C, H, W]
  ├─ spatial mixing : depthwise 3×3 + pointwise 1×1, applied per frame (residual)
  └─ SSM scan       : (H, W) folded into batch; scan runs over t, one C-vector per cell
       repeated for n_layers, so spatial and temporal coupling interleave
```

Documented in `docs/temporal_model_architecture.md` §3. Flattening `H·W` into the
scan axis — what VMamba/Vim/MambaVision do, and what MambaRain most likely does —
would model **no temporal dependence at all**, which is explicitly not what this
branch is for.

---

## 5. ConvGRU/ConvLSTM vs Mamba for spatially organized memory

**There is no dedicated head-to-head benchmark.** The evidence lives inside other
papers' baseline tables and is mixed. We do not manufacture a winner; we implement
both and measure.

**Pure Mamba loses to the ConvGRU family on gridded radar** [VERIFIED] — from
MambaRain's own tables (SWAN radar, SE China, CSI at ≥20 / ≥30 dBZ):

| model | CSI ≥20 | CSI ≥30 | params | latency |
|---|---|---|---|---|
| TrajGRU | **0.470** | **0.339** | 11.92 M | 3965 ms |
| MambaUnet (pure Mamba U-Net) | 0.412 | 0.284 | 8.97 M | 613 ms |

Replicated on their Xinjiang domain. Caveat: single paper, written by Mamba
proponents, and `MambaUnet` is their own re-implementation.

**The 1-D-scan/2-D-field mismatch is universally acknowledged** [VERIFIED]:
VMamba ([2401.10166](https://arxiv.org/abs/2401.10166), NeurIPS 2024) adds four
scan routes to "bridge the gap between the ordered nature of 1D selective scan and
the non-sequential structure of 2D vision data"; Vim
([2401.09417](https://arxiv.org/abs/2401.09417), ICML 2024) cites
"position-sensitivity"; Mamba-ND ([2402.05892](https://arxiv.org/abs/2402.05892))
benchmarks ERA5 against bidirectional LSTMs and reports being "**competitive
with** the state-of-the-art", not better.

**The best designs are hybrids** [VERIFIED]. VMRNN
([2403.16536](https://arxiv.org/abs/2403.16536)) keeps ConvLSTM-style recurrent
gating and swaps the content transform for Mamba — "*VMRNN Module removes all
weights W and biases b in ConvLSTM*" — and tops Moving MNIST (MSE 16.5 vs
ConvLSTM's 103.3, SwinLSTM's 17.7).

**But recurrence is not obviously required either** [VERIFIED]: OpenSTL
([2306.11249](https://arxiv.org/abs/2306.11249), NeurIPS 2023), the standard
fair-comparison harness including weather forecasting, finds "**Surprisingly** …
recurrent-free models achieve a good balance between efficiency and performance
than recurrent models."

There is **no survey of Mamba for weather/climate/earth-system modelling**.

### Decision [OUR ADAPTATION]

* **ConvGRU is the default backend.** One state tensor instead of ConvLSTM's two
  (halves the state that must be held across a truncated-BPTT boundary and across
  every inference chunk) and ~25% fewer gate parameters, with no consistent
  accuracy disadvantage in the literature that introduced both. `cell: convlstm`
  is implemented for comparison.
* **The Mamba backend is a hybrid, not pure Mamba** — interleaved convolutional
  spatial mixing plus a time-axis SSM. Given §5's evidence that *pure* Mamba
  underperforms ConvGRU on gridded fields while hybrids lead, shipping pure Mamba
  would have been shipping the configuration the literature says loses.
* Both are compared under an identical pipeline; the two case YAMLs differ in
  exactly one semantic key (`tests/test_temporal_configs.py` asserts this).

---

## 6. Losses: the spectral-loss trap, and what we do instead

The brief's warning is exactly right, and MambaRain is the instance. Its Eq. (6),
verbatim:

> $\mathcal{L}_{\text{spectral}}=\frac{1}{N}\sum_{n=1}^{N}\left\|\mathcal{F}(\hat{\mathbf{y}}_{n})-\mathcal{F}(\mathbf{y}_{n})\right\|_{2}^{2}$
> where $\mathcal{F}(\cdot)$ denotes the **2D FFT**

[VERIFIED] Three properties, checked directly against the text:

* **Not wavenumber-weighted.** No `k`-dependent weight; the string "wavenumber"
  appears **zero** times in the paper.
* **Complex, not magnitude.** The difference is taken between raw complex spectra
  *before* the norm.
* **No combination weight with MSE is given.** No total-loss equation, no `λ`;
  "total loss"/"lambda" return zero hits.

By Parseval's theorem the squared error of the complex Fourier coefficients equals
the spatial squared error up to a constant, so **as written this term is the MSE it
is claimed to fix**, and it cannot be an independent anti-blurring mechanism. Their
ablation does show it helps (Avg. CSI 0.263 → 0.321), which means the benefit must
come from the implicit reweighting of their particular normalization, not from the
stated mechanism.

**We do not implement it.** `granitewxc/temporal/losses.py` says so in its module
docstring. Precedents for a spectral term that *does* add information, both
wavenumber-binned and magnitude-based:

* **RALSD** from CORDEX-ML-Bench itself [VERIFIED]:
  $\mathrm{RALSD}=\sqrt{\frac{1}{K}\sum_{k}[\log S_{Y}(k)-\log S_{\hat{Y}}(k)]^{2}}$,
  "where $k$ indexes the **radial wavenumber** and $K$ is the total number of
  wavenumber bins", on radially averaged 2-D power spectral density. Used there as
  a *metric*, not a loss.
* Yan et al., Fourier amplitude and correlation loss, NeurIPS 2024 (cited but not
  adopted by MambaRain).

The existing repo already has scale-selective terms (`loss.multiscale`,
`loss.spatial_gradient`) that target the same thing directly, so we layer temporal
terms on top of them rather than adding a redundant spectral term.

### Our temporal terms [OUR ADAPTATION]

Each matches an *observed* statistic; none penalizes variability. The distinction
is pinned by tests that use an explicit adversary:

| Term | Definition | Adversary it must reject | Test |
|---|---|---|---|
| tendency | `|Δŷ − Δy| / Δt` | a constant field | `test_tendency_loss_does_not_prefer_a_constant_field` |
| accumulation | `k`-day running totals matched | a correctly-totalled but mistimed event | `test_accumulation_loss_detects_mistimed_events` |
| occurrence | `P(wet)` **and** `P(wet\|wet)`, `P(wet\|dry)` matched | drizzle everywhere (right frequency, no persistence) | `test_occurrence_loss_penalizes_drizzle_everywhere` |
| lag autocorrelation | lag-`k` ACF of anomalies matched, gated by `min_samples` | white noise | `test_lag_autocorr_loss_rewards_matching_persistence` |
| Tmax/Tmin | `relu(tmin − tmax)` | — (zero when consistent) | `test_tmax_tmin_consistency_only_penalizes_violations` |

`|Δŷ|` (penalizing the model's own change) would be a smoothness prior whose
minimum is a constant field. We implement `|Δŷ − Δy|`, whose minimum is the
correct evolution.

---

## 7. Temporal dependence in daily downscaling: the state of the field

### 7.1 The gap this branch fills [VERIFIED]

**CORDEX-ML-Bench has no temporal-dependence metric at all.** Its core metric set
is RMSE, climatological mean, SDII, Rx1day, TXx, RALSD, LHD, PSS, interannual-
variability RMSE and climate-change-signal error. A full-text search for "temporal
consistency", "wet spell", "dry spell", "autocorrelat", "lag", "autoregress",
"previous day", "LSTM", "recurren" returns only `GNN4CD [ICTP]`'s "GRU-based
temporal encoder" and the interannual-variability metric. **No lag-1
autocorrelation, no spell-length distribution, no transition probability, no
day-to-day variability diagnostic.**

And the benchmark includes an entrant named **"Prithvi-UNet [JPL]"**:

> fine-tunes NASA's Prithvi-WxC geoscience foundation model with a convolutional
> U-Net decoder, fine-tuning only selected layers, requiring only 10 epochs per
> experiment. Coarse predictors are bilinearly interpolated to the high-resolution
> grid prior to input… **This is the only foundation model algorithm in the
> benchmark.**

That is this repository. Its 40 configurations map day → day independently.

Two of the benchmark's own findings bear directly on our acceptance criteria:

> nearly all 40 models **systematically underestimate mid-century climate-change
> signals** for temperature (TXx) and precipitation (Rx1day) extremes

> **pixelwise daily RMSE is a poor proxy for overall downscaling skill.** Models
> with low RMSE often rank poorly on climatological and extreme-value metrics, and
> vice versa.

The second is why our scorecard scores tendency error, autocorrelation, spell
statistics and extremes *alongside* RMSE, rather than reporting RMSE alone.

### 7.2 The most on-topic paper [VERIFIED]

**EnScale — [2509.26258](https://arxiv.org/abs/2509.26258)**, Schillinger, Samarin,
Shen, Knutti, Meinshausen. v1 30 Sep 2025, **v3 10 Apr 2026**. Code:
[m-schillinger/enscale](https://github.com/m-schillinger/enscale). EURO-CORDEX,
**daily** tas/pr/sfcWind/rsds, EUR-11, 8 GCM–RCM pairs, 1971–2099. Loss is the
**energy score** (multivariate CRPS generalization), not diffusion.

> Temporal consistency of downscaled data is important for accurate
> representation of multi-day statistics… **Yet, most existing methods do not
> consider temporal consistency.**

> To our knowledge, **EnScale-t is the first downscaling approach to model
> temporal dependencies in the high-resolution output**.

Its §5.5 gives the quantitative diagnosis we treat as the target signature:

> **EnScale underestimates local autocorrelation for all variables**, with the
> effect being strongest for temperature.

> **Underestimating autocorrelation can lead to an underrepresentation of
> variability in multi-day averages, therefore potentially underestimating risk.**

Its signed lag-1 ACF error table (tas / pr / sfcWind / rsds): NN-det
+0.01/+0.17/+0.08/+0.11 · EasyUQ −0.03/−0.07/−0.15/−0.03 · Analogues
−0.10/−0.12/−0.29/−0.05 · GAN −0.06/−0.10/−0.21/+0.02 · CorrDiff
+0.01/+0.22/+0.10/+0.09 · CorrDiff-s −0.06/−0.09/−0.21/−0.04. EnScale-t best on
all four.

⚠️ Note the sign heterogeneity: deterministic and shared-noise methods
*over*-estimate ACF, per-day-independent stochastic methods *under*-estimate it.

[MEASURED] Our own SA baseline sits on the **over**-estimating side, measured on the
held-out 1981–1983 test period (`examples/CORDEX_ML/runs_temporal/experiment/baseline/evaluation.json`):

| | `pr` | `tasmax` |
|---|---|---|
| lag-1 ACF error (pred − truth) | **+0.0249** | +0.0049 |
| lag-2 ACF error | +0.0141 | +0.0004 |
| wet-day frequency error | **+0.1034** | — |
| `P(wet｜wet)` error | **+0.1435** | — |
| mean wet-spell length error (days) | **+1.04** | — |
| day-to-day tendency std ratio (pred/truth) | **0.548** | 0.929 |
| field std ratio (pred/truth) | 0.622 | 1.007 |
| 99th-percentile bias | **−18.7 mm/day** | −0.17 K |

So our objective is reducing `|error|`, **not** increasing persistence — which is why
every acceptance criterion is written on the **absolute** ACF error, and why a term
that simply made the output more persistent would be scored as a regression.

Two things this table makes concrete:

1. The baseline's precipitation problem is **not primarily RMSE**. It has 10
   percentage points too many wet days, wet spells over a day too long, and
   day-to-day variability at 55% of observed — the classic drizzle-and-oversmooth
   signature. That is exactly what the occurrence/transition objective targets, and
   RMSE does not measure it at all. It is also consistent with CORDEX-ML-Bench's own
   finding that "pixelwise daily RMSE is a poor proxy for overall downscaling skill".
2. `tasmax`'s lag-1 ACF error is already only +0.0049. A pre-registered "reduce by
   ≥10%" criterion on that quantity asks for a 0.0005 change, which is at or below
   the noise floor of the estimate. That threshold turned out to be ill-calibrated;
   see `docs/temporal_model_results.md`, where it is reported as such rather than
   moved after the fact.

### 7.3 The failure mode, stated by the practitioners themselves [VERIFIED]

**[2509.21844](https://arxiv.org/abs/2509.21844)** — Lewis, Rampal, Gibson,
Harrington, Holgate, Ukkola, Maher, *Generative AI-Downscaling of Large Ensembles
Project Unprecedented Future Droughts*, 26 Sep 2025. Written by the group whose own
AI-downscaled 12 km ensembles it uses:

> precipitation and temperature are downscaled independently of each other, with
> **each day being downscaled independently of any other timestep**. This means
> that **the land surface has no persistent memory of the soil moisture state**…

> Moreover, the independent simulation of each day has implications for the
> **simulation of persistent synoptic systems**. Architectures with a 'memory' of
> previous states (e.g., autoregressive approaches), could possibly better
> represent these systems, and **represent an important gap in the literature**.

**[2412.15361](https://arxiv.org/abs/2412.15361)** (Schmidt et al., *npj Clim
Atmos Sci* 8, 2025, DOI `10.1038/s41612-025-01157-y`) frames it as: current methods
infer weather as "**temporally decoupled spatial patches**", and

> **Inconsistencies between downscaled time steps and variables make predicted
> time series as a whole physically implausible.**

⚠️ Hourly, not daily (8×8 → 128×128, 1-hourly), so architecturally adjacent but not
a cadence match.

**The classical literature is where the spell-statistics evidence actually
lives** [VERIFIED, via Crossref]:

* Maraun (2013), `10.1175/jcli-d-12-00821.1`: "the spatial and temporal structure
  of the corrected time series is misrepresented, **the drizzle effect for area
  means is overcorrected**…"
* Maraun, Huth, Gutiérrez et al. (2019), **the VALUE perfect-predictor temporal
  experiment**, `10.1002/joc.5222` — spell-length distributions, transition
  probabilities, short-term and interannual variability across ~50 SD methods.
  **This is the reference intercomparison for exactly our diagnostics.**
* Vrac & Thao (2020), `10.5194/gmd-13-5367-2020`: most multivariate bias-correction
  methods "do not correct – and **sometimes even degrade** – the associated
  temporal features."
* Robin & Vrac (2021), `10.5194/esd-12-1253-2021`: adding more lags does **not**
  monotonically improve temporal properties — a direct caution against a large
  `context_length`.

⚠️ **Genuine gap:** no paper documents biased wet/dry spell lengths or transition
probabilities *specifically as a failure of deep-learning frame-independent
downscaling*. `"wet spell precipitation downscaling"` returns zero arXiv hits. The
diagnosis exists only in the classical literature; the DL side has the capability
demonstrated (DiffESM evaluates "hot streaks or dry spells" via ACF/PACF) but not
the failure documented.

### 7.4 Recurrent downscaling precedents [VERIFIED]

* **[2005.10374](https://arxiv.org/abs/2005.10374)** Leinonen, Nerini, Berne,
  *Stochastic Super-Resolution for Downscaling Time-Evolving Atmospheric Fields
  with a GAN*, **IEEE TGRS 59(9) 7211–7223, 2021**, code
  [jleinonen/downscaling-rnn-gan](https://github.com/jleinonen/downscaling-rnn-gan).
  The recurrent stochastic downscaling ancestor; sub-daily radar.
* **[2008.09090](https://arxiv.org/abs/2008.09090)** TRU-NET — convolutional-recurrent
  U-Net with cross-attention between recurrent layers, UK rainfall.
* **[2207.00808](https://arxiv.org/abs/2207.00808)** Kumar et al., *Earth Science
  Informatics*, `10.1007/s12145-023-00970-4` — benchmarks an augmented ConvLSTM
  against DeepSD/U-Net/SR-GAN on the same data, and **SR-GAN wins on their point
  metrics**, i.e. recurrence did not help there. Cited as a caution.
* Journal-only ConvLSTM downscaling: Misra et al. (2018) `10.1007/s00704-017-2307-2`;
  Misra et al. (2024) `10.2166/wcc.2024.497`; Miao et al. (2019) `10.3390/w11050977`.

* **[2312.06071](https://arxiv.org/abs/2312.06071)** Srivastava et al.,
  *Precipitation Downscaling with Spatiotemporal Video Diffusion* — deterministic
  downscaler + temporally-conditioned residual diffusion, factorized
  spatio-temporal attention over a 5-frame context. Its **STVD-1 ablation** (single
  context frame, temporal attention removed) "performs significantly worse" — a
  clean quantification of the value of temporal context, and the closest published
  analogue of our `time_only` ablation.
* **[2409.11601](https://arxiv.org/abs/2409.11601)** DiffESM (accepted, *JAMES*) —
  3-D video diffusion conditioned on monthly means generating month-long **daily**
  sequences; U-Net stages are "multiple spatial-only convolutions, followed by a
  single temporal-only convolutional block", evaluated with ACF/PACF and covering
  "hot streaks or dry spells". Architecturally the closest published design to
  ours (spatial convs + a separate temporal operator).
* **[2505.09089](https://arxiv.org/abs/2505.09089)** Hess et al. — trains a
  *time-consistency discriminator* to guide a pretrained **image** diffusion model,
  giving stable centennial rollouts of daily ERA5 precipitation without training a
  video model. The cheapest retrofit path for an existing per-day refiner, and a
  fallback if our Phase-2 integration proves expensive.

---

## 8. Correlated vs i.i.d. noise for the stochastic refinement

**Key negative finding** [VERIFIED]: **no weather or climate diffusion paper
discusses temporally correlated vs i.i.d.-per-frame noise.** Searches across
`"temporally correlated noise"`, `"correlated noise"`, `"noise prior"` combined
with weather/climate/precipitation return zero relevant hits. STVD's full text
contains no occurrence of "i.i.d."/"independent noise" — its temporal coupling is
architectural. DiffCast/CorrDiff decompose the **signal** into mean + residual, not
the **noise** into shared + residual.

The one climate-side data point is EnScale's `CorrDiff` vs `CorrDiff-s`
comparison, effectively an accidental ablation showing the failure in **both**
directions: shared initializations give over-correlated, under-dispersed series
(+0.22 lag-1 ACF error for pr); per-day independent noise restores dispersion but
"removes temporal autocorrelation".

The transferable methodology is from video diffusion [VERIFIED]:

* **PYoCo — [2305.10474](https://arxiv.org/abs/2305.10474)** (Ge et al., NVIDIA).
  "**naively extending the image noise prior to video noise prior in video
  diffusion leads to sub-optimal performance**"; defines mixed noising
  (`ε_shared` + `ε_ind`) and progressive noising, and notes a model trained with
  i.i.d. noise is "**coerced to forget** such correlation".
* **VideoFusion — [2303.08320](https://arxiv.org/abs/2303.08320)**: "frames in the
  same video clip are destroyed with independent noises, **ignoring the content
  redundancy and temporal correlation**"; base + residual noise decomposition.
* **Go-with-the-Flow — [2501.08331](https://arxiv.org/abs/2501.08331)**: flow-warped
  noise, "**agnostic to diffusion model design, requiring no changes to model
  architectures or training pipelines… achieved by just a change in data**". The
  natural analogue of advecting noise with the atmospheric wind field.
* **∫-noise — [2504.03072](https://arxiv.org/abs/2504.03072)**: reinterprets noise
  as a continuously integrated field so it can be advected along a flow while
  preserving its statistics.
* ⚠️ **[2608.02575](https://arxiv.org/abs/2608.02575)**: "structured-noise training
  can reduce prediction loss below the IID reference, but **replacing the test noise
  with IID reverses this advantage**" — train/test noise-structure mismatch is a
  real failure mode.

### Decision [OUR ADAPTATION]

`granitewxc/temporal/refinement.py::AR1NoiseSource` implements the stationary AR(1)
form

```
e_t = rho * e_{t-1} + sqrt(1 - rho^2) * z_t
```

so **every frame's marginal stays exactly N(0,1)** — the refiner still sees the
noise distribution it was trained against, and only the correlation *between*
frames changes. That directly addresses the train/test-mismatch warning.

* `rho = 0` reproduces the legacy i.i.d. path bit-for-bit; it is the shipped default.
* `rho → 1` approaches the shared-noise failure mode, and `rho = 1` is **rejected
  by the config validator** because it manufactures persistence instead of
  modelling it.
* Per-member `torch.Generator` *and* per-member AR(1) state, so a member's
  trajectory is identical whether drawn alone or inside any batch.

Flow-warped noise (Go-with-the-Flow / ∫-noise) is the better long-term option
because it would advect structure with the wind rather than merely persist it, but
it needs an optical-flow/wind field per date and is left as documented future work
rather than half-implemented.

---

## 9. Nowcasting → daily downscaling: why most of §4–5 does not transfer

The brief's caution is well founded. Differences that matter:

| | radar nowcasting (MambaRain, ConvLSTM, TrajGRU) | this task |
|---|---|---|
| cadence | 6 min | **1 day** |
| horizon | 0–3 h | none (same-day downscaling) |
| predictability source | extrapolating observed motion | conditioning on coarse fields at the *same* date |
| resolution | ~4 km, 375×575 | ~10 km, 128×128 (SA) |
| target | reflectivity, extremely intermittent | mm/day, K |
| supervision | future frames from the same field | a *different* (high-resolution) field, same date |

The decisive structural difference: **a nowcaster's only information about time
`t+1` is the state at `t`, whereas we already have the coarse predictors at `t`.**
Our temporal module supplies the *residual* information in the history, not the
primary signal. That bounds the achievable gain, and we measured that bound rather
than assuming it.

[MEASURED] Linear variance decomposition on the SA training period (1961–1970,
deseasonalized, 25 sampled grid points, `pr`/`tasmax` against the 15 coarse
predictors), produced by `cordex_temporal_diagnostics.py`:

| predictors used | `pr` R² | `tasmax` R² |
|---|---|---|
| same day only | 0.164 | 0.837 |
| + history lags 1–3 | 0.186 | 0.859 |
| + history lags 1–5 | 0.193 | 0.862 |
| + history lags 1–9 | 0.208 | 0.866 |

The absolute gain from lags 1–3 is almost the same for both variables (+0.022), but
**the two natural normalizations of that gain differ by an order of magnitude, in
opposite directions**, and conflating them badly misstates the headroom:

| | `pr` | `tasmax` |
|---|---|---|
| increase in **explained** variance, `gain / R²` | **+13.3%** | +2.7% |
| reduction in **unexplained** variance, `gain / (1 − R²)` | +2.6% | **+13.6%** |

⚠️ **Correction.** An earlier revision of this document stated that history "removes
~13% of `pr`'s remaining unexplained variance". That is wrong: 0.022/0.836 = **2.6%**.
The 13% figure is `pr`'s increase in *explained* variance, a different quantity. The
diagnostics script now reports both fields explicitly
(`explained_variance_increase_lags_1_3`, `residual_variance_reduction_lags_1_3`) so
the two cannot be confused again.

What this means for expectations, honestly:

* For **`tasmax`**, history removes a meaningful 13.6% of an already-small residual —
  the most likely place to see a real improvement.
* For **`pr`**, the residual is huge (0.836 of the variance) and history addresses
  only 2.6% of it. **A large precipitation RMSE improvement should not be expected
  from temporal conditioning.** The measured precipitation deficits in the baseline
  are of a different kind — excess wet-day frequency and over-persistent wet spells
  (§7.2 measurements) — which the occurrence/transition objective targets directly
  and which RMSE does not even measure.

⚠️ The apparent continued gain past lag ~5 flattens to a roughly constant increment
per lag, the signature of in-sample overfitting (15 free parameters per added lag on
~3640 samples), not signal. These are in-sample and not cross-validated, so treat
them as an upper bound on the *linear* contribution of history — and as a lower
bound on what a nonlinear model could extract.

Also [MEASURED]: for `pr`, *predictor* history contributes +0.023 R² while
*target* history contributes only +0.005. So the useful memory is in the coarse
dynamics (moisture advection/tendency), not in target persistence — which is
fortunate, because target history is unavailable at inference by construction.

**We therefore do not claim a nowcasting-scale improvement, and the acceptance
thresholds (2% tendency-RMSE reduction, 10% lag-1 ACF error reduction) were set to
be consistent with a ~13% residual-variance headroom rather than with nowcasting
literature.**

---

## 10. Decision summary

| Decision | Choice | Basis |
|---|---|---|
| Integration point | U-Net bottleneck, `[B, 1024, 16, 16]` for SA | [MEASURED] 64× cheaper than the post-backbone map while retaining spatial extent |
| Adapter form | near-identity gated residual, learnable per-channel gate init `1e-3` | [OUR ADAPTATION] preserves pretrained prediction to ~1e-5 relative while all params get gradient from step 1 |
| Default backend | ConvGRU | §5: pure Mamba loses to ConvGRU family on gridded fields; ConvGRU halves carried state vs ConvLSTM |
| Mamba backend | hybrid: conv spatial mixing + **time-axis** SSD scan | §4: no published weather-Mamba actually scans time over a spatial latent |
| Mamba Δ and physical time | learned Δ **×** measured interval ratio | §3.3 [OUR ADAPTATION], ZOH-consistent; papers explicitly give no physical-time hook |
| Kernel policy | `reference` (in-repo pure PyTorch); `fused` raises if unavailable | [MEASURED] `mamba_ssm` uninstallable on this Windows/CUDA host |
| `context_length` | **7** (SA), 5 (NARR/PRISM) | §9 [MEASURED] memory concentrated in lags 1–3; Robin & Vrac (2021) caution against more lags |
| Spectral loss | **not implemented** | §6 [VERIFIED] MambaRain's is complex+unweighted ⇒ MSE by Parseval |
| Temporal losses | tendency, accumulation, occurrence+transition, (optional) lag ACF | §6, each pinned by an adversarial test |
| Refinement noise | AR(1), `rho = 0` default, `rho = 1` rejected | §8, between the two documented failure modes; marginals preserved |
| Primary metric | **absolute** lag-1/2 ACF error, tendency RMSE | §7.2 [MEASURED] our baseline over-persists `tasmax`, so signed ACF error is the wrong target |

---

## 11. Things we could not verify, and open risks

1. **No peer-reviewed venue** for MetMamba, WSSM, MambaRain, CORDEX-ML-Bench,
   Prithvi-WxC, or Mamba-1. Cite all as preprints.
2. **No code** for MetMamba or WSSM; MambaRain has a project page only.
3. **MambaRain's scan axis is unknowable from the paper** — treated as not usable
   evidence rather than assumed.
4. **WSSM's quantitative claims are internally inconsistent** (§4.3); not relied on.
5. **No survey of SSMs for weather/climate**, and **no dedicated ConvLSTM-vs-Mamba
   benchmark**. §5 is assembled from other papers' baseline tables.
6. **No DL paper documents biased spell statistics** as a frame-independence
   failure (§7.3). Our spell/transition diagnostics are therefore adapted from the
   classical VALUE protocol, not from a DL precedent.
7. Discovery ran through arXiv's HTML search (title/abstract only, not full text)
   because `export.arxiv.org/api` rate-limited (HTTP 429). "No paper about X" above
   means "no paper naming X in title or abstract" — weaker than a full-text absence
   proof.
8. **`arXiv:2603.15569` (Mamba-3)** appears in the `state-spaces/mamba` README but
   was **not** verified against arXiv. Not relied on.
9. Venue attributions taken from arXiv `Comments` (PYoCo→ICCV 2023, Vim→ICML 2024,
   VMamba→NeurIPS 2024, ∫-noise→ICLR 2024) are author self-reports, not
   publisher-confirmed.

---

## 12. Reproducing the measurements in this document

```bash
PY=~/AppData/Local/miniforge3/envs/Prithvi/python.exe

# §2.2  no time conditioning in the UNet backbone
$PY -c "import inspect, PrithviWxC.model as M; print(inspect.signature(M.PrithviWxCEncoderDecoder.__init__))"

# §3.4  fused Mamba availability
$PY -c "from granitewxc.temporal.backends import fused_mamba_available; print(fused_mamba_available())"

# §7.2 / §9  event alignment, headroom, and baseline ACF error
$PY examples/CORDEX_ML/cordex_temporal_diagnostics.py --config \
    examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml

# §10  engineering checks on real data
$PY examples/CORDEX_ML/cordex_temporal_training.py check --config \
    examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml
```

See `docs/temporal_model_architecture.md` for the implementation and
`docs/temporal_model_results.md` for measured outcomes and remaining limitations.
