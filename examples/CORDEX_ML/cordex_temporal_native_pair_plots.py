"""Figures for the Prithvi-native paired-state bounded comparison.

Reads only what the experiment already wrote -- ``predictions.npz`` and
``evaluation.json`` per variant, plus ``manifest.json``/``scorecard.json`` -- so
the figures cannot disagree with the scorecard, and regenerating them never
re-runs a model.

    python examples/CORDEX_ML/cordex_temporal_native_pair_plots.py \
        --experiment examples/CORDEX_ML/runs_temporal/experiment_native_pair \
        --out docs/figures/native_pair

Design constraints applied deliberately:

* **Categorical** colors identify a *variant* and are assigned in a fixed order,
  never cycled, and never re-assigned when the variant list changes. The six-slot
  order is validated (worst adjacent CVD dE 9.1, normal-vision 19.6). Three slots
  sit below 3:1 contrast on a light surface, so every categorical chart carries
  **direct value labels** -- the documented relief for that.
* **Diverging** (two hues, neutral midpoint) for signed error fields only, always
  on symmetric limits so zero is the neutral middle. Never a rainbow.
* **Sequential single-hue** for magnitude fields.
* **One y-axis per panel.** Quantities on different scales get separate panels,
  never a twin axis.
* Lines are 2 px with >= 8 px markers; grid and spines are recessive; all text is
  neutral ink rather than the series color.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- design tokens -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e5e4e0"

#: Fixed categorical order. Never cycled; a variant keeps its color regardless of
#: which other variants are present.
SERIES = {
    "baseline": "#2a78d6",              # blue
    "spatial_ft": "#eb6834",            # orange
    "time_only": "#1baf7a",             # aqua
    "native_pair": "#eda100",           # yellow
    "native_pair_nohistory": "#e87ba4",  # magenta
    "native_pair_pretext": "#008300",   # green
    "convgru": "#4a3aa7",               # violet  (archived reference)
    "mamba": "#e34948",                 # red     (archived reference)
}
TRUTH = INK
SEQUENTIAL = {"pr": "Blues", "tasmax": "Oranges"}
DIVERGING = "RdBu_r"

ORDER = [
    "baseline",
    "spatial_ft",
    "time_only",
    "native_pair",
    "native_pair_nohistory",
    "native_pair_pretext",
]

UNITS = {"pr": "mm/day", "tasmax": "K"}


def style_axes(ax, *, ygrid: bool = True, xgrid: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, text: str, subtitle: str | None = None) -> None:
    ax.set_title(text, color=INK, fontsize=11, loc="left", pad=14 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes,
            color=INK_MUTED, fontsize=8.5, ha="left", va="bottom",
        )


def new_figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def guard_empty(ax, label: str = "no data") -> bool:
    """Annotate and fix limits on an axes with nothing plotted.

    An axes with no artists autoscales to a degenerate range, and a subsequent
    ``bbox_inches='tight'`` render can blow the canvas size up to nonsense. Saying
    "no data" is also the honest output when a metric is genuinely absent.
    """
    if ax.lines or ax.patches or ax.collections or ax.images:
        return False
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center",
            color=INK_MUTED, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return True


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] {path}")
    return path


# --- loading -----------------------------------------------------------------
def load(root: Path):
    """Return ``(evaluations, predictions, variables, manifest, scorecard)``."""
    evaluations: dict[str, dict] = {}
    predictions: dict[str, dict] = {}
    for name in ORDER:
        ev = root / name / "evaluation.json"
        if ev.is_file():
            evaluations[name] = json.loads(ev.read_text(encoding="utf-8"))
        npz = root / name / "predictions.npz"
        if npz.is_file():
            predictions[name] = npz
    manifest = {}
    if (root / "manifest.json").is_file():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scorecard = {}
    if (root / "scorecard.json").is_file():
        scorecard = json.loads((root / "scorecard.json").read_text(encoding="utf-8"))
    variables: list[str] = []
    for ev in evaluations.values():
        variables = list(ev.get("variables", {}))
        break
    return evaluations, predictions, variables, manifest, scorecard


def metric(evaluations, variant, var, path):
    """Look up a metric, returning ``None`` for absent OR non-finite values.

    Non-finite is treated as absent on purpose. A NaN that reaches a text label
    coordinate makes matplotlib's tight bounding box degenerate into an
    astronomically large canvas and the render dies with an opaque constructor
    error a long way from the cause -- so it is filtered at the source instead.
    """
    node = evaluations.get(variant, {}).get("variables", {}).get(var, {})
    # maxsplit=1 is load-bearing: metric NAMES contain dots ("q0.99_bias",
    # "q0.1"), and the record is at most two levels deep ("paired.rmse",
    # "seams.seam_jump_ratio", or a bare "rmse_boundary"). Splitting on every dot
    # silently resolved every quantile metric to None, which then produced an
    # empty axes and a degenerate tight bounding box rather than an error.
    for part in path.split(".", 1):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    return float(node) if np.isfinite(node) else None


def bar_panel(
    ax,
    names,
    values,
    *,
    horizontal: bool = False,
    reference=None,
    zero_line: float | None = None,
    fmt: str = "{:.4g}",
) -> None:
    """One categorical bar panel with direct value labels.

    Direct labels are not decoration here: three of the six categorical slots sit
    below 3:1 contrast on a light surface, and a visible label is the documented
    relief for that. They also mean a reader never has to interpolate a value off
    the axis. Absent or non-finite metrics are drawn as a zero-height bar labelled
    ``n/a`` rather than silently as 0.
    """
    idx = np.arange(len(names))
    heights = [0.0 if v is None else float(v) for v in values]
    finite = [h for h, v in zip(heights, values) if v is not None]
    span = max((abs(h) for h in finite), default=1.0) or 1.0
    colors = [SERIES[n] for n in names]

    if horizontal:
        ax.barh(idx, heights, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2.0)
    else:
        ax.bar(idx, heights, color=colors, width=0.66, edgecolor=SURFACE, linewidth=2.0)

    if zero_line is not None:
        (ax.axvline if horizontal else ax.axhline)(
            zero_line, color=INK_MUTED, linewidth=1.2, zorder=0
        )
    if reference is not None and np.isfinite(reference):
        (ax.axvline if horizontal else ax.axhline)(
            float(reference), color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)), zorder=0
        )

    for i, (h, raw) in enumerate(zip(heights, values)):
        text = "n/a" if raw is None else fmt.format(h)
        pad = 0.02 * span * (1 if h >= 0 else -1)
        if horizontal:
            ax.text(h + pad, i, text, va="center", ha="left" if h >= 0 else "right",
                    color=INK, fontsize=8.5)
        else:
            ax.text(i, h + pad, text, ha="center", va="bottom" if h >= 0 else "top",
                    color=INK, fontsize=8.5)

    if horizontal:
        ax.set_yticks(idx, names, fontsize=8.5)
        ax.margins(x=0.22)
    else:
        ax.set_xticks(idx, names, rotation=32, ha="right", fontsize=8.5)
        ax.margins(y=0.24)


def present(evaluations) -> list[str]:
    return [v for v in ORDER if v in evaluations]


# --- 1. categorical metric comparison ---------------------------------------
def plot_metric_bars(evaluations, variables, out: Path) -> None:
    """Horizontal bars per variant for the pre-registered primary metrics.

    Bars, not lines: the job is magnitude-by-identity, and a category axis has no
    meaningful ordering to interpolate along. Direct value labels are the required
    relief for the low-contrast slots, and they also remove any need to read a
    value off the axis.
    """
    panels = [
        ("paired.autocorr_lag1_error", "lag-1 autocorrelation error", "|.| toward 0 is better"),
        ("paired.autocorr_lag2_error", "lag-2 autocorrelation error", "|.| toward 0 is better"),
        ("paired.tendency_rmse", "day-to-day tendency RMSE", "lower is better"),
        ("paired.acc3d_rmse", "3-day accumulation RMSE", "lower is better"),
        ("paired.rmse", "per-frame RMSE (guardrail)", "lower is better"),
        ("paired.tendency_std_ratio", "tendency std ratio (guardrail)", "toward 1.0 is better"),
    ]
    names = present(evaluations)
    if not names or not variables:
        return
    for var in variables:
        fig, axes = new_figure(
            len(panels), 1, figsize=(8.4, 2.05 * len(panels)), sharey=True
        )
        axes = np.atleast_1d(axes)
        for ax, (path, label, hint) in zip(axes, panels):
            values = [metric(evaluations, n, var, path) for n in names]
            zero = 1.0 if "ratio" in path else (0.0 if "error" in path else None)
            bar_panel(
                ax, names, values, horizontal=True, zero_line=zero,
                reference=values[0] if names and names[0] == "baseline" else None,
            )
            style_axes(ax, ygrid=False, xgrid=True)
            title(ax, f"{label}", hint)
        fig.suptitle(
            f"{var} - pre-registered metrics by variant   (dashed = baseline)",
            color=INK, fontsize=12, x=0.01, ha="left", y=1.002,
        )
        fig.tight_layout()
        save(fig, out, f"metrics_{var}.png")


# --- 2. temporal evolution ---------------------------------------------------
def plot_temporal_evolution(predictions, variables, out: Path, *, days: int = 120) -> None:
    """Domain-mean trajectory: observed versus a deliberately small set of variants.

    Capped at three model series plus the truth. Six overlapping lines would be
    unreadable no matter how well separated the hues are, and the comparison that
    matters is candidate versus its own no-history control versus the baseline.
    """
    want = [v for v in ("baseline", "native_pair", "native_pair_nohistory") if v in predictions]
    if not want or not variables:
        return
    ref = np.load(predictions[want[0]], allow_pickle=False)
    names = [str(v) for v in ref["output_vars"]]
    fig, axes = new_figure(len(names), 1, figsize=(10.5, 2.9 * len(names)))
    axes = np.atleast_1d(axes)
    for i, var in enumerate(names):
        ax = axes[i]
        ax.plot(np.nanmean(ref["target"][:days, i], axis=(1, 2)),
                color=TRUTH, linewidth=2.0, label="observed", zorder=5)
        for variant in want:
            d = np.load(predictions[variant], allow_pickle=False)
            ax.plot(np.nanmean(d["pred"][:days, i], axis=(1, 2)),
                    color=SERIES[variant], linewidth=2.0, alpha=0.95, label=variant)
        style_axes(ax)
        ax.set_xlabel("day of the held-out test period", color=INK_2, fontsize=9)
        ax.set_ylabel(f"{var} ({UNITS.get(var, '')})", color=INK_2, fontsize=9)
        title(ax, f"{var}: domain-mean trajectory",
              f"first {days} days of 1981-1983; observed in black")
        leg = ax.legend(frameon=False, fontsize=8.5, ncols=4, loc="upper left")
        for text in leg.get_texts():
            text.set_color(INK_2)
    fig.tight_layout()
    save(fig, out, "temporal_evolution.png")


def plot_autocorrelation(evaluations, variables, out: Path) -> None:
    """Predicted vs observed lag-k autocorrelation -- the primary temporal claim.

    Plotted as predicted-minus-observed against lag so the target is the zero
    line: a variant that *overshoots* persistence lands below zero and is visibly
    not better than one that merely under-shoots, which a plot of |error| would
    have hidden.
    """
    names = present(evaluations)
    lags = [1, 2, 3, 5]
    if not names or not variables:
        return
    fig, axes = new_figure(1, len(variables), figsize=(5.4 * len(variables), 4.0))
    axes = np.atleast_1d(axes)
    for ax, var in zip(axes, variables):
        ax.axhline(0.0, color=INK_MUTED, linewidth=1.2, zorder=1)
        for variant in names:
            ys = [metric(evaluations, variant, var, f"paired.autocorr_lag{k}_error") for k in lags]
            if any(y is None for y in ys):
                continue
            ax.plot(lags, ys, marker="o", markersize=8, linewidth=2.0,
                    color=SERIES[variant], label=variant, zorder=3)
        style_axes(ax)
        guard_empty(ax, "autocorrelation record unavailable")
        ax.set_xticks(lags)
        ax.set_xlabel("lag (days)", color=INK_2, fontsize=9)
        ax.set_ylabel("predicted - observed autocorrelation", color=INK_2, fontsize=9)
        title(ax, f"{var}: persistence error by lag",
              "zero is perfect; below zero is OVER-estimated persistence")
    leg = axes[0].legend(frameon=False, fontsize=8.5, loc="best")
    for text in leg.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    save(fig, out, "autocorrelation_error.png")


# --- 3. accumulations --------------------------------------------------------
def plot_accumulations(evaluations, variables, out: Path) -> None:
    """3-day and 5-day accumulation error, as separate panels.

    Separate panels rather than one axis with two groups: the two windows have
    different magnitudes, and a shared axis would compress the 3-day differences.
    Never a twin axis.
    """
    names = present(evaluations)
    if not names or not variables:
        return
    for var in variables:
        fig, axes = new_figure(1, 2, figsize=(11.0, 3.6))
        for ax, window in zip(np.atleast_1d(axes), (3, 5)):
            vals = [metric(evaluations, n, var, f"paired.acc{window}d_rmse") for n in names]
            bar_panel(ax, names, vals, fmt="{:.3g}",
                      reference=vals[0] if names and names[0] == "baseline" else None)
            style_axes(ax)
            ax.set_ylabel(f"RMSE ({UNITS.get(var, '')})", color=INK_2, fontsize=9)
            title(ax, f"{window}-day accumulation RMSE", "lower is better; dashed = baseline")
        fig.suptitle(f"{var} - multi-day accumulation", color=INK, fontsize=12,
                     x=0.01, ha="left", y=1.02)
        fig.tight_layout()
        save(fig, out, f"accumulation_{var}.png")


# --- 4. spatial error --------------------------------------------------------
def plot_spatial_error(predictions, out: Path, *, day: int = 30) -> None:
    """Field, prediction and signed error for the candidate against the baseline.

    Magnitude panels use a single-hue sequential ramp; the error panel uses a
    diverging ramp on symmetric limits so the neutral midpoint is exactly zero.
    """
    want = [v for v in ("baseline", "native_pair") if v in predictions]
    if not want:
        return
    ref = np.load(predictions[want[0]], allow_pickle=False)
    names = [str(v) for v in ref["output_vars"]]
    ncols = 1 + 2 * len(want)
    fig, axes = new_figure(len(names), ncols, figsize=(3.5 * ncols, 3.3 * len(names)))
    axes = np.atleast_2d(axes)
    for i, var in enumerate(names):
        target = ref["target"][day, i]
        vmin, vmax = np.nanpercentile(target, [2, 98])
        im = axes[i, 0].imshow(target, origin="lower", vmin=vmin, vmax=vmax,
                               cmap=SEQUENTIAL.get(var, "Greys"))
        axes[i, 0].set_title(f"{var} observed", color=INK, fontsize=10, loc="left")
        fig.colorbar(im, ax=axes[i, 0], shrink=0.78)
        errors = []
        for variant in want:
            errors.append(np.load(predictions[variant], allow_pickle=False)["pred"][day, i] - target)
        lim = float(np.nanpercentile(np.abs(np.stack(errors)), 98)) or 1.0
        for j, variant in enumerate(want):
            pred = np.load(predictions[variant], allow_pickle=False)["pred"][day, i]
            im = axes[i, 1 + 2 * j].imshow(pred, origin="lower", vmin=vmin, vmax=vmax,
                                           cmap=SEQUENTIAL.get(var, "Greys"))
            axes[i, 1 + 2 * j].set_title(variant, color=INK, fontsize=10, loc="left")
            fig.colorbar(im, ax=axes[i, 1 + 2 * j], shrink=0.78)
            im = axes[i, 2 + 2 * j].imshow(errors[j], origin="lower", cmap=DIVERGING,
                                           vmin=-lim, vmax=lim)
            axes[i, 2 + 2 * j].set_title(f"{variant} - observed", color=INK, fontsize=10,
                                         loc="left")
            fig.colorbar(im, ax=axes[i, 2 + 2 * j], shrink=0.78)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"Spatial fields and signed error, test day {day} "
        "(error panels share one symmetric scale per row)",
        color=INK, fontsize=12, x=0.01, ha="left", y=1.005,
    )
    fig.tight_layout()
    save(fig, out, "spatial_error.png")


# --- 5. extremes -------------------------------------------------------------
def plot_extremes(evaluations, variables, out: Path) -> None:
    """Observed vs predicted upper quantiles, and the q99 bias by variant."""
    names = present(evaluations)
    if not names or not variables:
        return
    quantiles = ["q0.1", "q0.5", "q0.9", "q0.99"]
    for var in variables:
        fig, axes = new_figure(1, 2, figsize=(11.4, 3.8))
        ax = np.atleast_1d(axes)[0]
        truth = [
            metric(evaluations, names[0], var, f"distribution_truth.{q}") for q in quantiles
        ]
        xs = np.arange(len(quantiles))
        if all(t is not None for t in truth):
            ax.plot(xs, truth, color=TRUTH, linewidth=2.0, marker="s", markersize=8,
                    label="observed", zorder=5)
        for variant in names:
            ys = [metric(evaluations, variant, var, f"distribution_pred.{q}") for q in quantiles]
            if any(y is None for y in ys):
                continue
            ax.plot(xs, ys, color=SERIES[variant], linewidth=2.0, marker="o", markersize=8,
                    label=variant)
        ax.set_xticks(xs, quantiles, fontsize=9)
        style_axes(ax)
        guard_empty(ax, "quantile record unavailable")
        ax.set_ylabel(f"{var} ({UNITS.get(var, '')})", color=INK_2, fontsize=9)
        title(ax, f"{var}: upper-tail quantiles", "observed in black")
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
        for text in leg.get_texts():
            text.set_color(INK_2)

        ax = np.atleast_1d(axes)[1]
        vals = [metric(evaluations, n, var, "paired.q0.99_bias") for n in names]
        bar_panel(ax, names, vals, zero_line=0.0, fmt="{:.3g}")
        style_axes(ax)
        ax.set_ylabel(f"q99 bias ({UNITS.get(var, '')})", color=INK_2, fontsize=9)
        title(ax, "99th-percentile bias", "zero is perfect; negative = extremes too weak")
        fig.tight_layout()
        save(fig, out, f"extremes_{var}.png")


# --- 6. boundary vs interior + seams ----------------------------------------
def plot_boundary(evaluations, variables, out: Path) -> None:
    """Boundary versus interior RMSE, and the chunk-seam null.

    Reported separately, never averaged, so a change that improves the interior
    while degrading the rim stays visible. The seam panel is shown against the
    baseline's own measured ratio, not against 1.0: the baseline has no temporal
    pathway at all, so its ratio *is* the null level of that statistic at this
    sample size.
    """
    names = present(evaluations)
    if not names or not variables:
        return
    fig, axes = new_figure(2, len(variables), figsize=(5.6 * len(variables), 7.2))
    axes = np.atleast_2d(axes.reshape(2, -1))
    for j, var in enumerate(variables):
        ax = axes[0, j]
        width = 0.38
        xs = np.arange(len(names))
        for k, (key, hatch) in enumerate((("rmse_boundary", None), ("rmse_interior", "///"))):
            vals = [metric(evaluations, n, var, key) for n in names]
            vals = [0.0 if v is None else float(v) for v in vals]
            ax.bar(xs + (k - 0.5) * width, vals, width=width,
                   color=[SERIES[n] for n in names], edgecolor=SURFACE, linewidth=2.0,
                   hatch=hatch, label="boundary" if k == 0 else "interior (hatched)")
            for x, v in zip(xs, vals):
                ax.text(x + (k - 0.5) * width, v, f"{v:.3g}", ha="center", va="bottom",
                        color=INK, fontsize=7.5, rotation=90)
        ax.set_xticks(xs, names, rotation=32, ha="right", fontsize=8.5)
        ax.margins(y=0.26)
        style_axes(ax)
        ax.set_ylabel(f"RMSE ({UNITS.get(var, '')})", color=INK_2, fontsize=9)
        title(ax, f"{var}: boundary vs interior RMSE",
              "8-px rim vs interior; texture separates the two, not color")
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
        for text in leg.get_texts():
            text.set_color(INK_2)

        ax = axes[1, j]
        vals = [metric(evaluations, n, var, "seams.seam_jump_ratio") for n in names]
        bar_panel(ax, names, vals, fmt="{:.3g}",
                  reference=vals[0] if names and names[0] == "baseline" else None)
        style_axes(ax)
        ax.set_ylabel("seam jump ratio", color=INK_2, fontsize=9)
        title(ax, f"{var}: chunk-seam jump ratio",
              "compare against the dashed baseline (the measured null), not 1.0")
    fig.tight_layout()
    save(fig, out, "boundary_and_seams.png")


# --- 7. wet/dry structure ----------------------------------------------------
def plot_wet_dry(evaluations, out: Path, var: str = "pr") -> None:
    """Occurrence and spell-structure errors -- diagnostics, not a promotion basis."""
    names = present(evaluations)
    if not names:
        return
    panels = [
        ("distribution_delta.wet_day_frequency", "wet-day frequency error", ""),
        ("distribution_delta.p_wet_given_wet", "P(wet|wet) error", ""),
        ("distribution_delta.wet_spell_mean", "mean wet-spell length error (days)", ""),
        ("distribution_delta.dry_spell_mean", "mean dry-spell length error (days)", ""),
    ]
    if metric(evaluations, names[0], var, panels[0][0]) is None:
        return
    fig, axes = new_figure(2, 2, figsize=(11.4, 6.6))
    for ax, (path, label, _) in zip(axes.ravel(), panels):
        vals = [metric(evaluations, n, var, path) for n in names]
        bar_panel(ax, names, vals, zero_line=0.0, fmt="{:.3g}")
        style_axes(ax)
        title(ax, label, "zero is perfect")
    fig.suptitle(
        f"{var} - occurrence and spell structure  "
        "(NOT part of the pre-registered primary set)",
        color=INK, fontsize=12, x=0.01, ha="left", y=1.01,
    )
    fig.tight_layout()
    save(fig, out, f"wet_dry_{var}.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="examples/CORDEX_ML/runs_temporal/experiment_native_pair")
    ap.add_argument("--out", default="docs/figures/native_pair")
    ap.add_argument("--day", type=int, default=30, help="test-period day for the spatial maps")
    ap.add_argument("--days", type=int, default=120, help="days in the trajectory panel")
    args = ap.parse_args()

    root = Path(args.experiment)
    out = Path(args.out)
    evaluations, predictions, variables, manifest, scorecard = load(root)
    if not evaluations:
        print(f"No evaluation.json found under {root}; run the experiment first.")
        return 1
    print(f"[plot] variants: {present(evaluations)}")
    print(f"[plot] variables: {variables}")

    plot_metric_bars(evaluations, variables, out)
    plot_autocorrelation(evaluations, variables, out)
    plot_accumulations(evaluations, variables, out)
    plot_extremes(evaluations, variables, out)
    plot_boundary(evaluations, variables, out)
    plot_wet_dry(evaluations, out)
    plot_temporal_evolution(predictions, variables, out, days=args.days)
    plot_spatial_error(predictions, out, day=args.day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
