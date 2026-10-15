"""Charts for the README, written to reports/figures/.

    python scripts/make_charts.py

1. alert_frequency.png - alert starts per week since 2023, and per day around the split.
2. model_comparison.png - baseline and models on validation and test, from the saved
   reports/eval_val.csv and reports/eval_test.csv (run scripts/run_eval.py first).
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_alerts import config  # noqa: E402
from air_alerts.data import build_episodes, load_events  # noqa: E402

OUT = config.ROOT / "reports" / "figures"

# Reference palette (dataviz skill, light mode): categorical slots 1-2, text and surface tokens.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def title(ax, text, sub):
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", color=TEXT, pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=TEXT_2, va="bottom")


def alert_frequency() -> None:
    events = load_events()  # up to the evaluation cutoff, 2026-10-15 00:00 Kyiv
    starts = build_episodes(events)[0]["start"].dt.tz_convert(config.TZ)
    starts = starts[starts >= config.DATA_START]

    # Monday-Sunday weeks, Kyiv time. Weeks without any start count as 0 (not skipped), and only
    # weeks fully inside the data are shown: the data starts on Sunday 1 Jan 2023 and ends on
    # Wednesday 14 Oct 2026, so both edge weeks would be partial.
    local = starts.dt.tz_localize(None)
    first_day = config.DATA_START.tz_localize(None)
    last_day = (config.CUTOFF - pd.Timedelta(days=1)).tz_localize(None).normalize()
    counts = local.dt.to_period("W-SUN").value_counts()
    weeks = pd.period_range(first_day, last_day, freq="W-SUN")
    weeks = weeks[(weeks.start_time >= first_day) & (weeks.end_time.normalize() <= last_day)]
    weekly = counts.reindex(weeks, fill_value=0)
    weekly.index = weekly.index.start_time

    zoom_from = pd.Timestamp("2026-06-01")
    daily = starts.dt.tz_localize(None).dt.floor("D").value_counts()
    daily = daily.reindex(pd.date_range(zoom_from, "2026-10-14", freq="D"), fill_value=0)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.4), gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.55})

    top.plot(weekly.index, weekly.values, color=BLUE, lw=2, solid_joinstyle="round", solid_capstyle="round")
    top.fill_between(weekly.index, weekly.values, color=BLUE, alpha=0.10, lw=0)
    top.set_ylim(0, None)
    top.set_ylabel("alert starts per week")
    title(top, "Alert starts in Kyiv City, per week",
          f"Complete weeks {weekly.index[0]:%d %b %Y} - {weekly.index[-1] + pd.Timedelta(days=6):%d %b %Y}, Kyiv Digital history. "
          "Mostly 3-15 a week until August 2026, then a sharp rise.")
    peak = weekly.idxmax()
    top.annotate(f"{weekly.max()} starts", (peak, weekly.max()), xytext=(-70, -4), textcoords="offset points",
                 color=TEXT_2, fontsize=9, va="center", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})

    periods = [("train (part)", zoom_from, config.VAL_START), ("validation", config.VAL_START, config.TEST_START),
               ("test", config.TEST_START, config.CUTOFF)]
    ymax = daily.max() * 1.32
    for i, (name, a, b) in enumerate(periods):
        a, b = pd.Timestamp(a).tz_localize(None), pd.Timestamp(b).tz_localize(None)
        if i:
            bottom.axvspan(a, b, color=GRID, alpha=0.5 if i == 1 else 0.9, lw=0)
        bottom.text(a + (b - a) / 2, ymax * 0.97, name, ha="center", va="top", fontsize=9, color=TEXT_2)
    bottom.bar(daily.index, daily.values, width=0.8, color=BLUE)
    bottom.set_ylim(0, ymax)
    bottom.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    bottom.set_ylabel("alert starts per day")
    change = pd.Timestamp("2026-09-06")
    bottom.axvline(change, color=MUTED, lw=1, ls=(0, (3, 3)))
    bottom.text(change, ymax * 0.80, " yellow/red levels\n introduced 6 Sep", fontsize=8.5, color=TEXT_2, va="top")
    title(bottom, "Alert starts per day, June - mid-October 2026",
          "Shaded: the periods used to validate and to test the models. The rise begins 24 Aug, before the new alert format.")
    bottom.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))

    fig.savefig(OUT / "alert_frequency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


MODELS = [  # (name in reports, label), in the order of the story
    ("recent_rate_24h_floor", "24h baseline\n(pre-registered primary)"),
    ("logistic", "Logistic regression,\ntrained once"),
    ("lightgbm", "LightGBM,\ntrained once"),
    ("lightgbm wf", "LightGBM,\nretrained daily"),
    ("logistic+decay14 wf", "Logistic, retrained daily,\n14-day decay (live model)"),
]


def model_comparison() -> None:
    reports = {name: pd.read_csv(config.ROOT / "reports" / f"eval_{name}.csv") for name in ("val", "test")}
    scores = {
        period: df[df["slice"] == "all"].set_index("model").loc[[m for m, _ in MODELS]]
        for period, df in reports.items()
    }
    labels = [label for _, label in MODELS]
    y = range(len(MODELS))
    h = 0.36
    periods = [("val", "Validation (24 Aug - 16 Sep)", BLUE, -h / 2), ("test", "Test (17 Sep - 14 Oct)", ORANGE, h / 2)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True, gridspec_kw={"wspace": 0.08})
    panels = [("bss_vs_period_rate", "Brier skill score", 0.0,
               "vs a constant at the period's own alert rate;\nabove 0 = more informative than that constant"),
              ("roc_auc", "ROC-AUC", 0.5, "ranking of hours;\n0.5 = no better than random")]
    for ax, (col, name, ref, note) in zip(axes, panels):
        for period, label, color, offset in periods:
            vals = scores[period][col].to_numpy()
            ax.barh([i + offset for i in y], vals - ref, left=ref, height=h, color=color, label=label)
            for i, v in enumerate(vals):
                ax.text(v + (0.012 if v >= ref else -0.012), i + offset, f"{v:+.2f}" if ref == 0 else f"{v:.2f}",
                        va="center", ha="left" if v >= ref else "right", fontsize=8, color=TEXT_2)
        ax.axvline(ref, color=TEXT_2, lw=1)
        ax.grid(axis="y", visible=False)
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold", color=TEXT)
        ax.text(0, -0.075, note, transform=ax.transAxes, fontsize=8.5, color=TEXT_2, va="top")
    axes[0].set_xlim(-1.05, 0.35)
    axes[1].set_xlim(0.3, 0.85)
    axes[0].set_yticks(list(y), labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle("Baseline and models: validation vs test (higher is better)", x=0.125, ha="left",
                 fontsize=12, fontweight="bold", color=TEXT, y=1.0)
    fig.text(0.125, 0.94, "The baseline won validation, then lost on test; retraining daily with recent data "
             "weighted more won on test.", fontsize=9, color=TEXT_2)
    fig.text(0.125, -0.04, '"Trained once" = fit on all data before the period: for test that includes the validation '
             "weeks, which is why LightGBM improves so much there.\n\"Retrained daily\" = refit every midnight on "
             "outcomes known by then (walk-forward). Each test score was computed once, with settings frozen beforehand.",
             fontsize=8.5, color=TEXT_2, va="top")

    fig.savefig(OUT / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    alert_frequency()
    model_comparison()
    print(f"saved {OUT.relative_to(config.ROOT)}/alert_frequency.png, model_comparison.png")


if __name__ == "__main__":
    main()
