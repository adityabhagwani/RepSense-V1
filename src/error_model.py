"""Count-error model: given a real set, how likely is its rep count wrong? Wrist only.

Like a forecaster reading the sky: it doesn't count, it reads a set's symptoms (how even the rhythm
was, how close the humps sat to the counting cut, how much the counters disagreed...) and says how
likely the count is wrong. It learns on spliced sets, where the truth is known by construction and
errors are plentiful, and is checked on real sets. Whether a set is a set at all is a separate model.

Protocol fixed in FEATURES.md (change log 2026-11-04) before any symptom was computed: target = the
candidate stack (min-reps-v1 with detector-v4 inside) gets the count wrong; main model = the core
features (#1, #4, #7, #10, #12, #25), L2 logistic, scored leave-one-source-session-out on spliced
sets; real check = each real session's curl sets, predicted by a model that never saw spliced sets
built from that session.

    python -m src.error_model
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.counter import ANGLE_FLOOR_DEG, HUMP_SHARE, MIN_SPACING_S, count_set, forearm_angle
from src.dataio import load_annotations, load_session, slice_set, with_true_reps
from src.detector import FEATURES, MATCH_S, combined_times, hump_features, min_reps
from src.labels import USB_AXIS_FILE
from src.splice import LIBRARY, build_lever_set, degrade, load_library, plan_levers, splice
from src.stream import UNRESOLVED, flagged_sets, model_for
from src.sync import REPORTS_DIR

SYMPTOMS = {  # FEATURES.md number: name. Every one is computed from the wrist alone.
    1: "rhythm_irregularity", 2: "longest_gap_ratio", 3: "shortest_gap_ratio", 4: "weakest_peak_ratio",
    5: "peak_height_spread", 6: "close_calls", 7: "rhythm_strength", 8: "period_vs_count",
    10: "threshold_stability", 11: "motion_outside_reps", 12: "seconds_per_rep", 13: "signal_quality",
    25: "rule_minus_combined", 26: "learned_close_calls", 27: "counter_disagreement",
}
CORE = [SYMPTOMS[k] for k in (1, 4, 7, 10, 12, 25)]  # the confirmatory set (FEATURES.md ★)
CUT_WIGGLE = 0.15  # #10: how far the rule's cut moves either way
REAL_SESSIONS = {"p1": ["dev1", "dev1b", "dev2b"], "p2": ["p2_s1"], "p3": ["p3_s1"], "p4": ["p4_s1"], "p5": ["p5_s1"]}


def ratio(a, b):
    return a / b if b else np.nan


def symptoms(seg: pd.DataFrame, model, usb_axis, version: str = "v4") -> dict:
    """The count-model features of one set, plus what each counter said. Rep gaps and heights are the
    shipped counter's (min-reps with the learned veto); #10 and #6 are about the rule's cut."""
    t = seg["t"].to_numpy()
    fs = 1 / np.median(np.diff(t))
    rule = count_set(seg)["angle"]
    humps = hump_features(seg, usb_axis, version)
    p = model.predict_proba(humps[FEATURES])[:, 1] if len(humps) else np.array([])
    tops = humps["t_top"].to_numpy()
    kept = combined_times(rule.times, tops, p)
    reps = np.sort(min_reps(kept))
    n = len(reps)

    gaps = np.diff(reps)
    gap = np.median(gaps) if len(gaps) else np.nan
    heights = np.array([h for x, h in zip(rule.times, rule.heights) if np.any(np.isclose(reps, x))])
    angle = forearm_angle(seg)
    lag = int(round(gap * fs)) if len(gaps) else 0
    strength = np.corrcoef(angle[:-lag], angle[lag:])[0, 1] if 0 < lag < len(angle) - 2 else np.nan

    _, props = find_peaks(angle, prominence=ANGLE_FLOOR_DEG, distance=max(1, int(MIN_SPACING_S * fs)))
    prom = props["prominences"]
    moved = [int((prom >= HUMP_SHARE * (1 + d) * np.median(prom)).sum()) - rule.n for d in (-CUT_WIGGLE, CUT_WIGGLE)] \
        if len(prom) else [0]

    speed = np.linalg.norm(seg[["gx", "gy", "gz"]].to_numpy(), axis=1)
    inside = np.zeros(len(t), bool)
    for x in reps:  # each shipped rep's lift span, from the hump it matched
        k = int(np.argmin(np.abs(tops - x)))
        if abs(tops[k] - x) <= MATCH_S:
            a, b = tops[k] - humps["rise_s"].iloc[k], tops[k] + humps["fall_s"].iloc[k]
            inside |= (t - t[0] >= a) & (t - t[0] <= b)
    lost = max(float(np.sum(np.round(np.diff(t) * fs) - 1)), 0.0) / len(t)
    gyro = np.abs(seg[["gx", "gy", "gz"]].to_numpy())
    ceiling = sum(max(int((gyro[:, i] >= gyro[:, i].max()).sum()) - 1, 0) for i in range(3)) / len(t)
    learned = int((p >= 0.5).sum())

    return {
        "rhythm_irregularity": ratio(gaps.std(), gaps.mean()) if len(gaps) >= 2 else np.nan,
        "longest_gap_ratio": ratio(gaps.max(), gap) if len(gaps) >= 2 else np.nan,
        "shortest_gap_ratio": ratio(gaps.min(), gap) if len(gaps) >= 2 else np.nan,
        "weakest_peak_ratio": ratio(heights.min(), np.median(heights)) if len(heights) >= 2 else np.nan,
        "peak_height_spread": ratio(heights.std(), heights.mean()) if len(heights) >= 2 else np.nan,
        "close_calls": ratio(rule.near_cut, rule.candidates),
        "rhythm_strength": strength,
        "period_vs_count": ratio((t[-1] - t[0]) / gap, n) if len(gaps) else np.nan,
        "threshold_stability": max(abs(m) for m in moved),
        "motion_outside_reps": ratio(speed[~inside].sum(), speed.sum()),
        "seconds_per_rep": ratio(t[-1] - t[0], n),
        "signal_quality": lost + ceiling,
        "rule_minus_combined": rule.n - len(kept),
        "learned_close_calls": float(np.mean((p >= 0.2) & (p <= 0.8))) if len(p) else np.nan,
        "counter_disagreement": (abs(n - rule.n) + abs(n - learned)) / 2,
        "rule": rule.n, "learned": learned, "combined": len(kept), "shipped": n,
    }


def spliced_rows(seed: int, n_sets: int, usb_axis, version: str) -> pd.DataFrame:
    """The screen's spliced sets (rebuilt from default_rng([seed, i])) with their symptoms and truth."""
    libs = {s: load_library(s, seed, version) for s in LIBRARY}
    rows = []
    for i in range(n_sets):
        rng = np.random.default_rng([seed, i])
        lib = libs[LIBRARY[rng.integers(len(LIBRARY))]]
        r = build_lever_set(rng, lib, plan_levers(rng))
        seg = degrade(splice(lib.raw, r.parts), r.levers["dropped_samples"], r.levers["clipped"], r.levers["degrade_seed"])
        rows.append({"set": i, "source": lib.session, "truth": r.truth, **symptoms(seg, lib.model, usb_axis, version)})
    return pd.DataFrame(rows)


def real_rows(usb_axis, version: str) -> pd.DataFrame:
    """Every real curl set (Enter presses) with a settled count, its symptoms and truth. Dev sessions get
    a learned model trained on the other dev sessions, as in the dev scores."""
    rows = []
    for person, sessions in REAL_SESSIONS.items():
        for session in sessions:
            raw, model = load_session(session), model_for(session, version)
            ann = with_true_reps(load_annotations(session), session)
            for _, a in ann[(ann["status"] == "ok") & (ann["exercise"] == "curl")].iterrows():
                if (session, int(a["set_idx"])) in UNRESOLVED:
                    continue
                rows.append({"person": person, "session": session, "set_idx": int(a["set_idx"]),
                             "condition": a["condition"], "flagged": int(a["set_idx"]) in flagged_sets(session),
                             "truth": int(a["true_reps"]), **symptoms(slice_set(raw, a), model, usb_axis, version)})
    return pd.DataFrame(rows)


def make_model(c: float = 1.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def out_of_fold(train: pd.DataFrame, features: list[str], groups: pd.Series, test: pd.DataFrame | None = None,
                test_groups: pd.Series | None = None) -> np.ndarray:
    """Risk for each row from a model that never saw rows of its own group. With `test`, risk for each
    test row from a model trained on the training rows of every other group."""
    y = (train["shipped"] != train["truth"]).astype(int)
    target, tgroups = (train, groups) if test is None else (test, test_groups)
    risk = np.full(len(target), np.nan)
    for g in tgroups.unique():
        fit = groups != g
        m = make_model().fit(train.loc[fit, features], y[fit])
        here = (tgroups == g).to_numpy()
        risk[here] = m.predict_proba(target.loc[here, features])[:, 1]
    return risk


def auc_interval(y: np.ndarray, risk: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float, float]:
    """AUC and a bootstrap 95% interval (resampling sets)."""
    boots = []
    for _ in range(n_boot):
        b = rng.integers(len(y), size=len(y))
        if 0 < y[b].sum() < len(b):
            boots.append(roc_auc_score(y[b], risk[b]))
    return roc_auc_score(y, risk), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def risk_coverage(y: np.ndarray, risk: np.ndarray) -> pd.DataFrame:
    """Keep the lowest-risk sets first: at each coverage, the error rate among the sets kept."""
    order = np.argsort(risk, kind="stable")
    kept = np.arange(1, len(y) + 1)
    return pd.DataFrame({"coverage": kept / len(y), "error": np.cumsum(y[order]) / kept})


def plot_error_model(curves: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ink, muted, grid = "#1f2328", "#6e7781", "#d0d7de"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (title, series) in zip(axes, curves.items()):
        for (label, rc), color in zip(series, ("#2a78d6", "#eb6834")):
            ax.plot(rc["coverage"], rc["error"], color=color, lw=2, label=label)
        overall = series[0][1]["error"].iloc[-1]
        ax.axhline(overall, color=muted, lw=1.2, ls="--", label=f"no abstaining ({overall:.0%} wrong)")
        ax.set_title(title, loc="left", fontsize=10.5, color=ink)
        ax.set_xlabel("sets the band still reports (lowest risk first)", fontsize=9, color=muted)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0.05, 1)
        ax.grid(color=grid, lw=0.6)
        ax.tick_params(colors=muted, labelsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylabel("of those, counted wrong", fontsize=9, color=muted)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Count-error model: train on spliced sets, check on real sets.")
    parser.add_argument("--detector", default="v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sets", type=int, default=3000)
    parser.add_argument("--recompute", action="store_true", help="recompute symptoms instead of reading the cache")
    args = parser.parse_args()
    usb_axis = json.loads(USB_AXIS_FILE.read_text())["axis"]
    sp_path = REPORTS_DIR / f"error_rows_spliced_{args.seed}_{args.detector}.csv"
    re_path = REPORTS_DIR / f"error_rows_real_{args.detector}.csv"
    if args.recompute or not sp_path.exists():
        spliced_rows(args.seed, args.sets, usb_axis, args.detector).to_csv(sp_path, index=False)
    if args.recompute or not re_path.exists():
        real_rows(usb_axis, args.detector).to_csv(re_path, index=False)
    sp, real = pd.read_csv(sp_path), pd.read_csv(re_path)
    rng = np.random.default_rng(args.seed)
    y_sp = (sp["shipped"] != sp["truth"]).to_numpy(int)
    y_re = (real["shipped"] != real["truth"]).to_numpy(int)
    # real sets of a library session are predicted without spliced sets built from that session
    real_group = real["session"].where(real["session"].isin(LIBRARY), "none")
    print(f"spliced: {len(sp)} sets, {y_sp.sum()} counted wrong ({y_sp.mean():.0%}) | "
          f"real: {len(real)} curl sets, {y_re.sum()} counted wrong")

    rows, curves = [], {}
    for name, feats in [("core (main, one run)", CORE)] + [(f"{f} alone", [f]) for f in CORE] + \
                       [("all 15 (exploratory)", list(SYMPTOMS.values()))]:
        r_sp = out_of_fold(sp, feats, sp["source"])
        r_re = out_of_fold(sp, feats, sp["source"], real, real_group)
        a_sp, a_re = auc_interval(y_sp, r_sp, rng), auc_interval(y_re, r_re, rng)
        rc_sp, rc_re = risk_coverage(y_sp, r_sp), risk_coverage(y_re, r_re)
        rows.append({"features": name, "spliced AUC": f"{a_sp[0]:.2f} ({a_sp[1]:.2f}-{a_sp[2]:.2f})",
                     "spliced error at 80% kept": f"{rc_sp.loc[rc_sp.coverage.sub(0.8).abs().idxmin(), 'error']:.1%}",
                     "real AUC": f"{a_re[0]:.2f} ({a_re[1]:.2f}-{a_re[2]:.2f})",
                     "real error at 80% kept": f"{rc_re.loc[rc_re.coverage.sub(0.8).abs().idxmin(), 'error']:.1%}"})
        if name.startswith("core"):
            curves = {f"Spliced sets, held out by source session (n = {len(sp)})": [("core model", rc_sp)],
                      f"Real curl sets, p1-p5 (n = {len(real)}, {y_re.sum()} wrong)": [("core model", rc_re)]}
            real["risk"] = r_re
    print(pd.DataFrame(rows).to_string(index=False))
    real["rank"] = real["risk"].rank(ascending=False).astype(int)
    print("\nreal sets counted wrong, and where the core model ranked them (1 = riskiest of "
          f"{len(real)}):")
    print(real.loc[y_re == 1, ["person", "session", "set_idx", "condition", "flagged", "truth", "shipped", "risk", "rank"]]
          .round(3).to_string(index=False))
    fit = make_model().fit(sp[CORE], y_sp)
    print("\ncore model weights (+ = more likely wrong):")
    print(pd.Series(fit[-1].coef_[0], index=CORE).sort_values(key=abs, ascending=False).round(2).to_string())
    plot = REPORTS_DIR / f"error_model_{args.detector}.png"
    plot_error_model(curves, plot)
    print(f"plot: {plot.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
