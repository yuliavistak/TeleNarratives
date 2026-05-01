#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

NO_NARRATIVE = "Не містить наративу"


def clean_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s = s.replace(r"^\s*$", np.nan, regex=True)
    s = s.astype("string")
    s = s.str.strip()
    return s


def to_binary(s: pd.Series) -> pd.Series:
    s = clean_series(s)
    lower = s.str.casefold()
    no_mask = lower == NO_NARRATIVE.casefold()
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out[no_mask] = "no_narrative"
    out[~no_mask & s.notna()] = "narrative"
    return out


def cohen_pair(a: pd.Series, b: pd.Series):
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n == 0:
        return np.nan, 0
    return float(cohen_kappa_score(a[mask], b[mask])), n


def agreement_pair(a: pd.Series, b: pd.Series):
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n == 0:
        return np.nan, 0
    return float((a[mask] == b[mask]).mean()), n


def fleiss_kappa_for_three(df3: pd.DataFrame):
    mask = df3.notna().all(axis=1)
    df3 = df3[mask]
    N = df3.shape[0]
    n = df3.shape[1]
    if N == 0:
        return np.nan, 0
    labels = pd.unique(df3.to_numpy().ravel())
    labels = [x for x in labels if pd.notna(x)]
    labels = sorted(labels, key=lambda x: str(x))
    k = len(labels)
    if k == 0:
        return np.nan, N
    idx = {lab: i for i, lab in enumerate(labels)}
    counts = np.zeros((N, k), dtype=int)
    for i, row in enumerate(df3.itertuples(index=False)):
        for lab in row:
            counts[i, idx[lab]] += 1
    P_i = (counts * (counts - 1)).sum(axis=1) / (n * (n - 1))
    P_bar = P_i.mean()
    p_j = counts.sum(axis=0) / (N * n)
    P_e = (p_j ** 2).sum()
    if P_e == 1:
        kappa = np.nan
    else:
        kappa = (P_bar - P_e) / (1 - P_e)
    return float(kappa), N


def all_three_agreement(df3: pd.DataFrame):
    mask = df3.notna().all(axis=1)
    df3 = df3[mask]
    N = df3.shape[0]
    if N == 0:
        return np.nan, 0
    agree = (df3.nunique(axis=1) == 1).mean()
    return float(agree), N


def krippendorff_alpha_nominal(df3: pd.DataFrame):
    # Uses all available ratings per item; items with <2 ratings are ignored in Do.
    total_counts = {}
    Do_num = 0
    Do_den = 0
    n_items_used = 0
    total_ratings = 0

    for row in df3.itertuples(index=False):
        vals = [v for v in row if pd.notna(v)]
        n_i = len(vals)
        if n_i == 0:
            continue
        total_ratings += n_i
        for v in vals:
            total_counts[v] = total_counts.get(v, 0) + 1
        if n_i < 2:
            continue
        n_items_used += 1
        item_counts = {}
        for v in vals:
            item_counts[v] = item_counts.get(v, 0) + 1
        for n_ic in item_counts.values():
            Do_num += n_ic * (n_i - n_ic)
        Do_den += n_i * (n_i - 1)

    if Do_den == 0:
        return np.nan, n_items_used
    Do = Do_num / Do_den

    N = total_ratings
    if N < 2:
        return np.nan, n_items_used
    De_num = 0
    for n_c in total_counts.values():
        De_num += n_c * (N - n_c)
    De_den = N * (N - 1)
    De = De_num / De_den
    if De == 0:
        return np.nan, n_items_used
    alpha = 1 - (Do / De)
    return float(alpha), n_items_used


def compute_case(case_name: str, df3: pd.DataFrame, rnames: list[str]):
    rows = []
    a, b, c = (df3[rnames[0]], df3[rnames[1]], df3[rnames[2]])

    for (x_name, y_name, x, y) in [
        (rnames[0], rnames[1], a, b),
        (rnames[0], rnames[2], a, c),
        (rnames[1], rnames[2], b, c),
    ]:
        kappa, n = cohen_pair(x, y)
        rows.append(
            {
                "case": case_name,
                "metric": "cohen_kappa",
                "pair": f"{x_name} vs {y_name}",
                "value": kappa,
                "n_items": n,
            }
        )
        agree, n_ag = agreement_pair(x, y)
        rows.append(
            {
                "case": case_name,
                "metric": "pairwise_agreement",
                "pair": f"{x_name} vs {y_name}",
                "value": agree,
                "n_items": n_ag,
            }
        )

    cohen_vals = [r["value"] for r in rows if r["metric"] == "cohen_kappa" and r["pair"]]
    mean_cohen = float(np.nanmean(cohen_vals)) if len(cohen_vals) else np.nan
    rows.append(
        {
            "case": case_name,
            "metric": "cohen_kappa_mean",
            "pair": "",
            "value": mean_cohen,
            "n_items": int(df3.notna().all(axis=1).sum()),
        }
    )

    fleiss, n_f = fleiss_kappa_for_three(df3[rnames])
    rows.append(
        {
            "case": case_name,
            "metric": "fleiss_kappa",
            "pair": "",
            "value": fleiss,
            "n_items": n_f,
        }
    )

    all3, n_all3 = all_three_agreement(df3[rnames])
    rows.append(
        {
            "case": case_name,
            "metric": "all_three_agreement",
            "pair": "",
            "value": all3,
            "n_items": n_all3,
        }
    )

    alpha, n_alpha = krippendorff_alpha_nominal(df3[rnames])
    rows.append(
        {
            "case": case_name,
            "metric": "krippendorff_alpha",
            "pair": "",
            "value": alpha,
            "n_items": n_alpha,
        }
    )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement metrics.")
    parser.add_argument(
        "--input",
        default="labeling - 3 reviewers.csv",
        help="Input CSV with reviewer columns.",
    )
    parser.add_argument(
        "--output",
        default="inter_annotator_agreement_metrics.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")

    bin_df = pd.DataFrame(
        {
            "Reviewer 1": to_binary(df["Reviewer 1"]),
            "Reviewer 2": to_binary(df["Reviewer 2"]),
            "Reviewer 3": to_binary(df["Reviewer 3"]),
        }
    )

    multi_df = pd.DataFrame(
        {
            "Reviewer 1": clean_series(df["Reviewer 1"]),
            "Reviewer 2": clean_series(df["Reviewer 2"]),
            "Reviewer 3": clean_series(df["Reviewer 3"]),
        }
    )

    lvl1_df = pd.DataFrame(
        {
            "Reviewer 1 (level-1)": clean_series(df["Reviewer 1 (level-1)"]),
            "Reviewer 2 (level-1)": clean_series(df["Reviewer 2 (level-1)"]),
            "Reviewer 3 (level-1)": clean_series(df["Reviewer 3 (level-1)"]),
        }
    )

    rows = []
    rows += compute_case("binary", bin_df, ["Reviewer 1", "Reviewer 2", "Reviewer 3"])
    rows += compute_case("multiclass", multi_df, ["Reviewer 1", "Reviewer 2", "Reviewer 3"])
    rows += compute_case(
        "multiclass_level_1",
        lvl1_df,
        ["Reviewer 1 (level-1)", "Reviewer 2 (level-1)", "Reviewer 3 (level-1)"],
    )

    out = pd.DataFrame(rows)
    out["value"] = out["value"].round(6)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
