from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ------------------- CONFIG -------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # TeleNarratives/

TASK = "llm few-shot prompting"
EXPERIMENT = "best_gemini_400"

GOLD_CSV = "labeling - full labeling (short).csv"
PRED_CSV = _PROJECT_ROOT / "data" / "labeling results" / TASK / f"labeled_messages_{EXPERIMENT}.csv"

RESULTS_CSV = f"evaluation_results_{EXPERIMENT}.csv"

ID_COL = "message_id"

GOLD_GENERAL_COL = "General label"
GOLD_LEVEL1_COL = "General label level-1"
GOLD_BINARY_COL = "Binary label"

PRED_LABEL_COL = "narrative_id"
PRED_CONF_COL = "confidence"  # set to None to skip AUC metrics

NARRATIVE_MAP_CSV = "labeling - sub-narratives.csv"
MAP_ID_COL = "narrative_id"
MAP_LABEL_COL = "narrative"
MAP_LEVEL1_COL = "meta_narrative"
# ---------------------------------------------


def _load_mapping():
    df = pd.read_csv(NARRATIVE_MAP_CSV)
    if MAP_ID_COL not in df.columns or MAP_LABEL_COL not in df.columns:
        raise ValueError(
            f"NARRATIVE_MAP_CSV must have columns: {MAP_ID_COL}, {MAP_LABEL_COL}"
        )
    return dict(zip(df[MAP_ID_COL], df[MAP_LABEL_COL]))


def _load_level1_mapping():
    df = pd.read_csv(NARRATIVE_MAP_CSV)
    if MAP_ID_COL not in df.columns or MAP_LEVEL1_COL not in df.columns:
        raise ValueError(
            f"NARRATIVE_MAP_CSV must have columns: {MAP_ID_COL}, {MAP_LEVEL1_COL}"
        )
    df = df.dropna(subset=[MAP_ID_COL])
    df = df.drop_duplicates(subset=[MAP_ID_COL], keep="first")
    return dict(zip(df[MAP_ID_COL], df[MAP_LEVEL1_COL]))


def eval_binary(df):
    y_true = df["gold_binary"].astype(bool)
    y_pred = df["pred_binary"].astype(bool)

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    print(cm)
    tn, fp, fn, tp = cm.ravel()

    mcc = matthews_corrcoef(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    acc = accuracy_score(y_true, y_pred)

    result = {
        "task": "binary",
        "subset": "all",
        "support": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "mcc": float(mcc),
        "balanced_accuracy": float(bacc),
        "macro_f1": None,
        "weighted_f1": None,
        "accuracy": float(acc),
        "hamming_loss": None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pr_auc": None,
        "roc_auc": None,
    }

    if PRED_CONF_COL and PRED_CONF_COL in df.columns:
        scores = pd.to_numeric(df[PRED_CONF_COL], errors="coerce")
        mask = scores.notna()
        if mask.any():
            scores = scores[mask]
            y_true_scores = y_true[mask]
            try:
                result["pr_auc"] = float(average_precision_score(y_true_scores, scores))
            except Exception:
                pass
            try:
                result["roc_auc"] = float(roc_auc_score(y_true_scores, scores))
            except Exception:
                pass
    return result


def eval_multiclass(y_true, y_pred, title, subset_name, task="narrative_match"):
    acc = accuracy_score(y_true, y_pred)
    hamming = (y_true != y_pred).mean()
    mcc = matthews_corrcoef(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    result = {
        "task": task,
        "subset": subset_name,
        "support": int(len(y_true)),
        "positive_rate": None,
        "precision": None,
        "recall": None,
        "specificity": None,
        "f1": None,
        "mcc": float(mcc),
        "balanced_accuracy": float(bacc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "accuracy": float(acc),
        "hamming_loss": float(hamming),
        "tn": None,
        "fp": None,
        "fn": None,
        "tp": None,
        "pr_auc": None,
        "roc_auc": None,
    }
    print(f"=== {title} ===")
    print(
        classification_report(
            y_true, y_pred, zero_division=0, digits=4
        )
    )
    print()
    return result


def per_label_pr_auc(y_true, y_pred, scores, subset_name, task="narrative_match"):
    if scores is None:
        return []
    scores = pd.to_numeric(scores, errors="coerce").fillna(0.0)
    labels = sorted(pd.unique(pd.concat([y_true, y_pred], ignore_index=True)))
    rows = []
    for label in labels:
        y_true_bin = (y_true == label).astype(int)
        positives = int(y_true_bin.sum())
        negatives = int(len(y_true_bin) - positives)
        if positives == 0 or negatives == 0:
            pr_auc = None
        else:
            print(scores)
            print('\n')
            print('==================================')
            print('\n')
            y_score = scores.where(y_pred == label, 0.0)
            print(y_score)
            try:
                pr_auc = float(average_precision_score(y_true_bin, y_score))
            except Exception:
                pr_auc = None
        rows.append(
            {
                "task": task,
                "subset": subset_name,
                "label": label,
                "support": int(len(y_true_bin)),
                "positives": positives,
                "negatives": negatives,
                "pr_auc": pr_auc,
            }
        )
    return rows


def main():
    gold = pd.read_csv(GOLD_CSV)
    pred = pd.read_csv(PRED_CSV)

    mapping = _load_mapping()
    level1_mapping = _load_level1_mapping()

    df = gold.merge(pred, on=ID_COL, how="inner")

    # Binary labels
    df["gold_binary"] = df[GOLD_BINARY_COL]
    df["pred_binary"] = df[PRED_LABEL_COL].notna()

    # Narrative labels
    df["gold_label"] = df.apply(
        lambda r: r[GOLD_GENERAL_COL] if r[GOLD_BINARY_COL] else "NONE", axis=1
    )
    df["pred_label"] = df[PRED_LABEL_COL].map(mapping).fillna("NONE")

    df["gold_level1_label"] = df.apply(
        lambda r: r[GOLD_LEVEL1_COL] if r[GOLD_BINARY_COL] else "NONE", axis=1
    )
    df["pred_level1_label"] = df[PRED_LABEL_COL].map(level1_mapping).fillna("NONE")

    print(f"Merged rows: {len(df)}")
    print(f"Gold rows: {len(gold)}, Pred rows: {len(pred)}")
    print()

    results = [eval_binary(df)]

    results.append(
        eval_multiclass(
            df["gold_label"],
            df["pred_label"],
            "Narrative match (all rows)",
            "all",
        )
    )

    results.append(
        eval_multiclass(
            df["gold_level1_label"],
            df["pred_level1_label"],
            "Meta-narrative match (all rows)",
            "all",
            task="meta_narrative_match",
        )
    )

    mask = df["gold_binary"].astype(bool)
    if mask.any():
        results.append(
            eval_multiclass(
                df.loc[mask, "gold_label"],
                df.loc[mask, "pred_label"],
                "Narrative match (gold narrative only)",
                "gold_narrative_only",
            )
        )

        results.append(
            eval_multiclass(
                df.loc[mask, "gold_level1_label"],
                df.loc[mask, "pred_level1_label"],
                "Meta-narrative match (gold narrative only)",
                "gold_narrative_only",
                task="meta_narrative_match",
            )
        )

    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"Saved results table to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
